# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main Trainer class for CoTracker using pure torch.distributed.
Replaces PyTorch Lightning Lite with explicit DDP control.
"""

import os
import json
import logging
import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.amp import GradScaler
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# CoTracker imports
from functools import partial
from cotracker.datasets.utils import collate_fn_train, dataclass_to_cuda_
from cotracker.evaluation.core.evaluator import Evaluator

# Training imports
from training.distributed import (
    init_distributed,
    cleanup,
    is_main_process,
    get_rank,
    barrier,
    set_seeds,
    seed_worker,
)
from training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    save_final_model,
    find_latest_checkpoint,
    cleanup_old_checkpoints,
    cleanup_checkpoints_by_metric,
)
from training.eval import get_eval_dataloader, run_eval
from training.logging_utils import create_logger
from training.forward import forward_batch, compute_total_loss, create_forward_config
from training.data.composed_dataset import create_dataloader
from training.data.infinite_loader import InfiniteBatchIterator

class Trainer:
    """
    CoTracker Trainer with explicit torch.distributed control.

    Handles:
    - DDP initialization and model wrapping
    - Mixed precision training with GradScaler
    - Gradient accumulation with model.no_sync()
    - Checkpoint save/load with legacy format support
    - Wandb logging
    - Evaluation during training

    Subclass and override forward_batch() for custom forward passes.
    """

    supports_no_sync = True

    def __init__(self, cfg: DictConfig, eval_only: bool = False):
        """
        Initialize trainer.

        Args:
            cfg: Hydra configuration
            eval_only: If True, skip distributed setup and only prepare for evaluation
        """
        self.cfg = cfg
        self.eval_only = eval_only

        if eval_only:
            # Minimal setup for eval-only mode
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Set environment variables
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

            # Initialize distributed
            self.rank, self.world_size, self.local_rank, self.device = init_distributed(
                backend=cfg.distributed.backend,
                timeout_mins=cfg.distributed.timeout_mins,
            )

        # Set seeds first, then configure CUDA (config is the authority)
        set_seeds(cfg.seed, rank=self.rank)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cfg.cuda.cudnn_deterministic
            torch.backends.cudnn.benchmark = cfg.cuda.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cfg.cuda.allow_tf32
            torch.backends.cudnn.allow_tf32 = cfg.cuda.allow_tf32

        # Initialize components (will be set in setup())
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.batch_iterator = None
        self.logger = None
        self.evaluator = None
        self.teacher = None
        self.eval_dataloaders = []
        self.final_dataloaders = []
        self.profiler = None

        # Training state
        self.total_steps = 0
        self.epoch = 0
        self.batches_in_epoch = 0  # For mid-epoch resume
        self.loss_history = []  # For test observability

        # Forward config
        self.forward_cfg = create_forward_config(cfg)

    def setup(self):
        """Set up all training components."""
        cfg = self.cfg

        # Create directories
        Path(cfg.exp_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.checkpoint.save_dir).mkdir(parents=True, exist_ok=True)

        # Save config
        if is_main_process():
            config_path = Path(cfg.exp_dir) / "config.yaml"
            with open(config_path, "w") as f:
                f.write(OmegaConf.to_yaml(cfg))

            meta_path = Path(cfg.exp_dir) / "meta.json"
            with open(meta_path, "w") as f:
                json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

        # Build model
        self._build_model()

        # Build teacher ensemble (pseudo-label mode only)
        self._build_teacher()

        # Build optimizer and scheduler
        self._build_optimizer()

        # Build scaler for AMP
        self._build_scaler()

        # Build dataloader
        self._build_dataloader()

        # Load checkpoint if resuming (sets self.epoch, self.batches_in_epoch, self.total_steps)
        self._load_checkpoint()

        # Build infinite batch iterator (uses resume state from checkpoint)
        self._build_batch_iterator()

        # Build logger AFTER checkpoint load (so we can pass resume metadata)
        self._build_logger()

        # Compile model AFTER checkpoint load (clean state_dict keys) but BEFORE DDP
        self._maybe_compile()

        # Wrap model with DDP (after checkpoint loading and compilation!)
        self._wrap_ddp()

        # Set up evaluation (only on main process)
        if is_main_process():
            self._setup_evaluation()

        barrier()

        # Validate and log setup
        self._validate_and_log_setup()

        # Build profiler after setup (optional, rank 0 only)
        self._build_profiler()

        logging.info(f"Trainer setup complete. Rank {self.rank}/{self.world_size}")

    def setup_eval_only(self):
        """
        Minimal setup for evaluation-only mode.
        
        Only builds the model, loads checkpoint, and sets up evaluation.
        Skips optimizer, scheduler, scaler, train dataloader, and DDP.
        """
        from training.logging_utils import DummyLogger
        
        cfg = self.cfg
        
        # Create directories
        Path(cfg.exp_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.checkpoint.save_dir).mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Eval visualizations will be saved to: {cfg.exp_dir}/eval/")
        
        # Build model
        self._build_model()
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Use dummy logger (no wandb)
        self.logger = DummyLogger()
        
        # Set up evaluation
        self._setup_evaluation()
        
        self.model.eval()
        
        logging.info("Eval-only setup complete")

    def _validate_and_log_setup(self):
        """Validate setup and log configuration summary."""
        cfg = self.cfg

        if not is_main_process():
            return

        logging.info("\n" + "=" * 60)
        logging.info("TRAINING CONFIGURATION SUMMARY")
        logging.info("=" * 60)

        # Experiment info
        logging.info(f"\n[Experiment]")
        logging.info(f"  Name: {cfg.exp_name}")
        logging.info(f"  Dir: {cfg.exp_dir}")
        logging.info(f"  Seed: {cfg.seed}")

        # Distributed info
        logging.info(f"\n[Distributed]")
        logging.info(f"  World size: {self.world_size}")
        logging.info(f"  Backend: {cfg.distributed.backend}")
        logging.info(f"  Device: {self.device}")

        # Model info
        logging.info(f"\n[Model]")
        model = self.model.module if hasattr(self.model, 'module') else self.model
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"  Type: {type(model).__name__}")
        logging.info(f"  Total params: {total_params:,}")
        logging.info(f"  Trainable params: {trainable_params:,}")

        # Check model dtype
        param = next(model.parameters())
        logging.info(f"  Param dtype: {param.dtype}")
        logging.info(f"  Param device: {param.device}")

        # Mixed precision info
        logging.info(f"\n[Mixed Precision]")
        logging.info(f"  Enabled: {cfg.training.mixed_precision}")
        logging.info(f"  Precision: {cfg.training.precision}")
        amp_dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float16
        logging.info(f"  AMP dtype: {amp_dtype}")
        logging.info(f"  GradScaler enabled: {self.scaler.is_enabled()}")

        # Check bf16 support
        if cfg.training.precision == "bf16":
            bf16_supported = torch.cuda.is_bf16_supported()
            logging.info(f"  BF16 hardware support: {bf16_supported}")
            if not bf16_supported:
                logging.warning("  WARNING: BF16 requested but not supported by hardware!")

        # Optimizer info
        logging.info(f"\n[Optimizer]")
        logging.info(f"  Type: {type(self.optimizer).__name__}")
        logging.info(f"  LR: {cfg.optimizer.lr}")
        logging.info(f"  Weight decay: {cfg.optimizer.weight_decay}")
        logging.info(f"  Scheduler: {type(self.scheduler).__name__}")

        # Training info
        accum_steps = cfg.training.gradient_accumulation_steps
        effective_batch = cfg.training.batch_size * self.world_size * accum_steps
        logging.info(f"\n[Training]")
        logging.info(f"  Num steps: {cfg.training.num_steps}")
        logging.info(f"  Batch size: {cfg.training.batch_size}")
        logging.info(f"  Gradient accumulation steps: {accum_steps}")
        logging.info(f"  Effective batch size: {effective_batch} (batch_size * world_size * accum_steps)")
        logging.info(f"  Sequence length: {cfg.training.sequence_len}")
        logging.info(f"  Traj per sample: {cfg.training.traj_per_sample}")
        logging.info(f"  Train iters: {cfg.training.train_iters}")
        logging.info(f"  Gradient clip norm: {cfg.training.gradient_clip_norm}")
        logging.info(f"  Torch compile: {cfg.training.torch_compile}")

        # Data info
        logging.info(f"\n[Data]")
        logging.info(f"  Train loader batches: {len(self.train_loader)}")
        logging.info(f"  Num workers: {cfg.training.num_workers}")

        # Checkpoint info
        logging.info(f"\n[Checkpoint]")
        logging.info(f"  Save dir: {cfg.checkpoint.save_dir}")
        logging.info(f"  Resumed from step: {self.total_steps}")
        logging.info(f"  Resumed from epoch: {self.epoch}")
        logging.info(f"  Current LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        logging.info(f"  Scheduler last_epoch: {self.scheduler.last_epoch}")

        # Save frequency
        if cfg.checkpoint.get("save_every_n_steps", None):
            logging.info(f"  Save mode: step-based (every {cfg.checkpoint.save_every_n_steps} steps)")
            if cfg.checkpoint.get("save_every_n_epoch", None):
                logging.warning(f"  WARNING: save_every_n_epoch={cfg.checkpoint.save_every_n_epoch} ignored (step-based is set)")
        elif cfg.checkpoint.get("save_every_n_epoch", None):
            logging.info(f"  Save mode: epoch-based (every {cfg.checkpoint.save_every_n_epoch} epochs)")
        else:
            logging.warning(f"  WARNING: No checkpoint saving configured!")

        # Eval frequency
        if cfg.checkpoint.get("evaluate_every_n_steps", None):
            logging.info(f"  Eval mode: step-based (every {cfg.checkpoint.evaluate_every_n_steps} steps)")
            if cfg.checkpoint.get("evaluate_every_n_epoch", None):
                logging.warning(f"  WARNING: evaluate_every_n_epoch={cfg.checkpoint.evaluate_every_n_epoch} ignored (step-based is set)")
        elif cfg.checkpoint.get("evaluate_every_n_epoch", None):
            logging.info(f"  Eval mode: epoch-based (every {cfg.checkpoint.evaluate_every_n_epoch} epochs)")
        else:
            logging.warning(f"  WARNING: No evaluation configured!")

        # CUDA info
        logging.info(f"\n[CUDA]")
        logging.info(f"  cuDNN deterministic: {torch.backends.cudnn.deterministic}")
        logging.info(f"  cuDNN benchmark: {torch.backends.cudnn.benchmark}")
        logging.info(f"  TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
        logging.info(f"  GPU: {torch.cuda.get_device_name(self.device)}")
        logging.info(f"  GPU memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB")

        # Sanity checks
        logging.info(f"\n[Sanity Checks]")
        checks_passed = True

        # Check 1: Model on correct device
        if param.device != self.device:
            logging.error(f"  FAIL: Model not on expected device ({param.device} vs {self.device})")
            checks_passed = False
        else:
            logging.info(f"  PASS: Model on correct device")

        # Check 2: Scaler matches mixed precision setting
        if cfg.training.mixed_precision and not self.scaler.is_enabled():
            # Note: scaler is disabled for bf16, that's expected
            if cfg.training.precision != "bf16":
                logging.error(f"  FAIL: Mixed precision enabled but scaler disabled")
                checks_passed = False
            else:
                logging.info(f"  PASS: Scaler disabled for bf16 (expected)")
        else:
            logging.info(f"  PASS: Scaler config consistent")

        # Check 3: Optimizer has params
        if len(self.optimizer.param_groups) == 0:
            logging.error(f"  FAIL: Optimizer has no param groups")
            checks_passed = False
        else:
            logging.info(f"  PASS: Optimizer has {len(self.optimizer.param_groups)} param group(s)")

        # Check 4: Dataloader not empty
        if len(self.train_loader) == 0:
            logging.error(f"  FAIL: Train loader is empty")
            checks_passed = False
        else:
            logging.info(f"  PASS: Train loader has {len(self.train_loader)} batches")

        # Check 5: DDP wrapping (if multi-gpu)
        if self.world_size > 1:
            if not isinstance(self.model, DDP):
                logging.error(f"  FAIL: Model not wrapped with DDP")
                checks_passed = False
            else:
                logging.info(f"  PASS: Model wrapped with DDP")

        logging.info("=" * 60)

        if not checks_passed:
            raise RuntimeError("Setup validation failed! Check logs above.")

        logging.info("All sanity checks passed!\n")

    def _build_model(self):
        """Build model from config."""
        logging.info("Building model...")
        self.model = instantiate(self.cfg.model, _recursive_=False)
        self.model.to(self.device)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {total_params:,}")

        # Freeze vis_conf_head if configured (pseudo-label mode)
        if self.cfg.training.get("freeze_vis_conf_head", False):
            frozen_count = 0
            for name, param in self.model.named_parameters():
                if "vis_conf_head" in name:
                    param.requires_grad = False
                    frozen_count += 1
            trainable_after = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            logging.info(
                f"Froze {frozen_count} vis_conf_head parameters. "
                f"Trainable params after freeze: {trainable_after:,}"
            )

        # torch.compile is deferred to _maybe_compile(), called after checkpoint loading

    def _build_teacher(self):
        """Build teacher ensemble. No-op in base Trainer; overridden in PseudoLabelTrainer."""
        pass

    def forward_batch(self, batch, model, cfg):
        """Run the forward pass for one micro-batch. Override in subclasses."""
        return forward_batch(batch, model, cfg)

    def _build_optimizer(self):
        """Build optimizer and scheduler."""
        logging.info("Building optimizer...")

        betas = tuple(self.cfg.optimizer.get("betas", [0.9, 0.999]))
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            betas=betas,
            weight_decay=self.cfg.optimizer.weight_decay,
            eps=self.cfg.optimizer.eps,
        )

        # OneCycleLR scheduler
        pct_start = self.cfg.scheduler.get("pct_start", 0.05)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.optimizer.lr,
            total_steps=self.cfg.training.num_steps + 100,
            pct_start=pct_start,
            cycle_momentum=False,
            anneal_strategy="cos",
        )

    def _build_scaler(self):
        """Build GradScaler for mixed precision.

        Note: GradScaler is only needed for fp16, not bf16.
        bf16 has larger dynamic range and doesn't need loss scaling.
        """
        cfg = self.cfg.training
        # Disable scaler for bf16 (not needed) or if mixed precision is off
        use_scaler = cfg.mixed_precision and cfg.precision != "bf16"
        self.scaler = GradScaler(enabled=use_scaler)
        logging.info(f"GradScaler enabled: {use_scaler} (precision={cfg.precision})")

    def _build_dataloader(self):
        """Build training dataloader."""
        logging.info("Building dataloader...")

        # Instantiate dataset
        train_dataset = instantiate(self.cfg.data.train, _recursive_=False)

        random_seq_len = self.cfg.training.get("random_seq_len", False)
        collate_fn = partial(collate_fn_train, random_seq_len=random_seq_len)

        self.train_loader = create_dataloader(
            dataset=train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
            distributed=self.world_size > 1,
            rank=self.rank,
            world_size=self.world_size,
            seed=self.cfg.seed,
        )

        logging.info(f"Train loader length: {len(self.train_loader)}")

    def _build_profiler(self):
        """Build optional torch profiler for memory debugging."""
        profiler_cfg = self.cfg.training.get("profiler", {})
        enabled = bool(profiler_cfg.get("enabled", False))
        if not enabled or not is_main_process():
            return

        start_step = int(profiler_cfg.get("start_step", 0))
        num_steps = int(profiler_cfg.get("num_steps", 5))
        warmup_steps = int(profiler_cfg.get("warmup_steps", 1))
        output_dir = profiler_cfg.get("output_dir", None)
        if output_dir is None:
            output_dir = os.path.join(self.cfg.exp_dir, "profiler")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self.profiler = profile(
            activities=activities,
            schedule=schedule(wait=start_step, warmup=warmup_steps, active=num_steps, repeat=1),
            on_trace_ready=tensorboard_trace_handler(output_dir),
            profile_memory=True,
            record_shapes=bool(profiler_cfg.get("record_shapes", True)),
            with_stack=bool(profiler_cfg.get("with_stack", False)),
            with_flops=bool(profiler_cfg.get("with_flops", False)),
        )
        self.profiler.start()
        logging.info(
            f"Profiler enabled: output_dir={output_dir}, start_step={start_step}, "
            f"warmup_steps={warmup_steps}, num_steps={num_steps}"
        )

    def _build_batch_iterator(self):
        """Build infinite batch iterator with resume state from checkpoint.

        Must be called AFTER _load_checkpoint() so self.epoch and
        self.batches_in_epoch are populated from the checkpoint.
        """
        self.batch_iterator = InfiniteBatchIterator(
            dataloader=self.train_loader,
            start_epoch=self.epoch,
            batches_to_skip=self.batches_in_epoch,
        )
        logging.info(
            f"Built InfiniteBatchIterator: start_epoch={self.epoch}, "
            f"batches_to_skip={self.batches_in_epoch}"
        )

    def _build_logger(self):
        """Build logger with resume support.

        Must be called AFTER _load_checkpoint() so we can pass resume metadata.
        Resolves log_dir: null in config means {checkpoint.save_dir}/logs.
        """
        # Determine checkpoint path for resume metadata
        ckpt_path = find_latest_checkpoint(self.cfg.checkpoint.save_dir)
        if ckpt_path is None and self.cfg.checkpoint.resume_from is not None:
            ckpt_path = self.cfg.checkpoint.resume_from

        # Resolve log_dir: null means colocate with checkpoints
        log_dir = self.cfg.logging.log_dir
        if log_dir is None:
            log_dir = os.path.join(self.cfg.checkpoint.save_dir, "logs")

        self.logger = create_logger(
            project=self.cfg.logging.wandb_project,
            name=self.cfg.exp_name,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            entity=self.cfg.logging.wandb_entity,
            model=self.model,
            rank=self.rank,
            log_dir=log_dir,
            local_logging=True,
            use_wandb=self.cfg.logging.use_wandb,
            resume="allow",  # Auto-resume if run exists (for preemption)
            saved_step=self.total_steps,
            ckpt_path=ckpt_path,
        )

    def _restore_rng_states(self, rng_states: dict):
        """Restore all RNG states for deterministic resume."""
        import random
        import numpy as np

        random.setstate(rng_states["python"])
        np.random.set_state(rng_states["numpy"])
        torch.set_rng_state(rng_states["torch"])
        if rng_states.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_states["cuda"])
        logging.info("Restored RNG states for deterministic resume")

    def _load_checkpoint(self):
        """Load checkpoint if resuming or restoring."""
        cfg = self.cfg.checkpoint

        # For eval-only mode, skip auto-resume and use restore_ckpt directly
        if self.eval_only:
            if cfg.restore_ckpt is not None:
                logging.info(f"Loading checkpoint for evaluation: {cfg.restore_ckpt}")
                load_checkpoint(
                    cfg.restore_ckpt,
                    self.model,
                    optimizer=None,
                    scheduler=None,
                    scaler=None,
                    strict=False,
                )
            else:
                logging.warning("No checkpoint specified (checkpoint.restore_ckpt=null), evaluating random weights")
            return

        # Check for existing checkpoint in save_dir (auto-resume)
        latest_ckpt = find_latest_checkpoint(cfg.save_dir)
        if latest_ckpt is not None:
            logging.info(f"Found existing checkpoint: {latest_ckpt}")
            self.total_steps, self.epoch, self.batches_in_epoch, rng_states = load_checkpoint(
                latest_ckpt,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                strict=False,
            )
            if rng_states is not None:
                self._restore_rng_states(rng_states)
            return

        # Check for explicit resume path
        if cfg.resume_from is not None:
            logging.info(f"Resuming from: {cfg.resume_from}")
            self.total_steps, self.epoch, self.batches_in_epoch, rng_states = load_checkpoint(
                cfg.resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                strict=False,
            )
            if rng_states is not None:
                self._restore_rng_states(rng_states)
            return

        # Check for pretrained weights only
        if cfg.restore_ckpt is not None:
            logging.info(f"Loading pretrained weights from: {cfg.restore_ckpt}")
            load_checkpoint(
                cfg.restore_ckpt,
                self.model,
                optimizer=None,  # Don't load optimizer
                scheduler=None,
                scaler=None,
                strict=False,
            )

    def _maybe_compile(self):
        """Apply torch.compile after checkpoint loading but before DDP wrapping.

        Must be called AFTER _load_checkpoint() so we load clean state dicts,
        and BEFORE _wrap_ddp() so DDP wraps the compiled module.
        """
        if not self.cfg.training.torch_compile:
            return

        import torch._dynamo
        torch._dynamo.config.optimize_ddp = False

        logging.info("Compiling model with torch.compile(fullgraph=False)...")
        self.model = torch.compile(self.model, fullgraph=False)
        logging.info("Model compilation complete.")

    def _wrap_ddp(self):
        """Wrap model with DistributedDataParallel."""
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.cfg.distributed.find_unused_parameters,
                gradient_as_bucket_view=self.cfg.distributed.gradient_as_bucket_view,
                bucket_cap_mb=self.cfg.distributed.bucket_cap_mb,
                broadcast_buffers=self.cfg.distributed.broadcast_buffers,
            )

    def _setup_evaluation(self):
        """Set up evaluation components (main process only)."""
        cfg = self.cfg

        # Evaluator (base dir — run_eval creates step-specific dirs)
        self.evaluator = Evaluator(cfg.exp_dir)

        # Evaluation dataloaders
        for ds_name in cfg.evaluation.datasets:
            self.eval_dataloaders.append(
                (ds_name, get_eval_dataloader(cfg.evaluation.dataset_root, ds_name))
            )
        
        # Print out dataset sizes for logging
        for ds_name, dataloader in self.eval_dataloaders:
            logging.info(f"Eval dataloader for {ds_name}: {len(dataloader.dataset)} samples, {len(dataloader)} batches")

        # Final datasets are loaded lazily in _finish_training() to save memory
        self._final_dataset_names = list(cfg.evaluation.final_datasets)
        logging.info(f"Final eval datasets (loaded at end of training): {self._final_dataset_names}")

    def train(self):
        """Main training loop using InfiniteBatchIterator.

        Flat step-based loop: each iteration is one optimizer step.
        The batch iterator handles epoch cycling, gotit filtering, and
        sampler epoch management internally.
        """
        cfg = self.cfg
        model = self.model
        optimizer = self.optimizer
        optimizer.zero_grad(set_to_none=True)
        scheduler = self.scheduler
        scaler = self.scaler
        batch_iter = self.batch_iterator

        accum_steps = cfg.training.gradient_accumulation_steps

        model.train()
        amp_dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float16

        # Estimate total epochs for display
        batches_per_epoch = len(self.train_loader)
        steps_per_epoch = batches_per_epoch // accum_steps
        estimated_epochs = cfg.training.num_steps / steps_per_epoch if steps_per_epoch > 0 else 0

        total_pbar = tqdm(
            total=cfg.training.num_steps,
            desc=f"Total steps (~{estimated_epochs:.1f} epochs)",
            disable=not is_main_process(),
            position=0,
            initial=self.total_steps,
        )
        samples_seen = self.total_steps * accum_steps * cfg.training.batch_size * self.world_size

        self._prev_epoch = batch_iter.epoch

        if cfg.checkpoint.validate_at_start and self.total_steps == 0:
            if is_main_process():
                logging.info("Running initial validation before training starts...")
                self._evaluate(f"step_{self.total_steps:06d}")
            barrier()

        import time as _time
        _data_time_accum = 0.0
        _compute_time_accum = 0.0
        _timing_count = 0
        _sync_for_timing = bool(self.cfg.training.get("profiler", {}).get("enabled", False))

        for step in range(self.total_steps, cfg.training.num_steps):
            valid_micro_steps = 0
            output = None
            loss_accum = 0.0
            accumulated_losses = {}
            # --- Gradient accumulation micro-steps ---
            for micro in range(accum_steps):
                _t0 = _time.monotonic()
                batch = next(batch_iter)
                dataclass_to_cuda_(batch)
                if _sync_for_timing:
                    torch.cuda.synchronize()
                _t1 = _time.monotonic()

                assert model.training

                is_accumulating = micro < accum_steps - 1
                model_for_forward = model
                use_no_sync = (
                    is_accumulating
                    and self.world_size > 1
                    and self.supports_no_sync
                )
                sync_context = model.no_sync() if use_no_sync else contextlib.nullcontext()

                with sync_context:
                    with torch.amp.autocast('cuda', enabled=cfg.training.mixed_precision, dtype=amp_dtype):
                        output = self.forward_batch(batch, model_for_forward, self.forward_cfg)
                        if output is None:
                            continue
                        valid_micro_steps += 1
                        loss = compute_total_loss(output) / accum_steps
                        loss_accum += loss.detach().item()

                        for k, v in output.items():
                            if "loss" in v:
                                accumulated_losses[k] = accumulated_losses.get(k, 0.0) + v["loss"].detach().item()

                    scaler.scale(loss).backward()
                if _sync_for_timing:
                    torch.cuda.synchronize()
                _t2 = _time.monotonic()
                _data_time_accum += (_t1 - _t0)
                _compute_time_accum += (_t2 - _t1)

            # If last micro-step was invalid but earlier ones were valid,
            # gradients were accumulated under no_sync and never all-reduced.
            # Trigger an explicit all-reduce so all ranks stay consistent.
            if (
                valid_micro_steps > 0
                and valid_micro_steps < accum_steps
                and self.world_size > 1
                and self.supports_no_sync
            ):
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            _timing_count += 1
            if _timing_count % 20 == 0:
                _avg_data = _data_time_accum / 20
                _avg_compute = _compute_time_accum / 20
                logging.info(
                    f"[Timing] avg over last 20 steps: "
                    f"data={_avg_data:.2f}s, compute={_avg_compute:.2f}s, "
                    f"total={_avg_data + _avg_compute:.2f}s, "
                    f"data%={_avg_data / (_avg_data + _avg_compute) * 100:.0f}%"
                )
                _data_time_accum = 0.0
                _compute_time_accum = 0.0

            if valid_micro_steps == 0:
                optimizer.zero_grad(set_to_none=True)
                self.total_steps = step + 1
                samples_seen += accum_steps * cfg.training.batch_size * self.world_size
                total_pbar.update(1)
                total_pbar.set_postfix(
                    loss="skip",
                    epoch=batch_iter.epoch,
                    samples=f"{samples_seen:,}",
                )
                continue

            # --- Optimizer step (after all micro-steps) ---
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            self.total_steps = step + 1
            self.loss_history.append(loss_accum)
            samples_seen += accum_steps * cfg.training.batch_size * self.world_size
            total_pbar.update(1)
            total_pbar.set_postfix(
                loss=f"{loss_accum:.4f}",
                epoch=batch_iter.epoch,
                samples=f"{samples_seen:,}",
            )

            # Average individual losses across micro-steps
            if valid_micro_steps > 0:
                avg_component_losses = {k: v / valid_micro_steps for k, v in accumulated_losses.items()}
            else:
                avg_component_losses = {}

            # --- Periodic actions ---
            self._maybe_log_step(avg_component_losses, loss_accum, batch)
            self._maybe_save_checkpoint()
            self._maybe_evaluate()
            self._maybe_end_of_epoch()
            if self.profiler is not None:
                self.profiler.step()

        total_pbar.close()

        # Final save and evaluation
        if is_main_process():
            self._finish_training()

        if self.profiler is not None:
            self.profiler.stop()
            logging.info("Profiler stopped and traces flushed.")

        cleanup()

    # --- Periodic action wrappers (called by all ranks) ---

    def _maybe_log_step(self, avg_component_losses, loss, batch):
        """Log training metrics if it's time and we're rank 0."""
        log_every = self.cfg.logging.log_every_n_steps
        if not log_every or self.total_steps % log_every != 0:
            return
        if is_main_process():
            self._log_training_step(avg_component_losses, loss)

    def _log_training_step(self, avg_component_losses, loss):
        """Log training metrics averaged across micro-steps."""
        metrics = {}
        for k, v in avg_component_losses.items():
            metrics[f"train/{k}_loss"] = v
        metrics["train/total_loss"] = loss if isinstance(loss, float) else loss.item()
        metrics["train/lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["train/model_param_abs_mean"] = self._compute_model_param_abs_mean()

        self.logger.log(metrics, step=self.total_steps)

    def _compute_model_param_abs_mean(self) -> float:
        """Compute mean absolute value across all model parameters."""
        model = self.model.module if hasattr(self.model, "module") else self.model
        total_abs = 0.0
        total_numel = 0
        with torch.no_grad():
            for p in model.parameters():
                total_abs += p.detach().abs().sum().item()
                total_numel += p.numel()
        if total_numel == 0:
            return 0.0
        return total_abs / total_numel

    PINNED_STEPS = frozenset({15000})

    def _maybe_save_checkpoint(self):
        """Save checkpoint if it's time. Rank 0 saves, all ranks barrier."""
        save_every = self.cfg.checkpoint.save_every_n_steps
        is_scheduled = save_every and self.total_steps % save_every == 0
        is_pinned = self.total_steps in self.PINNED_STEPS
        if not is_scheduled and not is_pinned:
            return
        if is_main_process():
            self._save_step_checkpoint()
        barrier()
    
    def _save_step_checkpoint(self):
        """Save checkpoint at current step.

        Gets epoch and batches_in_epoch from the batch iterator so the
        checkpoint contains the correct resume position.
        """
        save_path = Path(self.cfg.checkpoint.save_dir) / f"step_{self.total_steps:06d}.pth"

        # Get iterator state for accurate resume
        iter_state = self.batch_iterator.state_dict()

        save_checkpoint(
            str(save_path),
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.total_steps,
            iter_state["epoch"],
            self.cfg,
            rank=self.rank,
            batches_in_epoch=iter_state["batches_in_epoch"],
        )
        logging.info(f"Saved step checkpoint: {save_path}")

        # Cleanup old checkpoints
        num_checkpoints = getattr(self.cfg.checkpoint, 'num_checkpoints', None)
        best_metric = getattr(self.cfg.checkpoint, 'best_metric', None)
        if best_metric is not None:
            eval_dir = str(Path(self.cfg.exp_dir) / "eval")
            metric_mode = getattr(self.cfg.checkpoint, 'best_metric_mode', 'max')
            cleanup_checkpoints_by_metric(
                self.cfg.checkpoint.save_dir,
                eval_dir,
                num_checkpoints,
                metric_key=best_metric,
                metric_mode=metric_mode,
                rank=self.rank,
            )
        else:
            cleanup_old_checkpoints(self.cfg.checkpoint.save_dir, num_checkpoints, rank=self.rank)

    def _maybe_evaluate(self):
        """Run evaluation if it's time. Rank 0 evaluates, all ranks barrier."""
        eval_every = self.cfg.checkpoint.evaluate_every_n_steps
        if not eval_every or self.total_steps % eval_every != 0:
            return
        if is_main_process():
            self._evaluate(eval_name=f"step_{self.total_steps:06d}")
        barrier()

    def _evaluate(self, eval_name=None, dataloaders=None):
        """Run evaluation.

        Args:
            eval_name: Subdirectory name under eval/. None saves directly to eval/.
            dataloaders: List of (name, dataloader) pairs. Defaults to self.eval_dataloaders.
        """
        if dataloaders is None:
            dataloaders = self.eval_dataloaders

        amp_dtype = torch.bfloat16 if self.cfg.training.precision == "bf16" else torch.float16
        eval_base = Path(self.cfg.exp_dir) / "eval"
        viz_dir = str(eval_base / eval_name) if eval_name else str(eval_base)
        mem_profile = self.cfg.evaluation.get("memory_profile", False)
        mem_profile_dir = str(eval_base / "memory_profile") if mem_profile else None
        run_eval(
            self.evaluator,
            self.model,  # run_eval handles unwrapping
            dataloaders,
            logger=self.logger,
            step=self.total_steps,
            query_random=(
                self.cfg.training.query_sampling_method is not None
                and "random" in self.cfg.training.query_sampling_method
            ),
            amp_dtype=amp_dtype,
            viz_dir=viz_dir,
            memory_profile_dir=mem_profile_dir,
            eval_protocol=self.cfg.evaluation.get("eval_protocol", "baseline"),
            offline_model=self.cfg.training.offline_model,
        )

        self.model.train()
        torch.cuda.empty_cache()

    def _maybe_end_of_epoch(self):
        """Handle epoch transition if one occurred."""
        current_epoch = self.batch_iterator.epoch
        if current_epoch == self._prev_epoch:
            return
        if is_main_process():
            self._end_of_epoch(self._prev_epoch)
        self._prev_epoch = current_epoch

    def _end_of_epoch(self, epoch):
        """Handle end of epoch: save checkpoint and evaluate.

        Note: Epoch-based save/eval is skipped if step-based is configured.
        """
        cfg = self.cfg

        # Save checkpoint (skip if step-based saving is enabled)
        if cfg.checkpoint.save_every_n_steps is None:
            if (epoch + 1) % cfg.checkpoint.save_every_n_epoch == 0:
                ckpt_iter = str(self.total_steps).zfill(6)
                save_path = Path(cfg.checkpoint.save_dir) / f"model_{cfg.exp_name}_{ckpt_iter}.pth"

                # Get iterator state for accurate resume
                iter_state = self.batch_iterator.state_dict()

                save_checkpoint(
                    str(save_path),
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler,
                    self.total_steps,
                    iter_state["epoch"],
                    self.cfg,
                    rank=self.rank,
                    batches_in_epoch=iter_state["batches_in_epoch"],
                )

        # Evaluate (skip if step-based eval is enabled)
        should_eval_epoch = (epoch + 1) % cfg.checkpoint.get('evaluate_every_n_epoch', 1) == 0
        if cfg.checkpoint.get('evaluate_every_n_steps') is None and should_eval_epoch:
            self._evaluate(f"step_{self.total_steps:06d}")

    def _finish_training(self):
        """Final save and evaluation."""
        logging.info("FINISHED TRAINING")

        # Full-state checkpoint (skip if periodic already saved at this step)
        ckpt_path = Path(self.cfg.checkpoint.save_dir) / f"step_{self.total_steps:06d}.pth"
        if not ckpt_path.exists():
            self._save_step_checkpoint()

        # Weights-only export for inference
        final_path = Path(self.cfg.checkpoint.save_dir) / "final_model.pth"
        save_final_model(str(final_path), self.model, rank=self.rank)

        # Final eval (load each dataset one at a time to save memory)
        dataset_root = self.cfg.evaluation.dataset_root
        for ds_name in self._final_dataset_names:
            logging.info(f"Loading final eval dataset: {ds_name}")
            dl = get_eval_dataloader(dataset_root, ds_name)
            self._evaluate(dataloaders=[(ds_name, dl)])
            del dl

        self.logger.finish()


class RuntimePseudoLabelTrainer(Trainer):
    """[DEPRECATED] Trainer that generates pseudo-labels at runtime via a teacher ensemble.

    Superseded by precomputed_pseudo_label mode, which loads pre-computed teacher
    predictions from disk (no teacher loaded at training time). See
    ``cotracker.datasets.pseudo_label_dataset`` and ``training/config/real_videos.yaml``.

    Kept for reference / fallback. Requires ``teacher`` config and ``training/teacher.py``.
    """

    def _build_teacher(self):
        """Build teacher ensemble for generating pseudo-labels."""
        from training.teacher import TeacherEnsemble

        logging.info("Building teacher ensemble for runtime pseudo-label training...")
        self.teacher = TeacherEnsemble(self.cfg.teacher, self.device)

    def forward_batch(self, batch, model, cfg):
        from training.forward_pseudo_label import forward_batch_pseudo_label

        return forward_batch_pseudo_label(batch, model, cfg, self.teacher)


class DenseTrackerTrainer(Trainer):
    """Trainer for dense tracker (CowTracker) mode.

    Disables no_sync because the dense tracker path may skip arbitrary
    micro-steps, risking missed DDP allreduce on the last valid step.
    """

    supports_no_sync = False

    def forward_batch(self, batch, model, cfg):
        from training.forward_dense_tracker import forward_batch_dense_tracker
        return forward_batch_dense_tracker(batch, model, cfg)


def create_trainer(cfg: DictConfig, **kwargs) -> Trainer:
    """Factory: pick the right Trainer subclass based on cfg.training.mode."""
    mode = cfg.training.get("mode", "supervised")
    if mode == "pseudo_label":
        logging.warning(
            "training.mode='pseudo_label' uses the deprecated RuntimePseudoLabelTrainer "
            "(teacher loaded at runtime). Consider 'precomputed_pseudo_label' instead."
        )
        return RuntimePseudoLabelTrainer(cfg, **kwargs)
    elif mode == "precomputed_pseudo_label":
        return Trainer(cfg, **kwargs)  # No teacher needed
    elif mode == "dense_tracker":
        return DenseTrackerTrainer(cfg, **kwargs)
    else:
        return Trainer(cfg, **kwargs)
