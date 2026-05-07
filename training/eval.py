# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation utilities for CoTracker training.
Uses WandbLogger directly instead of TensorBoard.
"""

import os
import glob
import logging
from typing import List, Tuple, Optional
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader

from cotracker.datasets.utils import collate_fn, collate_fn_train
from cotracker.models.evaluation_predictor import EvaluationPredictor
from cotracker.evaluation.core.evaluator import Evaluator

_HF_REPO = "RoboTrack24/RoboTrack"


def _download_hf_subset(subset: str) -> str:
    """Download a subset of the RoboTrack HF dataset and return the local path."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        _HF_REPO,
        repo_type="dataset",
        allow_patterns=f"{subset}/**",
    )
    return os.path.join(root, subset)


def get_eval_dataloader(dataset_root: str, ds_name: str) -> DataLoader:
    """
    Create evaluation dataloader for a specific dataset.

    Args:
        dataset_root: Root directory containing evaluation datasets
        ds_name: Dataset name (e.g., 'tapvid_davis_first', 'tapvid_robotap')

    Returns:
        DataLoader for the evaluation dataset
    """
    from cotracker.datasets.tap_vid_datasets import TapVidDataset

    collate_fn_local = collate_fn

    if ds_name == "tapvid_davis_first":
        data_root = os.path.join(dataset_root, "tapvid/tapvid_davis/tapvid_davis.pkl")
        eval_dataset = TapVidDataset(
            dataset_type="davis", data_root=data_root, queried_first=True
        )
    elif ds_name == "tapvid_kinetics_first":
        eval_dataset = TapVidDataset(
            dataset_type="kinetics",
            data_root=os.path.join(dataset_root, "tapvid", "tapvid_kinetics"),
        )
    elif ds_name == "tapvid_stacking":
        eval_dataset = TapVidDataset(
            dataset_type="stacking",
            data_root=os.path.join(
                dataset_root, "tapvid", "tapvid_rgb_stacking", "tapvid_rgb_stacking.pkl"
            ),
        )
    elif ds_name == "tapvid_robotap":
        eval_dataset = TapVidDataset(
            dataset_type="robotap",
            data_root=os.path.join(dataset_root, "tapvid", "tapvid_robotap"),
        )
    elif ds_name == "robotrack-sim":
        from cotracker.datasets.molmospaces_dataset import MolmoSpacesDataset

        molmo_root = _download_hf_subset("RoboTrack-Sim")
        eval_dataset = MolmoSpacesDataset(
            data_root=molmo_root,
            crop_size=(384, 512),
            seq_len=200,
            traj_per_sample=30,
            cameras=None,
            configs=None,
            max_samples=None,
        )
        collate_fn_local = collate_fn_train
    elif ds_name == "robotrack-real":
        from cotracker.datasets.droid_dataset import DroidDataset

        droid_root = _download_hf_subset("RoboTrack-Real")
        eval_dataset = DroidDataset(
            data_root=droid_root,
            resize_to=(256, 256),
        )
    else:
        raise ValueError(f"Unknown eval dataset: {ds_name}")

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_local,
    )
    return eval_dataloader

def _get_eval_support_settings(eval_protocol: str, offline_model: bool) -> dict:
    """Return EvaluationPredictor kwargs for CoTracker3 evaluation.

    Applied to all evaluation datasets so CoTracker gets the support-point
    context it was designed for.

    Uses batch mode (single_point=False) with a global 5×5 grid for
    scaled / online models, and 1000 uniformly sampled points for
    baseline offline models.
    """
    if not offline_model or eval_protocol == "scaled":
        return dict(
            single_point=False,
            grid_size=5,
            local_grid_size=0,
            num_uniformly_sampled_pts=0,
        )
    else:
        return dict(
            single_point=False,
            grid_size=0,
            local_grid_size=0,
            num_uniformly_sampled_pts=1000,
        )


def run_eval(
    evaluator: Evaluator,
    model: torch.nn.Module,
    dataloaders: List[Tuple[str, DataLoader]],
    logger,
    step: int,
    query_random: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
    viz_dir: Optional[str] = None,
    memory_profile_dir: Optional[str] = None,
    eval_protocol: str = "baseline",
    offline_model: bool = True,
) -> dict:
    """
    Run evaluation on multiple datasets and log to wandb.

    Args:
        evaluator: Evaluator instance (used as template; overridden by viz_dir)
        model: Model to evaluate (can be DDP wrapped)
        dataloaders: List of (dataset_name, dataloader) tuples
        logger: Logger(s)
        step: Current training step
        query_random: Whether to use random query sampling
        viz_dir: Directory for eval visualizations. If set, creates a fresh
                 Evaluator pointing at this directory so videos are organized
                 per-step.
        eval_protocol: "scaled" or "baseline" — selects the TAP-Vid support
                       point protocol from the CoTracker3 paper.
        offline_model: Whether the model is an offline variant.

    Returns:
        Dictionary of all metrics
    """
    model.eval()

    # Create step-specific evaluator if viz_dir is provided
    if viz_dir is not None:
        evaluator = Evaluator(viz_dir)

    def unwrap_model(model):
        """Unwrap model from DDP/Lightning wrappers."""
        while hasattr(model, "module"):
            model = model.module
        return model

    # Belt-and-suspenders: ensure the inner module is in eval mode too.
    # DDP's train()/eval() should propagate, but for custom CUDA modules
    # (like Mamba) we want to be certain.
    unwrap_model(model).eval()

    all_metrics = {}

    eval_settings = _get_eval_support_settings(eval_protocol, offline_model)

    for ds_name, dataloader in dataloaders:
        # Dataset-specific settings
        visualize_every = 1
        n_iters = 6

        # Apply paper-matching support points to all datasets so CoTracker
        # gets the context it was designed for (matching standalone evaluate.py
        # which uses the same settings globally).
        grid_size = eval_settings["grid_size"]
        local_grid_size = eval_settings["local_grid_size"]
        single_point = eval_settings["single_point"]
        num_uniformly_sampled_pts = eval_settings["num_uniformly_sampled_pts"]

        if ds_name == "robotrack-sim":
            visualize_every = 10
        elif ds_name == "robotrack-real":
            visualize_every = 5
        elif "davis" in ds_name:
            visualize_every = 1
        elif "tapvid_stacking" in ds_name:
            visualize_every = 2
        elif "robotap" in ds_name:
            visualize_every = 10
        elif "kinetics" in ds_name:
            visualize_every = 50

        # Create predictor with unwrapped model.
        # For models that expose model_resolution (e.g. CowTracker wrapper),
        # use it so eval interpolation matches model input constraints.
        unwrapped = unwrap_model(model)
        interp_shape = getattr(unwrapped, "model_resolution", (384, 512))
        predictor = EvaluationPredictor(
            unwrapped,
            interp_shape=tuple(interp_shape),
            grid_size=grid_size,
            local_grid_size=local_grid_size,
            single_point=single_point,
            num_uniformly_sampled_pts=num_uniformly_sampled_pts,
            n_iters=n_iters,
        )

        # Run evaluation. We wrap in torch.no_grad() and synchronize/clear
        # the CUDA cache first because some custom CUDA kernels (notably
        # Mamba's selective_scan) misbehave when called right after a
        # multi-GPU training step with pending CUDA work or a stale
        # autograd graph -- which silently produces garbage predictions
        # while the standalone-loaded checkpoint is fine.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        with torch.no_grad(), torch.amp.autocast(
            'cuda', enabled=amp_dtype is not None, dtype=amp_dtype or torch.float32
        ):
            metrics = evaluator.evaluate_sequence(
                model=predictor,
                test_dataloader=dataloader,
                dataset_name=ds_name,
                train_mode=True,
                writer=None,  # We handle logging ourselves
                step=step,
                visualize_every=visualize_every,
                memory_profile_dir=memory_profile_dir,
            )

        # Process metrics based on dataset type
        if "tapvid" in ds_name or ds_name in ("robotrack-sim", "robotrack-real"):
            processed_metrics = {
                f"eval/{ds_name}_avg_OA": metrics["avg"]["occlusion_accuracy"],
                f"eval/{ds_name}_avg_delta": metrics["avg"]["average_pts_within_thresh"],
                f"eval/{ds_name}_avg_Jaccard": metrics["avg"]["average_jaccard"],
            }
        else:
            processed_metrics = {
                f"eval/{ds_name}_{k}": v for k, v in metrics.get("avg", metrics).items()
            }

        # Log metrics
        logger.log(processed_metrics, step=step)

        # Log eval visualization videos to wandb (grouped per dataset)
        if viz_dir is not None:
            for video_path in sorted(glob.glob(os.path.join(viz_dir, f"{ds_name}_*.mp4"))):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                logger.log_video_file(
                    f"eval_viz_{ds_name}/{video_name}", video_path, step=step, fps=7,
                )

        # Store for return
        all_metrics.update(processed_metrics)

        logging.info(f"Eval {ds_name} @ step {step}: {processed_metrics}")

    # Write per-step eval metrics JSON alongside visualizations
    if viz_dir is not None and all_metrics:
        import json
        metrics_path = os.path.join(viz_dir, "eval_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"step": step, **all_metrics}, f, indent=2)
        logging.info(f"Eval metrics saved to {metrics_path}")

    return all_metrics


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    """
    Standalone evaluation entry point.

    Reuses Trainer in eval-only mode for consistency with training path.
    """
    from training.trainer import Trainer

    # Hydra changes cwd; resolve key output dirs against original cwd.
    original_cwd = hydra.utils.get_original_cwd()
    if not os.path.isabs(cfg.exp_dir):
        cfg.exp_dir = os.path.join(original_cwd, cfg.exp_dir)
    if not os.path.isabs(cfg.checkpoint.save_dir):
        cfg.checkpoint.save_dir = os.path.join(original_cwd, cfg.checkpoint.save_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    trainer = Trainer(cfg, eval_only=True)
    trainer.setup_eval_only()
    trainer._evaluate()


if __name__ == "__main__":
    main()
