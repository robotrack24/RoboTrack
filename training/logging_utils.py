# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Wandb-based logger for CoTracker training.
Replaces TensorBoard Logger from original implementation.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Install with: pip install wandb")

def setup_logging():
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    formatter = logging.Formatter(
        f'[Rank {rank}] [%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class WandbLogger:
    """
    Wandb-based logger with interface compatible with original CoTracker Logger.

    Supports:
    - Scalar logging
    - Image/video logging
    - Running metric aggregation
    - Model watching for gradient logging
    """

    SUM_FREQ = 100  # Aggregate metrics every N steps

    def __init__(
        self,
        project: str,
        name: str,
        config: Optional[Dict] = None,
        entity: Optional[str] = None,
        dir: Optional[str] = None,
        model: Optional[nn.Module] = None,
        rank: int = 0,
        enabled: bool = True,
        tags: Optional[list] = None,
        resume: Optional[str] = None,
    ):
        """
        Initialize wandb logger.

        Args:
            project: Wandb project name
            name: Run name
            config: Config dict to log
            entity: Wandb entity (team/user)
            dir: Directory for wandb files
            model: Optional model for gradient watching
            rank: Distributed rank (only rank 0 logs)
            enabled: Whether logging is enabled
            tags: Optional tags for the run
            resume: Resume mode ('allow', 'must', 'never', or run_id)
        """
        self.rank = rank
        self.enabled = enabled and WANDB_AVAILABLE and rank == 0
        self.running_metrics = {}
        self.step_count = 0

        if self.enabled:
            if dir is not None:
                Path(dir).mkdir(parents=True, exist_ok=True)

            # Use name as run ID for deterministic resume
            run_id = name.replace("/", "_").replace(" ", "_")[:64] if resume else None

            wandb.init(
                project=project,
                name=name,
                id=run_id,
                config=config,
                entity=entity,
                dir=dir,
                tags=tags,
                resume=resume,
                reinit=True,
            )

            # Note: wandb.watch() disabled - it uses internal step counter
            # that breaks on resume (logs to step 0,1,2... when training is at 500+)
            # if model is not None:
            #     wandb.watch(model, log="gradients", log_freq=1000)

            logging.info(f"Wandb initialized: {wandb.run.url}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to wandb.

        Args:
            metrics: Dict of metric name -> value
            step: Optional step number (uses internal counter if not provided)
        """
        if not self.enabled:
            return

        # Flatten nested metrics
        flat_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat_metrics[f"{k}/{k2}"] = self._to_scalar(v2)
            else:
                flat_metrics[k] = self._to_scalar(v)

        wandb.log(flat_metrics, step=step)

    def log_scalar(self, name: str, value: Union[float, torch.Tensor], step: int):
        """Log a single scalar value."""
        if not self.enabled:
            return
        wandb.log({name: self._to_scalar(value)}, step=step)

    def log_image(self, name: str, image: Union[torch.Tensor, np.ndarray], step: int, caption: Optional[str] = None):
        """
        Log an image to wandb.

        Args:
            name: Image name/tag
            image: Image tensor (C, H, W) or numpy array (H, W, C)
            step: Step number
            caption: Optional caption
        """
        if not self.enabled:
            return

        if isinstance(image, torch.Tensor):
            image = image.cpu()
            if image.dim() == 3 and image.shape[0] in [1, 3, 4]:
                # C, H, W -> H, W, C
                image = image.permute(1, 2, 0)
            image = image.numpy()

        # Normalize to 0-255 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        wandb.log({name: wandb.Image(image, caption=caption)}, step=step)

    def log_video(
        self,
        name: str,
        video: Union[torch.Tensor, np.ndarray],
        step: int,
        fps: int = 10,
        caption: Optional[str] = None,
    ):
        """
        Log a video to wandb.

        Args:
            name: Video name/tag
            video: Video tensor (T, C, H, W) or numpy array (T, H, W, C)
            step: Step number
            fps: Frames per second
            caption: Optional caption
        """
        if not self.enabled:
            return

        if isinstance(video, torch.Tensor):
            video = video.cpu()
            if video.dim() == 4 and video.shape[1] in [1, 3, 4]:
                # T, C, H, W -> T, H, W, C
                video = video.permute(0, 2, 3, 1)
            video = video.numpy()

        # Normalize to 0-255 if needed
        if video.max() <= 1.0:
            video = (video * 255).astype(np.uint8)

        wandb.log({name: wandb.Video(video, fps=fps, caption=caption)}, step=step)

    def log_video_file(self, name: str, path: str, step: int, fps: int = 7, caption: Optional[str] = None):
        """Log a video file from disk to wandb.

        Args:
            name: Metric key (e.g. "eval/tapvid_davis_0")
            path: Path to .mp4 file on disk
            step: Training step
            fps: Frames per second for playback
            caption: Optional caption
        """
        if not self.enabled:
            return
        wandb.log({name: wandb.Video(path, fps=fps, caption=caption)}, step=step)

    def log_video_table(self, name: str, video_paths: list, step: int, fps: int = 7):
        """Log multiple videos as a wandb Table (one row per video).

        Args:
            name: Table key (e.g. "eval_viz/tapvid_davis_first")
            video_paths: List of .mp4 file paths
            step: Training step
            fps: Playback FPS
        """
        if not self.enabled:
            return
        table = wandb.Table(columns=["video", "sequence"])
        for path in video_paths:
            seq_name = os.path.splitext(os.path.basename(path))[0]
            table.add_data(wandb.Video(path, fps=fps), seq_name)
        wandb.log({name: table}, step=step)

    def push(self, metrics: Dict[str, float], task: str):
        """
        Accumulate metrics for aggregation (compatibility with original Logger).

        Args:
            metrics: Dict of metric name -> value
            task: Task name prefix
        """
        self.step_count += 1

        for key, value in metrics.items():
            task_key = f"{task}/{key}"
            if task_key not in self.running_metrics:
                self.running_metrics[task_key] = 0.0
            self.running_metrics[task_key] += value

        # Log aggregated metrics periodically
        if self.step_count % self.SUM_FREQ == self.SUM_FREQ - 1:
            avg_metrics = {
                k: v / self.SUM_FREQ
                for k, v in self.running_metrics.items()
            }
            self.log(avg_metrics, step=self.step_count)
            self.running_metrics = {}

    def log_config(self, config: Dict):
        """Update wandb config."""
        if not self.enabled:
            return
        wandb.config.update(config, allow_val_change=True)

    def log_artifact(self, path: str, name: str, type: str = "model"):
        """Log a file as a wandb artifact."""
        if not self.enabled:
            return
        artifact = wandb.Artifact(name, type=type)
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def finish(self):
        """Finish wandb run."""
        if self.enabled:
            wandb.finish()
            logging.info("Wandb run finished")

    def _to_scalar(self, value: Any) -> float:
        """Convert value to scalar."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        return float(value)

    @property
    def run_url(self) -> Optional[str]:
        """Get wandb run URL."""
        if self.enabled and wandb.run is not None:
            return wandb.run.url
        return None


class DummyLogger:
    """Dummy logger that does nothing (for non-rank-0 processes)."""

    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def log_scalar(self, *args, **kwargs):
        pass

    def log_image(self, *args, **kwargs):
        pass

    def log_video(self, *args, **kwargs):
        pass

    def log_video_file(self, *args, **kwargs):
        pass

    def log_video_table(self, *args, **kwargs):
        pass

    def push(self, *args, **kwargs):
        pass

    def log_config(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def finish(self):
        pass


class LocalLogger:
    """File-based metric logger that writes JSONL to disk.

    Each session (fresh or resumed) creates a new timestamped file.
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = self.log_dir / f"metrics_{timestamp}.jsonl"
        self._file = open(self.log_path, "w")

        logging.info(f"LocalLogger writing to: {self.log_path}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        flat = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}/{k2}"] = self._to_scalar(v2)
            else:
                flat[k] = self._to_scalar(v)

        entry = {"step": step, "timestamp": datetime.now().isoformat()}
        entry.update(flat)
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def log_scalar(self, name: str, value: Any, step: int):
        self.log({name: value}, step=step)

    def log_config(self, config: Dict):
        config_path = self.log_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def log_image(self, *args, **kwargs):
        pass

    def log_video(self, *args, **kwargs):
        pass

    def log_video_file(self, *args, **kwargs):
        pass

    def log_video_table(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def push(self, *args, **kwargs):
        pass

    def finish(self):
        if self._file and not self._file.closed:
            self._file.close()

    @property
    def run_url(self) -> Optional[str]:
        return None

    def _to_scalar(self, value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        return float(value)


class CompositeLogger:
    """Fan-out logger that delegates to multiple loggers."""

    def __init__(self, loggers: List):
        self.loggers = loggers

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        for logger in self.loggers:
            logger.log(metrics, step=step)

    def log_scalar(self, name: str, value: Any, step: int):
        for logger in self.loggers:
            logger.log_scalar(name, value, step)

    def log_image(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log_image(*args, **kwargs)

    def log_video(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log_video(*args, **kwargs)

    def log_video_file(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log_video_file(*args, **kwargs)

    def log_video_table(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log_video_table(*args, **kwargs)

    def push(self, *args, **kwargs):
        for logger in self.loggers:
            logger.push(*args, **kwargs)

    def log_config(self, config: Dict):
        for logger in self.loggers:
            logger.log_config(config)

    def log_artifact(self, *args, **kwargs):
        for logger in self.loggers:
            logger.log_artifact(*args, **kwargs)

    def finish(self):
        for logger in self.loggers:
            logger.finish()

    @property
    def run_url(self) -> Optional[str]:
        for logger in self.loggers:
            url = getattr(logger, "run_url", None)
            if url is not None:
                return url
        return None


def create_logger(
    project: str,
    name: str,
    config: Optional[Dict] = None,
    rank: int = 0,
    use_wandb: bool = True,
    resume: Optional[str] = "allow",
    saved_step: int = 0,
    ckpt_path: Optional[str] = None,
    log_dir: Optional[str] = None,
    local_logging: bool = True,
    **kwargs,
):
    """
    Create appropriate logger based on rank and settings.

    Args:
        project: Wandb project name
        name: Run name
        config: Config dict
        rank: Distributed rank
        enabled: Whether wandb logging is enabled
        resume: Wandb resume mode ('allow', 'must', 'never', None)
                'allow' = resume if run exists, else create new
        saved_step: Step number from checkpoint (0 if not resuming)
        ckpt_path: Path to checkpoint (for logging metadata)
        log_dir: Directory for local JSONL logs (required if local_logging=True)
        local_logging: Whether to enable local file logging (default True)

    Returns:
        Appropriate logger: WandbLogger, LocalLogger, CompositeLogger, or DummyLogger
    """
    if rank != 0:
        return DummyLogger()

    loggers = []

    # Wandb logger
    if use_wandb:
        # Add resume info to tags and config
        tags = kwargs.pop("tags", None) or []
        if saved_step > 0:
            tags = list(tags) + ["resumed", name]

        # Update config with resume info
        if config and saved_step > 0:
            config = dict(config)
            config["resumed_from_step"] = saved_step
            if ckpt_path:
                config["resumed_from_checkpoint"] = str(ckpt_path)

        loggers.append(WandbLogger(
            project=project,
            name=name,
            config=config,
            rank=rank,
            enabled=use_wandb,
            resume=resume,
            tags=tags if tags else None,
            **kwargs,
        ))

    # Local logger
    if local_logging and log_dir is not None:
        loggers.append(LocalLogger(log_dir=log_dir))

    if len(loggers) == 0:
        return DummyLogger()
    elif len(loggers) == 1:
        return loggers[0]
    else:
        return CompositeLogger(loggers)
