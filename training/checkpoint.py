# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpoint utilities with backward compatibility for legacy CoTracker checkpoints.

Supports loading:
1. Plain state_dict (old format)
2. {'model': ..., 'optimizer': ...} dict format
3. Keys with 'module.' prefix (DDP wrapped)
4. Keys with '_orig_mod.' prefix (torch.compile wrapper before DDP unwrap)
5. Filtering out 'time_emb', 'pos_emb' keys (matching original behavior)
"""

import os
import re
import random
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, FrozenSet

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def _normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip DDP and torch.compile prefixes so weights load into a plain nn.Module.

    Training saves after DDP; unwrapping can leave torch.compile's OptimizedModule,
    whose state_dict uses '_orig_mod.' prefixes. Trainer loads checkpoints before
    compile, so those prefixes must be removed here.
    """
    if not state_dict:
        return state_dict
    normalized: Dict[str, Any] = {}
    stripped_compile = False
    for k, v in state_dict.items():
        nk = k
        while nk.startswith("module."):
            nk = nk[len("module.") :]
        while nk.startswith("_orig_mod."):
            stripped_compile = True
            nk = nk[len("_orig_mod.") :]
        normalized[nk] = v
    if stripped_compile:
        logging.info("Stripped '_orig_mod.' prefix from checkpoint keys (torch.compile)")
    return normalized


def _unwrap_for_state_dict(model: nn.Module) -> nn.Module:
    """DDP -> inner module; torch.compile -> original module (clean checkpoint keys)."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = False,
    filter_keys: Optional[List[str]] = None,
) -> Tuple[int, int, int, Optional[Dict]]:
    """
    Load checkpoint with legacy format support.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into (should NOT be DDP wrapped yet)
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        scaler: Optional GradScaler to restore state
        strict: Whether to strictly enforce state_dict key matching
        filter_keys: List of substrings to filter out from state_dict keys

    Returns:
        Tuple of (total_steps, epoch, batches_in_epoch, rng_states):
        - total_steps: Training step count (0 if not found in checkpoint)
        - epoch: Epoch number (0 if not found in checkpoint)
        - batches_in_epoch: Batches consumed in current epoch (for mid-epoch resume)
        - rng_states: Dict of RNG states or None if not found
    """
    if filter_keys is None:
        filter_keys = ["time_emb", "pos_emb"]

    logging.info(f"Loading checkpoint from {path}")
    state_dict = torch.load(path, weights_only=False, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in state_dict:
        model_state = state_dict["model"]
    else:
        # Plain state_dict (old format)
        model_state = state_dict

    model_state = _normalize_state_dict_keys(model_state)

    # Filter out specified keys (e.g., time_emb, pos_emb)
    if filter_keys:
        original_len = len(model_state)
        model_state = {
            k: v for k, v in model_state.items()
            if not any(fk in k for fk in filter_keys)
        }
        if len(model_state) < original_len:
            logging.info(f"Filtered out {original_len - len(model_state)} keys containing {filter_keys}")

    # Load model state
    missing, unexpected = model.load_state_dict(model_state, strict=strict)
    if missing:
        logging.warning(f"Missing keys: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys: {unexpected}")

    # Load optimizer state if available and requested
    total_steps = 0
    if optimizer is not None and "optimizer" in state_dict:
        logging.info("Loading optimizer state")
        try:
            optimizer.load_state_dict(state_dict["optimizer"])
        except Exception as e:
            logging.warning(f"Failed to load optimizer state: {e}")

    if scheduler is not None and "scheduler" in state_dict:
        logging.info("Loading scheduler state")
        try:
            scheduler.load_state_dict(state_dict["scheduler"])
        except Exception as e:
            logging.warning(f"Failed to load scheduler state: {e}")

    if scaler is not None and "scaler" in state_dict:
        logging.info("Loading scaler state")
        try:
            scaler.load_state_dict(state_dict["scaler"])
        except Exception as e:
            logging.warning(f"Failed to load scaler state: {e}")

    if "total_steps" in state_dict:
        total_steps = state_dict["total_steps"]
        logging.info(f"Resuming from step {total_steps}")

    epoch = state_dict.get("epoch", 0)
    batches_in_epoch = state_dict.get("batches_in_epoch", 0)
    rng_states = state_dict.get("rng_states", None)

    logging.info("Checkpoint loaded successfully")
    return total_steps, epoch, batches_in_epoch, rng_states


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    total_steps: int,
    epoch: int,
    config: Optional[DictConfig] = None,
    rank: int = 0,
    batches_in_epoch: int = 0,
):
    """
    Save checkpoint (only on rank 0).

    Args:
        path: Path to save checkpoint
        model: Model (can be DDP wrapped)
        optimizer: Optimizer
        scheduler: Scheduler
        scaler: GradScaler
        total_steps: Current training step
        epoch: Current epoch
        config: Optional config to save
        rank: Current rank (only saves on rank 0)
        batches_in_epoch: Batches consumed in current epoch (for mid-epoch resume)
    """
    if rank != 0:
        return

    model_to_save = _unwrap_for_state_dict(model)

    state = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "total_steps": total_steps,
        "epoch": epoch,
        "batches_in_epoch": batches_in_epoch,
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    if config is not None:
        state["config"] = OmegaConf.to_container(config, resolve=True)

    # Create directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Save with backup
    tmp_path = str(path) + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

    logging.info(f"Saved checkpoint to {path}")


def _extract_step(filename: str) -> int:
    """Extract step number from checkpoint filename."""
    # Try step_XXXXXX.pth pattern first
    match = re.search(r'step_(\d+)\.pth', filename)
    if match:
        return int(match.group(1))

    # Try model_*_XXXXXX.pth pattern (epoch-based)
    match = re.search(r'_(\d{6})\.pth$', filename)
    if match:
        return int(match.group(1))

    # Fallback: try to find any number sequence
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])

    return 0


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Handles both epoch-based (model_*.pth) and step-based (step_*.pth) checkpoints.
    Returns the checkpoint with the highest step number.

    Args:
        ckpt_dir: Directory to search

    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(ckpt_dir):
        return None

    ckpt_files = [
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".pth") and "final" not in f and not os.path.isdir(os.path.join(ckpt_dir, f))
    ]

    if not ckpt_files:
        return None

    # Find checkpoint with highest step number
    latest = max(ckpt_files, key=_extract_step)
    latest_step = _extract_step(latest)
    logging.info(f"Found latest checkpoint: {latest} (step {latest_step})")

    return os.path.join(ckpt_dir, latest)


PINNED_STEPS: FrozenSet[int] = frozenset({15000})


def cleanup_old_checkpoints(ckpt_dir: str, num_to_keep: Optional[int], rank: int = 0):
    """
    Delete old checkpoints, keeping only the N most recent.

    Always preserves the most recent checkpoint and any pinned steps.

    Args:
        ckpt_dir: Directory containing checkpoints
        num_to_keep: Number of checkpoints to keep (None = keep all)
        rank: Current rank (only rank 0 deletes)
    """
    if rank != 0 or num_to_keep is None:
        return

    if not os.path.exists(ckpt_dir):
        return

    # Find all checkpoints with their step numbers (exclude final)
    checkpoints = []
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pth") and "final" not in f:
            step = _extract_step(f)
            checkpoints.append((step, os.path.join(ckpt_dir, f)))

    # Sort by step (newest first) and delete old ones
    checkpoints.sort(reverse=True)
    # Always keep at least the most recent (index 0)
    num_to_keep = max(num_to_keep, 1)
    for step, path in checkpoints[num_to_keep:]:
        if step in PINNED_STEPS:
            continue
        logging.info(f"Deleting old checkpoint: {path}")
        os.remove(path)


def cleanup_checkpoints_by_metric(
    ckpt_dir: str,
    eval_dir: str,
    num_to_keep: Optional[int],
    metric_key: str,
    metric_mode: str = "max",
    rank: int = 0,
):
    """
    Delete checkpoints, keeping only the N with the best eval metric.

    Unevaluated checkpoints (no eval_metrics.json yet) are always kept so
    they survive until the next eval pass can score them.

    Args:
        ckpt_dir: Directory containing checkpoints (step_XXXXXX.pth)
        eval_dir: Directory containing eval results (eval/step_XXXXXX/eval_metrics.json)
        num_to_keep: Number of best checkpoints to keep (None = keep all)
        metric_key: Metric key in eval_metrics.json to rank by
                    (e.g. "eval/tapvid_robotap_avg_delta")
        metric_mode: "max" (higher is better) or "min" (lower is better)
        rank: Current rank (only rank 0 deletes)
    """
    if rank != 0 or num_to_keep is None:
        return

    if not os.path.exists(ckpt_dir):
        return

    import json

    evaluated = []    # (metric_value, step, path)
    unevaluated = []  # (step, path)

    for f in os.listdir(ckpt_dir):
        if not f.endswith(".pth") or "final" in f:
            continue
        step = _extract_step(f)
        ckpt_path = os.path.join(ckpt_dir, f)
        metrics_path = os.path.join(eval_dir, f"step_{step:06d}", "eval_metrics.json")

        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as mf:
                    metrics = json.load(mf)
                if metric_key in metrics:
                    evaluated.append((metrics[metric_key], step, ckpt_path))
                else:
                    logging.warning(
                        f"Metric '{metric_key}' not found in {metrics_path} "
                        f"(available: {list(metrics.keys())}). Keeping checkpoint."
                    )
                    unevaluated.append((step, ckpt_path))
            except (json.JSONDecodeError, OSError) as e:
                logging.warning(f"Failed to read {metrics_path}: {e}. Keeping checkpoint.")
                unevaluated.append((step, ckpt_path))
        else:
            unevaluated.append((step, ckpt_path))

    reverse = metric_mode == "max"
    evaluated.sort(key=lambda x: x[0], reverse=reverse)

    keep = {path for _, _, path in evaluated[:num_to_keep]}
    keep.update(path for _, path in unevaluated)

    # Always keep the most recent checkpoint so training can resume
    all_checkpoints = [(s, p) for _, s, p in evaluated] + list(unevaluated)
    if all_checkpoints:
        latest_path = max(all_checkpoints, key=lambda x: x[0])[1]
        keep.add(latest_path)

    for metric_val, step, path in evaluated:
        if path not in keep and step not in PINNED_STEPS:
            logging.info(
                f"Deleting checkpoint {os.path.basename(path)} "
                f"({metric_key}={metric_val:.4f})"
            )
            os.remove(path)

    if evaluated:
        best_val, best_step, _ = evaluated[0]
        logging.info(
            f"Best checkpoint: step {best_step} ({metric_key}={best_val:.4f}), "
            f"keeping {len(keep)} of {len(evaluated) + len(unevaluated)} checkpoints"
        )


def save_final_model(
    path: str,
    model: nn.Module,
    rank: int = 0,
):
    """
    Save final model weights only (no optimizer/scheduler state).

    Args:
        path: Path to save model
        model: Model (can be DDP wrapped)
        rank: Current rank (only saves on rank 0)
    """
    if rank != 0:
        return

    model_to_save = _unwrap_for_state_dict(model)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_to_save.state_dict(), path)

    logging.info(f"Saved final model to {path}")
