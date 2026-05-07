# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed training utilities for torch.distributed.
Handles DDP initialization, rank management, and synchronization.
"""

import os
import random
import logging
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist


def get_machine_local_and_dist_rank() -> Tuple[int, int]:
    """
    Get local rank and distributed rank from environment variables.
    Supports both torchrun and SLURM environments.

    Returns:
        Tuple of (local_rank, distributed_rank)
    """
    # torchrun sets these
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist_rank = int(os.environ.get("RANK", 0))
    # SLURM sets these
    elif "SLURM_LOCALID" in os.environ:
        local_rank = int(os.environ["SLURM_LOCALID"])
        dist_rank = int(os.environ.get("SLURM_PROCID", 0))
    else:
        local_rank = 0
        dist_rank = 0

    return local_rank, dist_rank


def get_world_size() -> int:
    """Get world size from environment or return 1 for single GPU."""
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    elif "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    return 1


def init_distributed(backend: str = "nccl", timeout_mins: int = 30) -> Tuple[int, int, int, torch.device]:
    """
    Initialize distributed training from environment variables.

    Args:
        backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
        timeout_mins: Timeout for distributed operations in minutes

    Returns:
        Tuple of (rank, world_size, local_rank, device)
    """
    from datetime import timedelta

    local_rank, rank = get_machine_local_and_dist_rank()
    world_size = get_world_size()

    if world_size > 1:
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=timeout_mins),
        )
        logging.info(f"Initialized distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        logging.info("Running in single GPU mode (no distributed)")

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device


def cleanup():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current rank, or 0 if not distributed."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def barrier():
    """Synchronization barrier across all processes."""
    if dist.is_initialized():
        dist.barrier()


def set_seeds(seed: int, max_epochs: int = 100, rank: int = 0):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Base seed value
        max_epochs: Used to offset seed by rank
        rank: Distributed rank for seed offset
    """
    seed = seed + rank
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    """Seed function for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)
