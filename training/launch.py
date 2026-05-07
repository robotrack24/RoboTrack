#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Hydra entry point for CoTracker training.

Usage:
    # Single GPU
    python training/launch.py exp_name=my_exp

    # Multi-GPU (single node)
    torchrun --nproc_per_node=8 training/launch.py exp_name=my_exp

    # Multi-node
    torchrun --nnodes=2 --nproc_per_node=8 training/launch.py exp_name=my_exp

    # Override model
    torchrun --nproc_per_node=8 training/launch.py model=cotracker3_online

    # Override dataset root
    torchrun --nproc_per_node=8 training/launch.py evaluation.dataset_root=/path/to/data
"""

import os
import sys
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf

from training.trainer import create_trainer
from training.logging_utils import setup_logging

@hydra.main(config_path="config", config_name="kubric", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    # Configure logging
    setup_logging()

    # Print config
    if int(os.environ.get("RANK", 0)) == 0:
        logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Create and run trainer
    trainer = create_trainer(cfg)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
