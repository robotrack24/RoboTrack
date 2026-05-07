# =============================================================================
# Standalone evaluation entry point
# =============================================================================

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import logging

from training.logging_utils import setup_logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig):
    """
    Standalone evaluation entry point.
    
    Reuses the Trainer class in eval-only mode for consistency.
    
    Usage:
        python training/eval.py checkpoint.restore_ckpt=/path/to/model.pth
        python training/eval.py checkpoint.restore_ckpt=/path/to/model.pth evaluation.datasets=[tapvid_davis_first,tapvid_robotap]
    """
    from training.trainer import Trainer
    
    setup_logging()
    
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create trainer in eval-only mode (skips distributed setup)
    trainer = Trainer(cfg, eval_only=True)
    trainer.setup_eval_only()
    trainer._evaluate()


if __name__ == "__main__":
    main()