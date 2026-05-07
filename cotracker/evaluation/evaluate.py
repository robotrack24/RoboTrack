# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os
import hydra
import numpy as np
import torch

from dataclasses import dataclass

from omegaconf import OmegaConf

from cotracker.datasets.utils import collate_fn, collate_fn_train
from cotracker.models.evaluation_predictor import EvaluationPredictor

from cotracker.evaluation.core.evaluator import Evaluator
from cotracker.models.build_cotracker import build_cotracker
from cotracker.models.tapnext_torch_predictor import TAPNextTorchPredictor
from cotracker.models.alltracker_predictor import build_alltracker, AllTrackerPredictor

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


@dataclass(eq=False)
class DefaultConfig:
    # Directory where all outputs of the experiment will be saved.
    exp_dir: str = "./outputs"

    # Name of the dataset to be used for the evaluation.
    dataset_name: str = "tapvid_davis_first"
    # The root directory of the dataset.
    dataset_root: str = "./"

    # Path to the pre-trained model checkpoint to be used for the evaluation.
    # The default value is the path to a specific CoTracker model checkpoint.
    checkpoint: str = "./checkpoints/scaled_online.pth"
    # EvaluationPredictor parameters
    # The size (N) of the support grid used in the predictor.
    # The total number of points is (N*N).
    grid_size: int = 5
    # The size (N) of the local support grid.
    local_grid_size: int = 8
    num_uniformly_sampled_pts: int = 0
    sift_size: int = 0
    # A flag indicating whether to evaluate one ground truth point at a time.
    single_point: bool = False
    offline_model: bool = False
    window_len: int = 16
    # The number of iterative updates for each sliding window.
    n_iters: int = 6

    seed: int = 0
    gpu_idx: int = 0
    local_extent: int = 50

    v2: bool = False

    # TAP-Vid evaluation protocol from the CoTracker3 paper.
    # One of: scaled_offline, scaled_online, baseline_offline, baseline_online.
    # Overrides grid_size/local_grid_size/single_point/num_uniformly_sampled_pts
    # to match the paper. Set to "auto" to use the manual values above.
    eval_protocol: str = "auto"

    # One of: "cotracker" (default), "tapnext", "alltracker".
    model_type: str = "cotracker"

    # AllTracker-specific (ignored unless model_type == "alltracker").
    alltracker_window_len: int = 16
    alltracker_inference_iters: int = 4

    # Do not add a top-level `hydra` key here: on Hydra 1.2+ it overwrites
    # Hydra's reserved cfg.hydra and causes `assert cfg.hydra.mode == RunMode.RUN`
    # to fail. For run dir / no .hydra subdir, pass e.g.:
    #   hydra.run.dir=. hydra.output_subdir=null


def run_eval(cfg: DefaultConfig):
    """
    The function evaluates CoTracker on a specified benchmark dataset based on a provided configuration.

    Args:
        cfg (DefaultConfig): An instance of DefaultConfig class which includes:
            - exp_dir (str): The directory path for the experiment.
            - dataset_name (str): The name of the dataset to be used.
            - dataset_root (str): The root directory of the dataset.
            - checkpoint (str): The path to the CoTracker model's checkpoint.
            - single_point (bool): A flag indicating whether to evaluate one ground truth point at a time.
            - n_iters (int): The number of iterative updates for each sliding window.
            - seed (int): The seed for setting the random state for reproducibility.
            - gpu_idx (int): The index of the GPU to be used.
    """
    # Creating the experiment directory if it doesn't exist
    os.makedirs(cfg.exp_dir, exist_ok=True)

    # Saving the experiment configuration to a .yaml file in the experiment directory
    cfg_file = os.path.join(cfg.exp_dir, "expconfig.yaml")
    with open(cfg_file, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    evaluator = Evaluator(cfg.exp_dir)

    if cfg.model_type == "tapnext":
        predictor = TAPNextTorchPredictor(ckpt_path=cfg.checkpoint)
        if torch.cuda.is_available():
            predictor.cuda()
    elif cfg.model_type == "alltracker":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        alltracker_model = build_alltracker(
            checkpoint=cfg.checkpoint,
            window_len=cfg.alltracker_window_len,
            device=device,
        )
        predictor = AllTrackerPredictor(
            alltracker_model,
            interp_shape=(384, 512),
            inference_iters=cfg.alltracker_inference_iters,
        )
    else:
        cotracker_model = build_cotracker(
            cfg.checkpoint, offline=cfg.offline_model, window_len=cfg.window_len, v2=cfg.v2
        )

        # Apply paper-specified TAP-Vid eval protocol if requested.
        grid_size = cfg.grid_size
        local_grid_size = cfg.local_grid_size
        single_point = cfg.single_point
        num_uniformly_sampled_pts = cfg.num_uniformly_sampled_pts

        if cfg.eval_protocol != "auto":
            uses_grid = cfg.eval_protocol in ("scaled_offline", "scaled_online", "baseline_online")
            if uses_grid:
                single_point = False
                grid_size = 5
                local_grid_size = 0
                num_uniformly_sampled_pts = 0
            else:  # baseline_offline
                single_point = False
                grid_size = 0
                local_grid_size = 0
                num_uniformly_sampled_pts = 1000

        predictor = EvaluationPredictor(
            cotracker_model,
            grid_size=grid_size,
            local_grid_size=local_grid_size,
            sift_size=cfg.sift_size,
            single_point=single_point,
            num_uniformly_sampled_pts=num_uniformly_sampled_pts,
            n_iters=cfg.n_iters,
            local_extent=cfg.local_extent,
            interp_shape=(384, 512),
        )

        if torch.cuda.is_available():
            predictor.model = predictor.model.cuda()

    # Setting the random seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Constructing the specified dataset
    curr_collate_fn = collate_fn
    if "tapvid" in cfg.dataset_name:
        from cotracker.datasets.tap_vid_datasets import TapVidDataset

        dataset_type = cfg.dataset_name.split("_")[1]
        if dataset_type == "davis":
            a = os.path.join(cfg.dataset_root, "tapvid_davis", "tapvid_davis.pkl")
            b = os.path.join(cfg.dataset_root, "tapvid", "tapvid_davis", "tapvid_davis.pkl")
            data_root = b if os.path.exists(b) and not os.path.exists(a) else a
        elif dataset_type == "kinetics":
            a = os.path.join(cfg.dataset_root, "tapvid_kinetics")
            b = os.path.join(cfg.dataset_root, "tapvid", "tapvid_kinetics")
            data_root = b if os.path.isdir(b) and not os.path.isdir(a) else a
        elif dataset_type == "robotap":
            a = os.path.join(cfg.dataset_root, "tapvid_robotap")
            b = os.path.join(cfg.dataset_root, "tapvid", "tapvid_robotap")
            if glob.glob(os.path.join(a, "robotap_split*.pkl")):
                data_root = a
            elif glob.glob(os.path.join(b, "robotap_split*.pkl")):
                data_root = b
            else:
                data_root = a
        elif dataset_type == "stacking":
            a = os.path.join(cfg.dataset_root, "tapvid_rgb_stacking", "tapvid_rgb_stacking.pkl")
            b = os.path.join(cfg.dataset_root, "tapvid", "tapvid_rgb_stacking", "tapvid_rgb_stacking.pkl")
            data_root = b if os.path.exists(b) and not os.path.exists(a) else a

        test_dataset = TapVidDataset(
            dataset_type=dataset_type,
            data_root=data_root,
            queried_first=not "strided" in cfg.dataset_name,
            # resize_to=None,
        )
    elif cfg.dataset_name == "robotrack-real":
        from cotracker.datasets.droid_dataset import DroidDataset

        droid_root = _download_hf_subset("RoboTrack-Real")
        test_dataset = DroidDataset(
            data_root=droid_root,
        )
    elif cfg.dataset_name == "robotrack-sim":
        from cotracker.datasets.molmospaces_dataset import MolmoSpacesDataset

        molmo_root = _download_hf_subset("RoboTrack-Sim")
        test_dataset = MolmoSpacesDataset(
            data_root=molmo_root,
            crop_size=(384, 512),
            cameras=None,
            configs=None,
            max_samples=None,
            eval_mode=True,
        )
        curr_collate_fn = collate_fn_train
    else:
        raise ValueError(
            f"Unknown dataset: {cfg.dataset_name!r}. "
            f"Available: tapvid_davis_first, tapvid_kinetics_first, "
            f"tapvid_robotap_first, tapvid_stacking_first, robotrack-real, robotrack-sim"
        )

    # Creating the DataLoader object
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=curr_collate_fn,
    )

    # Timing and conducting the evaluation
    import time

    start = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        evaluate_result = evaluator.evaluate_sequence(
            predictor, test_dataloader, dataset_name=cfg.dataset_name
        )
    end = time.time()
    print(end - start)

    # Saving the evaluation results to a .json file
    evaluate_result = evaluate_result["avg"]
    print("evaluate_result", evaluate_result)
    result_file = os.path.join(cfg.exp_dir, f"result_eval_.json")
    evaluate_result["time"] = end - start
    print(f"Dumping eval results to {result_file}.")
    with open(result_file, "w") as f:
        json.dump(evaluate_result, f)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="default_config_eval", node=DefaultConfig)


@hydra.main(config_path="./configs/", config_name="default_config_eval")
def evaluate(cfg: DefaultConfig) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_idx)
    run_eval(cfg)


if __name__ == "__main__":
    evaluate()
