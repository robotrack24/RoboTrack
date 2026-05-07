# RoboTrack

RoboTrack is a benchmark for evaluating point tracking models in robotic manipulation settings. It provides simulated and real-world datasets of robotic scenes with dense ground-truth point tracks, and supports training and evaluation of multiple tracking architectures including CoTracker3, TAPNext, and AllTracker.

  
Dataset: [RoboTrack24/RoboTrack on HuggingFace](https://huggingface.co/datasets/RoboTrack24/RoboTrack)

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate robotrack
```

### 2. Install the package

```bash
pip install -e .
```

## Evaluation

RoboTrack supports evaluating three model architectures — **CoTracker3**, **TAPNext**, and **AllTracker** — across six benchmark datasets.

### Datasets


| Dataset                 | Source                                                                  | Auto-download |
| ----------------------- | ----------------------------------------------------------------------- | ------------- |
| `tapvid_davis_first`    | [TAP-Vid](https://github.com/google-deepmind/tapnet)                    | No            |
| `tapvid_kinetics_first` | [TAP-Vid](https://github.com/google-deepmind/tapnet)                    | No            |
| `tapvid_robotap`        | [TAP-Vid](https://github.com/google-deepmind/tapnet)                    | No            |
| `tapvid_stacking`       | [TAP-Vid](https://github.com/google-deepmind/tapnet)                    | No            |
| `robotrack-real`        | [RoboTrack-Real](https://huggingface.co/datasets/RoboTrack24/RoboTrack) | Yes           |
| `robotrack-sim`         | [RoboTrack-Sim](https://huggingface.co/datasets/RoboTrack24/RoboTrack)  | Yes           |


The `robotrack-real` and `robotrack-sim` datasets are loaded automatically from HuggingFace. For the TAP-Vid datasets, download them manually and set `dataset_root` to the directory containing them.

### Running evaluation

Evaluation uses Hydra configs in `cotracker/evaluation/configs/`. The general pattern is:

```bash
python cotracker/evaluation/evaluate.py \
    --config-name=eval_<dataset> \
    checkpoint=<path_to_checkpoint> \
    model_type=<model_type>
```

**CoTracker3** (offline):

```bash
python cotracker/evaluation/evaluate.py \
    --config-name=eval_tapvid_davis_first \
    checkpoint=<path_to_cotracker3.pth> \
    model_type=cotracker \
    offline_model=true
```

**TAPNext**:

```bash
python cotracker/evaluation/evaluate.py \
    --config-name=eval_robotrack_real \
    checkpoint=<path_to_tapnext.npz> \
    model_type=tapnext
```

**AllTracker**:

```bash
python cotracker/evaluation/evaluate.py \
    --config-name=eval_robotrack_sim \
    checkpoint=<path_to_alltracker.pth> \
    model_type=alltracker
```

Available config names: `eval_tapvid_davis_first`, `eval_tapvid_kinetics_first`, `eval_tapvid_robotap_first`, `eval_tapvid_stacking_first`, `eval_robotrack_real`, `eval_robotrack_sim`.

## Training

Training uses `torchrun` with Hydra configuration. The default config trains a CoTracker3 offline model on the Kubric dataset.

### Quick start

```bash
# Single GPU, default settings
bash scripts/train.sh

# Multi-GPU
bash scripts/train.sh my_experiment 4

# With a different model
bash scripts/train.sh my_experiment 4 cotracker3_online
```

### Direct launch

```bash
# Single GPU
torchrun --standalone --nproc_per_node=1 training/launch.py exp_name=my_exp

# 4 GPUs
torchrun --standalone --nproc_per_node=4 training/launch.py exp_name=my_exp

# Override config values
torchrun --standalone --nproc_per_node=4 training/launch.py \
    exp_name=my_exp \
    model=cotracker3_offline \
    training.num_steps=50000 \
    training.gradient_accumulation_steps=8
```

### Logging with W&B

W&B logging is disabled by default. To enable it, pass your project and entity:

```bash
bash scripts/train.sh my_exp 1 cotracker3_offline \
    logging.use_wandb=true \
    logging.wandb_project=my_project \
    logging.wandb_entity=my_team
```

### Key training options


| Option                                 | Default | Description                      |
| -------------------------------------- | ------- | -------------------------------- |
| `training.num_steps`                   | 50000   | Total optimizer steps            |
| `training.batch_size`                  | 1       | Per-GPU batch size               |
| `training.gradient_accumulation_steps` | 1       | Micro-batches per optimizer step |
| `training.sequence_len`                | 60      | Frames per training clip         |
| `training.traj_per_sample`             | 512     | Point trajectories per sample    |
| `training.precision`                   | bf16    | Mixed precision (bf16 or fp16)   |
| `checkpoint.save_every_n_steps`        | 5000    | Checkpoint frequency             |
| `checkpoint.resume_from`               | null    | Path to resume training          |


Configs are in `training/config/`. The main config is `kubric.yaml` (Kubric-only training), with model-specific configs in `training/config/model/`. To train on Kubric mixed with RoboTrack-Data data, use the `mlspaces_1` config:

```bash
bash scripts/train.sh my_exp 4 cotracker3_offline --config-name=mlspaces_1
```

## Acknowledgements

- [CoTracker3](https://github.com/facebookresearch/co-tracker) — Original model and codebase
- [TAPNet](https://github.com/google-deepmind/tapnet) — TAP-Vid benchmarks and TAPNext
- [AllTracker](https://github.com/ShenhanQian/alltracker) — AllTracker architecture

## License

CC-BY-NC (see [LICENSE](LICENSE) for details).