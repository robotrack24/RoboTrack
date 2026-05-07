# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ComposedDataset for CoTracker training.
Wraps existing datasets with support for:
- Multiple data sources
- Dataset repetition (matching original ConcatDataset pattern)
- Weighted sampling (optional)
"""

from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, DistributedSampler
from hydra.utils import instantiate


class ComposedDataset(Dataset):
    """
    Composes multiple datasets with optional repetition and weighting.

    Supports Hydra instantiation of nested datasets.

    Mixing is controlled by two knobs:
    - ``repeat``: global repeat applied to the entire combined dataset.
    - ``sampling_weights``: per-dataset weights that control how many
      *effective copies* of each dataset are included.  Weights are
      normalised so the smallest weight maps to 1 copy, then each
      dataset is repeated ``round(w_i / w_min)`` times before
      concatenation.  This is a simple, deterministic alternative to
      a ``WeightedRandomSampler`` and works with ``DistributedSampler``.

    Example config (2:1 ratio of dataset A to B, then 4x global repeat):
        datasets:
          - ...  # dataset A
          - ...  # dataset B
        sampling_weights: [2.0, 1.0]
        repeat: 4
    """

    def __init__(
        self,
        datasets: List[Dict[str, Any]],
        repeat: int = 1,
        sampling_weights: Optional[List[float]] = None,
        weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__()

        self.repeat = repeat
        # Accept legacy `weights` key as alias
        self.sampling_weights = sampling_weights or weights

        # Instantiate each dataset from config
        instantiated_datasets = []
        for ds_config in datasets:
            ds = instantiate(ds_config, _recursive_=False)
            instantiated_datasets.append(ds)

        self._datasets = instantiated_datasets

        # Compute per-dataset repeat counts from weights
        if self.sampling_weights is not None:
            assert len(self.sampling_weights) == len(instantiated_datasets), (
                f"sampling_weights length ({len(self.sampling_weights)}) != "
                f"number of datasets ({len(instantiated_datasets)})"
            )
            w_min = min(w for w in self.sampling_weights if w > 0)
            per_ds_repeats = [max(1, round(w / w_min)) for w in self.sampling_weights]
        else:
            per_ds_repeats = [1] * len(instantiated_datasets)

        self._per_ds_repeats = per_ds_repeats

        # Build repeated list, then apply global repeat
        repeated = []
        for ds, r in zip(instantiated_datasets, per_ds_repeats):
            repeated.extend([ds] * r)

        if repeat > 1:
            self.combined = ConcatDataset(repeat * repeated)
        else:
            self.combined = ConcatDataset(repeated)

        # Log mixing summary
        self._log_mixing_summary()

    def _log_mixing_summary(self):
        """Print dataset sizes and effective sampling rates."""
        import logging
        logger = logging.getLogger(__name__)

        total_raw = sum(len(ds) for ds in self._datasets)
        total_effective = len(self.combined)

        logger.info("=" * 60)
        logger.info("COMPOSED DATASET MIXING SUMMARY")
        logger.info("=" * 60)
        for i, ds in enumerate(self._datasets):
            raw_len = len(ds)
            ds_repeat = self._per_ds_repeats[i]
            effective_len = raw_len * ds_repeat * self.repeat
            weight = self.sampling_weights[i] if self.sampling_weights else 1.0
            rate = effective_len / total_effective * 100 if total_effective > 0 else 0
            name = getattr(ds, '__class__', type(ds)).__name__
            logger.info(
                f"  [{i}] {name}: {raw_len:,} samples "
                f"(weight={weight:.2f}, repeat={ds_repeat}x, "
                f"effective={effective_len:,}, rate={rate:.1f}%)"
            )
        logger.info(f"  Global repeat: {self.repeat}x")
        logger.info(f"  Total effective samples: {total_effective:,}")
        logger.info("=" * 60)

    def __len__(self) -> int:
        return len(self.combined)

    def __getitem__(self, idx: int):
        return self.combined[idx]

    @property
    def datasets(self) -> List[Dataset]:
        """Get list of underlying datasets."""
        return self._datasets


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn=None,
    distributed: bool = True,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader with optional distributed sampling.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        collate_fn: Custom collate function
        distributed: Whether to use distributed sampler
        rank: Distributed rank
        world_size: Number of distributed processes
        seed: Random seed for reproducibility

    Returns:
        DataLoader instance
    """
    from training.distributed import seed_worker

    sampler = None
    if distributed and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        # Don't shuffle in DataLoader when using sampler
        shuffle = False

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        sampler=sampler,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=num_workers > 0,
    )

    return loader
