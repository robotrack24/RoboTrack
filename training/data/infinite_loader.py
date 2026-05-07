# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Infinite batch iterator for step-based training.

Handles:
- Automatic dataloader cycling (no StopIteration)
- gotit=False batch filtering
- Sampler epoch management for proper shuffling
- Robust checkpoint resume with batch skipping
"""

import logging
from typing import Iterator, Any, Optional

from torch.utils.data import DataLoader


class InfiniteBatchIterator:
    """
    Infinite iterator over a DataLoader.

    The CoTracker datasets return (sample, gotit) where gotit=False means
    the sample is invalid (e.g., not enough visible points). With multi-batch,
    invalid samples already have valid=0, so they contribute zero loss.
    This iterator passes all batches through and logs failure rates.

    When the underlying dataloader is exhausted, it automatically cycles
    to the next epoch with proper sampler seeding.

    Supports checkpoint resume by tracking batches consumed within the
    current epoch, allowing exact position restoration.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        start_epoch: int = 0,
        batches_to_skip: int = 0,
    ):
        """
        Initialize infinite batch iterator.

        Args:
            dataloader: PyTorch DataLoader to iterate over
            start_epoch: Starting epoch number (for checkpoint resume)
            batches_to_skip: Number of valid batches to skip on first epoch
                            (for mid-epoch checkpoint resume)
        """
        self.dataloader = dataloader
        self._epoch = start_epoch
        self._iter: Optional[Iterator] = None
        self._batches_to_skip = batches_to_skip
        self._batches_in_epoch = 0  # Batches consumed in current epoch
        self._failed_samples = 0
        self._total_samples = 0

    @property
    def epoch(self) -> int:
        """Current epoch number (incremented when new epoch starts)."""
        return self._epoch

    @property
    def batches_in_epoch(self) -> int:
        """Number of valid batches consumed in current epoch (for checkpointing)."""
        return self._batches_in_epoch

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            "epoch": self._epoch,
            "batches_in_epoch": self._batches_in_epoch,
        }

    def __iter__(self) -> "InfiniteBatchIterator":
        return self

    def __next__(self) -> Any:
        """
        Return next batch, cycling dataloader as needed.

        Invalid samples (gotit=False) are passed through — they already have
        valid=0 so they contribute zero loss. Failure rates are logged
        periodically.

        Returns:
            CoTrackerData batch

        Note:
            Never raises StopIteration - cycles indefinitely.
        """
        while True:
            if self._iter is None:
                self._start_new_epoch()

            try:
                batch, gotit = next(self._iter)
            except StopIteration:
                self._epoch += 1
                self._start_new_epoch()
                batch, gotit = next(self._iter)

            self._batches_in_epoch += 1

            # Log failure rate
            n_failed = sum(1 for g in gotit if not g)
            self._total_samples += len(gotit)
            if n_failed > 0:
                self._failed_samples += n_failed
            if self._batches_in_epoch % 100 == 0 and self._total_samples > 0:
                rate = self._failed_samples / self._total_samples
                logging.info(
                    f"Sample failure rate: {rate:.2%} "
                    f"({self._failed_samples}/{self._total_samples})"
                )

            # Skip batches for resume (only on first epoch after init)
            if self._batches_to_skip > 0:
                self._batches_to_skip -= 1
                logging.debug(
                    f"InfiniteBatchIterator: skipping batch for resume "
                    f"({self._batches_to_skip} remaining)"
                )
                continue

            # Skip failed batches to match original training behavior
            if not all(gotit):
                continue

            return batch

    def _start_new_epoch(self) -> None:
        """Start new epoch with proper sampler seeding."""
        # Set epoch on distributed sampler for proper shuffling
        if hasattr(self.dataloader, "sampler") and hasattr(
            self.dataloader.sampler, "set_epoch"
        ):
            self.dataloader.sampler.set_epoch(self._epoch)

        self._iter = iter(self.dataloader)
        self._batches_in_epoch = 0
        logging.debug(f"InfiniteBatchIterator: starting epoch {self._epoch}")
