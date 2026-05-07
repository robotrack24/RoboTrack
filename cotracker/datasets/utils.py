# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import dataclasses
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Optional, Dict
import numpy as np
import os


@dataclass(eq=False)
class CoTrackerData:
    """
    Dataclass for storing video tracks data.
    """

    video: torch.Tensor  # B, S, C, H, W
    trajectory: torch.Tensor  # B, S, N, 2
    visibility: torch.Tensor  # B, S, N
    # optional data
    valid: Optional[torch.Tensor] = None  # B, S, N
    segmentation: Optional[torch.Tensor] = None  # B, S, 1, H, W
    seq_name: Optional[str] = None
    query_points: Optional[torch.Tensor] = None  # TapVID evaluation format
    transforms: Optional[Dict[str, Any]] = None
    aug_video: Optional[torch.Tensor] = None


def collate_fn(batch):
    """
    Collate function for video tracks data.
    """
    video = torch.stack([b.video for b in batch], dim=0)
    trajectory = torch.stack([b.trajectory for b in batch], dim=0)
    visibility = torch.stack([b.visibility for b in batch], dim=0)
    query_points = segmentation = None
    if batch[0].query_points is not None:
        query_points = torch.stack([b.query_points for b in batch], dim=0)
    if batch[0].segmentation is not None:
        segmentation = torch.stack([b.segmentation for b in batch], dim=0)
    seq_name = [b.seq_name for b in batch]

    return CoTrackerData(
        video=video,
        trajectory=trajectory,
        visibility=visibility,
        segmentation=segmentation,
        seq_name=seq_name,
        query_points=query_points,
    )


def collate_fn_train(batch, random_seq_len=False):
    """
    Collate function for video tracks data during training.

    All samples must have the same T and N (no padding). If random_seq_len
    is True, a random length is chosen per-batch and all samples are
    truncated uniformly — this replaces per-sample random_seq_len in the
    dataset, ensuring consistent shapes without padding.

    Args:
        batch: List of (CoTrackerData, gotit) tuples
        random_seq_len: Whether to randomly truncate sequence length per-batch
    """
    gotit = [gotit for _, gotit in batch]
    samples = [b for b, _ in batch]

    video = torch.stack([s.video for s in samples], dim=0)
    trajectory = torch.stack([s.trajectory for s in samples], dim=0)
    visibility = torch.stack([s.visibility for s in samples], dim=0)
    valid = torch.stack([s.valid for s in samples], dim=0)

    seq_name = [s.seq_name for s in samples]
    query_points = transforms = aug_video = None
    if samples[0].query_points is not None:
        query_points = torch.stack([s.query_points for s in samples], dim=0)

    if samples[0].transforms is not None:
        transforms = [s.transforms for s in samples]

    if samples[0].aug_video is not None:
        aug_video = torch.stack([s.aug_video for s in samples], dim=0)

    if random_seq_len:
        min_len = int(video.shape[1] / 2)
        max_len = video.shape[1]
        seq_len = torch.randint(min_len, max_len, (1,)).item()
        video = video[:, :seq_len]
        trajectory = trajectory[:, :seq_len]
        visibility = visibility[:, :seq_len]
        valid = valid[:, :seq_len]

    return (
        CoTrackerData(
            video=video,
            trajectory=trajectory,
            visibility=visibility,
            valid=valid,
            seq_name=seq_name,
            query_points=query_points,
            aug_video=aug_video,
            transforms=transforms,
        ),
        gotit,
    )


def try_to_cuda(t: Any) -> Any:
    """
    Try to move the input variable `t` to a cuda device.

    Args:
        t: Input.

    Returns:
        t_cuda: `t` moved to a cuda device, if supported.
    """
    try:
        t = t.float().cuda()
    except AttributeError:
        pass
    return t


def dataclass_to_cuda_(obj):
    """
    Move all contents of a dataclass to cuda inplace if supported.

    Args:
        batch: Input dataclass.

    Returns:
        batch_cuda: `batch` moved to a cuda device, if supported.
    """
    for f in dataclasses.fields(obj):
        setattr(obj, f.name, try_to_cuda(getattr(obj, f.name)))
    return obj
