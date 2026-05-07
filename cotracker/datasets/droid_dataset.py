"""
DROID point-track dataset for CoTracker evaluation.

Each sequence is a directory containing:
  - video.mp4
  - point_tracks.npz with keys:
      trajs_2d:      (T, N, 2) float32 — pixel coordinates (x, y)
      visibility:    (T, N) float32 — 1.0 = visible, 0.0 = occluded
      query_frames:  (N,) int32 — query frame index per point
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from cotracker.datasets.utils import CoTrackerData


class DroidDataset(torch.utils.data.Dataset):
    """Eval-only dataset for DROID-style per-sequence directories."""

    def __init__(self, data_root: str, resize_to: tuple[int, int] | None = (256, 256)):
        self.data_root = data_root
        self.resize_to = resize_to
        self.sequences: list[str] = sorted(
            d for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
            and os.path.exists(os.path.join(data_root, d, "point_tracks.npz"))
        )
        if not self.sequences:
            raise FileNotFoundError(
                f"No sequences found under {data_root!r}. "
                "Expected <seq>/point_tracks.npz + <seq>/video.mp4."
            )
        print(f"found {len(self.sequences)} DROID sequences in {data_root}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> CoTrackerData:
        seq_name = self.sequences[index]
        seq_dir = Path(self.data_root) / seq_name

        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(seq_dir / "video.mp4"), ctx=decord.cpu())
        frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # (T, H, W, 3)

        npz = np.load(str(seq_dir / "point_tracks.npz"))
        trajs_2d = npz["trajs_2d"].astype(np.float32)       # (T, N, 2)
        visibility = npz["visibility"].astype(np.float32)    # (T, N)
        query_frames = npz["query_frames"].astype(np.int64)  # (N,)

        T_vid = frames.shape[0]
        T_ann = trajs_2d.shape[0]
        T = min(T_vid, T_ann)
        frames = frames[:T]
        trajs_2d = trajs_2d[:T]
        visibility = visibility[:T]

        orig_h, orig_w = frames.shape[1], frames.shape[2]

        if self.resize_to is not None:
            import mediapy as media
            frames = media.resize_video(frames, self.resize_to)
            new_h, new_w = self.resize_to
            scale_x = (new_w - 1) / max(orig_w - 1, 1)
            scale_y = (new_h - 1) / max(orig_h - 1, 1)
            trajs_2d = trajs_2d.copy()
            trajs_2d[..., 0] *= scale_x
            trajs_2d[..., 1] *= scale_y

        N = trajs_2d.shape[1]
        query_points = np.zeros((N, 3), dtype=np.float32)
        query_points[:, 0] = query_frames
        for i in range(N):
            t = int(query_frames[i])
            if 0 <= t < T:
                query_points[i, 1] = trajs_2d[t, i, 1]  # y
                query_points[i, 2] = trajs_2d[t, i, 0]  # x

        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float()     # (T, 3, H, W)
        trajs = torch.from_numpy(trajs_2d).float()                       # (T, N, 2)
        visibles = torch.from_numpy(visibility > 0.5).permute(1, 0)      # wait, visibility is (T, N)

        # TapVid evaluator expects: trajs (T, N, 2), visibles (T, N), query_points (N, 3)
        visibles = torch.from_numpy(visibility > 0.5)                    # (T, N)

        return CoTrackerData(
            video=rgbs,
            trajectory=trajs,
            visibility=visibles,
            seq_name=seq_name,
            query_points=torch.from_numpy(query_points),
        )
