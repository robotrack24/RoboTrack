"""
MolmoSpaces point-track dataset for CoTracker training and evaluation.

Loads .npz annotations (trajs_2d, visibility) with companion .mp4 videos from
``house_*`` directories. Extends CoTrackerDataset so training uses the same
augmentation and sampling pipeline as PointOdyssey.

Supports two on-disk layouts:

1. Flat (legacy single-config dump)::

    data_root/
      experiment_config_*.pkl
      house_<N>/
        episode_*_<camera>_point_tracks.npz
        episode_*_<camera>_batch_*_of_*.mp4

2. Nested mixture (CoTracker3Eval-style multi-config dump)::

    data_root/
      mixture_spec.json
      mixture_summary.json
      <ConfigName>/                # e.g. FrankaPickAndPlacePointTrack
        experiment_config_*.pkl
        house_<N>/
          episode_*_<camera>_point_tracks.npz
          episode_*_<camera>_batch_*_of_*.mp4

``house_*`` directories are discovered recursively, so additional intermediate
directories are tolerated. ``seq_name`` is set to the path from ``data_root``
(e.g. ``FrankaPickAndPlacePointTrack/house_74260/episode_00000000_<camera>``)
so configs are distinguishable in logs and metrics.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np
import torch

from cotracker.datasets.kubric_movif_dataset import CoTrackerDataset
from cotracker.datasets.utils import CoTrackerData

_NPZ_RE = re.compile(r"^(episode_\d{8})_(.+)_point_tracks\.npz$")


class MolmoSpacesDataset(CoTrackerDataset):
    """MolmoSpaces sequences with PointOdyssey-style ``getitem_helper``."""

    def __init__(
        self,
        data_root,
        crop_size=(2048, 2048),
        seq_len=2048,
        traj_per_sample=2048,
        sample_vis_last_frame=False,
        use_augs=False,
        full_frame_dropout_prob=0.0,
        full_frame_dropout_len=5,
        full_frame_dropout_mode="average",
        random_seq_len=False,
        random_first_frame=False,
        random_frame_rate=False,
        random_number_traj=False,
        only_first=False,
        split="train",
        max_samples=30000,
        cameras: list | None = None,
        configs: list | None = None,
        min_visible_frames: int = 5,
        spatial_crop_anchor: str = "uniform",
        resize_lim: tuple | list = (0.75, 1.25),
        spatial_precrop_area_frac_min: float | None = None,
        spatial_precrop_area_frac_max: float | None = None,
        eval_mode: bool = False,
    ):
        super().__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            sample_vis_last_frame=sample_vis_last_frame,
            use_augs=use_augs,
            full_frame_dropout_prob=full_frame_dropout_prob,
            full_frame_dropout_len=full_frame_dropout_len,
            full_frame_dropout_mode=full_frame_dropout_mode,
            spatial_crop_anchor=spatial_crop_anchor,
            resize_lim=resize_lim,
            spatial_precrop_area_frac_min=spatial_precrop_area_frac_min,
            spatial_precrop_area_frac_max=spatial_precrop_area_frac_max,
        )
        self.random_seq_len = random_seq_len
        self.random_first_frame = random_first_frame
        self.random_frame_rate = random_frame_rate
        self.random_number_traj = random_number_traj
        if isinstance(only_first, str):
            self.only_first = only_first.lower() == "first"
        else:
            self.only_first = bool(only_first)
        self.pad_bounds = [0, 25]
        self.resize_delta = 0.05
        self.max_crop_offset = 15
        self.split = split
        self.max_samples = None if max_samples is None else int(max_samples)
        self.min_visible_frames = int(min_visible_frames)
        self.eval_mode = bool(eval_mode)

        self.samples: list[dict] = []
        self._discover(cameras, configs)

        if not self.samples:
            raise FileNotFoundError(
                f"No MolmoSpaces sequences found under {data_root!r} "
                "(expect [<config>/]house_*/episode_*_*_point_tracks.npz with "
                "matching episode_*_*_batch_*_of_*.mp4). "
                "Set evaluation.molmospaces_data_root=... in Hydra or "
                "MOLMOSPACES_DATA_ROOT when using scripts/train_tillicum.sh."
            )

        if self.max_samples is not None and self.max_samples > 0:
            self.samples = self.samples[
                : min(len(self.samples), self.max_samples)
            ]
        if self.split == "valid":
            self.samples = self.samples[:30]
            assert use_augs is False

        print(
            "found %d MolmoSpaces sequences in %s"
            % (len(self.samples), self.data_root)
        )

    def _discover(self, cameras: list | None, configs: list | None = None):
        """Find ``house_*`` directories anywhere under ``data_root``.

        Supports both the flat legacy layout (``data_root/house_*/...``) and the
        nested mixture layout (``data_root/<ConfigName>/house_*/...``). When
        ``configs`` is provided, only ``house_*`` directories whose path
        contains one of those names as an ancestor component are kept.
        """
        root = Path(self.data_root)
        if not root.is_dir():
            logging.warning("MolmoSpacesDataset: data_root is not a directory: %s", root)
            return

        config_filter = set(configs) if configs else None
        scene_dirs = sorted(
            p for p in root.rglob("house_*") if p.is_dir()
        )

        for scene_dir in scene_dirs:
            try:
                rel_parts = scene_dir.relative_to(root).parts
            except ValueError:
                continue
            ancestor_parts = rel_parts[:-1]
            if config_filter is not None and not (
                set(ancestor_parts) & config_filter
            ):
                continue

            for npz_path in sorted(scene_dir.glob("*_point_tracks.npz")):
                m = _NPZ_RE.match(npz_path.name)
                if m is None:
                    continue
                episode_tag, camera = m.group(1), m.group(2)
                if cameras and camera not in cameras:
                    continue

                video_candidates = sorted(
                    scene_dir.glob(f"{episode_tag}_{camera}_batch_*_of_*.mp4")
                )
                if not video_candidates:
                    logging.warning("No video for %s, skipping", npz_path.name)
                    continue

                rel_dir = "/".join(rel_parts)
                self.samples.append(
                    {
                        "npz": str(npz_path),
                        "video": str(video_candidates[0]),
                        "seq_name": f"{rel_dir}/{episode_tag}_{camera}",
                    }
                )

    @staticmethod
    def _open_video(video_path: str):
        """Open video with decord and return the VideoReader."""
        import decord
        decord.bridge.set_bridge("native")
        return decord.VideoReader(video_path, ctx=decord.cpu())

    @staticmethod
    def _get_video_info(vr) -> tuple[int, int, int]:
        """Return (num_frames, height, width) from an open VideoReader."""
        n = len(vr)
        h, w, _ = vr[0].shape
        return n, int(h), int(w)

    @staticmethod
    def _load_frames_hwc(vr, indices: list[int]) -> np.ndarray:
        """Decode only the requested frames. Returns (N, H, W, 3) uint8."""
        frames = vr.get_batch(indices).asnumpy()  # (N, H, W, 3)
        return frames

    @staticmethod
    def _safe_subsample_frame_indices(
        total_frames: int, seq_req: int, frame_rate: int
    ) -> list[int]:
        """Chose ``seq_req`` frame indices stepped by ``frame_rate``, all in-range.

        The PointOdyssey pattern ``range(start, start + seq_req * rate, rate)`` can
        run past the end of the clip when ``seq_req * rate >= total_frames`` but
        the stride still pushes the last index beyond ``total_frames - 1``.
        """
        if total_frames <= 0 or seq_req <= 0:
            return []
        if seq_req >= total_frames:
            return list(range(total_frames))

        fr = max(1, int(frame_rate))
        while fr >= 1:
            max_start = total_frames - 1 - (seq_req - 1) * fr
            if max_start >= 0:
                break
            fr -= 1
        if fr < 1:
            fr = 1
        max_start = total_frames - 1 - (seq_req - 1) * fr
        if max_start < 0:
            return list(range(total_frames))

        if seq_req * fr < total_frames:
            hi = total_frames - seq_req * fr
            start_ind = int(np.random.choice(hi)) if hi > 0 else 0
        else:
            start_ind = int(np.random.choice(max_start + 1))

        out = list(range(start_ind, start_ind + seq_req * fr, fr))
        if len(out) != seq_req or (out and out[-1] >= total_frames):
            out = list(np.linspace(0, total_frames - 1, seq_req, dtype=int))
            out = [int(np.clip(i, 0, total_frames - 1)) for i in out]
        return out

    def getitem_helper(self, index):
        gotit = True
        info = self.samples[index]
        seq_name = info["seq_name"]

        anno_dict = np.load(info["npz"], allow_pickle=True)
        try:
            traj_2d_all = np.asarray(anno_dict["trajs_2d"], dtype=np.float32)
            visibs_all = np.asarray(anno_dict["visibility"], dtype=np.float32)
        finally:
            anno_dict.close()

        total_frames = int(traj_2d_all.shape[0])
        vr = self._open_video(info["video"])
        t_vid, h_vid, w_vid = self._get_video_info(vr)
        t_min = min(total_frames, t_vid)
        traj_2d_all = traj_2d_all[:t_min].copy()
        visibs_all = visibs_all[:t_min].copy()
        total_frames = t_min
        oob = (
            (traj_2d_all[:, :, 0] < 0)
            | (traj_2d_all[:, :, 0] >= w_vid)
            | (traj_2d_all[:, :, 1] < 0)
            | (traj_2d_all[:, :, 1] >= h_vid)
        )
        visibs_all = visibs_all.copy()
        visibs_all[oob] = 0.0

        vis_count = visibs_all.sum(axis=0)
        keep = vis_count >= self.min_visible_frames
        traj_2d_all = traj_2d_all[:, keep]
        visibs_all = visibs_all[:, keep]

        if traj_2d_all.shape[1] == 0:
            return (
                CoTrackerData(
                    video=torch.zeros(
                        (self.seq_len, 3, self.crop_size[0], self.crop_size[1])
                    ),
                    trajectory=torch.zeros((self.seq_len, self.traj_per_sample, 2)),
                    visibility=torch.zeros(
                        (self.seq_len, self.traj_per_sample), dtype=torch.bool
                    ),
                    valid=torch.zeros((self.seq_len, self.traj_per_sample)),
                    seq_name=seq_name,
                ),
                False,
            )

        frame_rate = 1
        final_num_traj = self.traj_per_sample
        crop_size = self.crop_size

        min_num_traj = 1
        assert self.traj_per_sample >= min_num_traj
        if self.random_seq_len and self.random_number_traj:
            final_num_traj = np.random.randint(min_num_traj, self.traj_per_sample)
            alpha = final_num_traj / float(self.traj_per_sample)
            seq_len = int(alpha * 10 + (1 - alpha) * self.seq_len)
            seq_len = np.random.randint(seq_len - 2, seq_len + 2)
            if self.random_frame_rate:
                frame_rate = np.random.randint(1, int((120 / seq_len)) + 1)
        elif self.random_number_traj:
            final_num_traj = np.random.randint(min_num_traj, self.traj_per_sample)
            alpha = final_num_traj / float(self.traj_per_sample)
            seq_len = 8 * int(alpha * 2 + (1 - alpha) * self.seq_len // 8)
            if self.random_frame_rate:
                frame_rate = np.random.randint(1, int((120 / seq_len)) + 1)
        elif self.random_seq_len:
            seq_len = np.random.randint(int(self.seq_len / 2), self.seq_len)
            if self.random_frame_rate:
                frame_rate = np.random.randint(1, int((120 / seq_len)) + 1)
        else:
            seq_len = self.seq_len
            if self.random_frame_rate:
                frame_rate = np.random.randint(1, int((120 / seq_len)) + 1)

        no_augs = False
        random_first_ind = None
        seq_req = seq_len
        if seq_req < total_frames:
            if self.random_first_frame:
                random_first_ind = int(np.random.choice(total_frames))
            frame_indices = self._safe_subsample_frame_indices(
                total_frames, seq_req, frame_rate
            )
        else:
            frame_indices = list(range(total_frames))

        seq_len = len(frame_indices)
        if seq_len == 0:
            return (
                CoTrackerData(
                    video=torch.zeros(
                        (self.seq_len, 3, self.crop_size[0], self.crop_size[1])
                    ),
                    trajectory=torch.zeros((self.seq_len, self.traj_per_sample, 2)),
                    visibility=torch.zeros(
                        (self.seq_len, self.traj_per_sample), dtype=torch.bool
                    ),
                    valid=torch.zeros((self.seq_len, self.traj_per_sample)),
                    seq_name=seq_name,
                ),
                False,
            )

        load_indices = sorted(set(frame_indices) | (
            {int(random_first_ind)} if random_first_ind is not None else set()
        ))
        decoded = self._load_frames_hwc(vr, load_indices)
        del vr
        idx_to_pos = {idx: pos for pos, idx in enumerate(load_indices)}

        rgbs = np.stack([decoded[idx_to_pos[i]] for i in frame_indices])
        traj_2d = traj_2d_all[frame_indices]
        visibility = visibs_all[frame_indices]
        visibility = visibility > 0.5

        if random_first_ind is not None:
            rfi = int(random_first_ind)
            rgbs[0] = decoded[idx_to_pos[rfi]]
            traj_2d[0] = traj_2d_all[rfi]
            visibility[0] = visibs_all[rfi] > 0.5

        assert len(rgbs) == seq_len

        if not no_augs:
            if self.use_augs:
                rgbs, traj_2d, visibility = self.add_photometric_augs(
                    rgbs, traj_2d, visibility, replace=False
                )
                rgbs, traj_2d, visibility = self.add_full_frame_dropout(
                    rgbs, traj_2d, visibility
                )
                seq_len = len(rgbs)
                rgbs, traj_2d = self.add_spatial_augs(
                    rgbs, traj_2d, visibility, crop_size
                )
            else:
                rgbs, traj_2d = self.crop(rgbs, traj_2d, crop_size)

        visibility = visibility.astype(np.bool_)
        visibility[traj_2d[:, :, 0] > crop_size[1] - 1] = False
        visibility[traj_2d[:, :, 0] < 0] = False
        visibility[traj_2d[:, :, 1] > crop_size[0] - 1] = False
        visibility[traj_2d[:, :, 1] < 0] = False

        visibility_t = torch.from_numpy(visibility)
        traj_2d_t = torch.from_numpy(traj_2d)

        crop_tensor = torch.tensor(crop_size).flip(0)[None, None] / 2.0
        close_pts_inds = torch.all(
            torch.linalg.vector_norm(traj_2d_t[..., :2] - crop_tensor, dim=-1)
            < 1000.0,
            dim=0,
        )
        traj_2d_t = traj_2d_t[:, close_pts_inds]
        visibility_t = visibility_t[:, close_pts_inds]

        if self.eval_mode:
            # Eval protocol matches DROID / TAP-Vid: evaluate every surviving
            # GT point in its original order, each anchored at its first
            # visible frame. No randperm, no first-vs-mid candidate pool, no
            # duplicates -- so the per-sequence point count is deterministic
            # and matches what's in the .npz annotations (modulo points that
            # got dropped by the crop_size out-of-bounds filter).
            n_close = traj_2d_t.shape[1]
            visible_inds_sampled = torch.arange(n_close, dtype=torch.long)
            if n_close == 0:
                gotit = False
        else:
            if self.only_first:
                visibile_pts_inds = (visibility_t[0]).nonzero(as_tuple=False)[:, 0]
            else:
                visibile_pts_first_frame_inds = (visibility_t[0]).nonzero(
                    as_tuple=False
                )[:, 0]
                visibile_pts_mid_frame_inds = (visibility_t[seq_len // 2]).nonzero(
                    as_tuple=False
                )[:, 0]
                visibile_pts_inds = torch.cat(
                    (visibile_pts_first_frame_inds, visibile_pts_mid_frame_inds), dim=0
                )
                if self.sample_vis_last_frame:
                    visibile_pts_last_frame_inds = (visibility_t[seq_len - 1]).nonzero(
                        as_tuple=False
                    )[:, 0]
                    visibile_pts_inds = torch.cat(
                        (visibile_pts_inds, visibile_pts_last_frame_inds), dim=0
                    )
            point_inds = torch.randperm(len(visibile_pts_inds))[: self.traj_per_sample]
            if len(point_inds) == 0:
                gotit = False

            visible_inds_sampled = visibile_pts_inds[point_inds]

        trajs = traj_2d_t[:, visible_inds_sampled].float()
        visibles = visibility_t[:, visible_inds_sampled]
        valids = torch.ones_like(visibles, dtype=torch.float32)

        trajs = trajs[:, :final_num_traj]
        visibles = visibles[:, :final_num_traj].bool()
        valids = valids[:, :final_num_traj]

        n_pts = trajs.shape[1]
        ar = torch.arange(n_pts, dtype=torch.long)
        first_vis = visibles.long().argmax(dim=0)
        query_points = torch.stack(
            [
                first_vis.float(),
                trajs[first_vis, ar, 1],
                trajs[first_vis, ar, 0],
            ],
            dim=-1,
        )

        rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float()

        sample = CoTrackerData(
            video=rgbs,
            trajectory=trajs,
            visibility=visibles,
            valid=valids,
            seq_name=seq_name,
            query_points=query_points,
        )
        return sample, gotit

    def __len__(self):
        return len(self.samples)
