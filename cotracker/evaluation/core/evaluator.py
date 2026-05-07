# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import os
from typing import Optional
import torch
from tqdm import tqdm
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from cotracker.datasets.utils import dataclass_to_cuda_
from cotracker.utils.visualizer import Visualizer
from cotracker.models.core.model_utils import reduce_masked_mean
from cotracker.evaluation.core.eval_utils import compute_tapvid_metrics
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
import logging


class Evaluator:
    """
    A class defining the CoTracker evaluator.
    """

    def __init__(self, exp_dir) -> None:
        # Visualization
        self.exp_dir = exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        self.visualization_filepaths = defaultdict(lambda: defaultdict(list))
        self.visualize_dir = os.path.join(exp_dir, "visualisations")

    def compute_metrics(self, metrics, sample, pred_trajectory, dataset_name):
        if isinstance(pred_trajectory, tuple):
            pred_trajectory, pred_visibility = pred_trajectory
        else:
            pred_visibility = None
        if (
            "tapvid" in dataset_name
            or dataset_name == "robotrack-sim"
            or dataset_name == "robotrack-real"
        ):
            B, T, N, D = sample.trajectory.shape
            traj = sample.trajectory.clone()
            thr = 0.6

            if pred_visibility is None:
                logging.warning("visibility is NONE")
                pred_visibility = torch.zeros_like(sample.visibility)

            if not pred_visibility.dtype == torch.bool:
                pred_visibility = pred_visibility > thr

            query_points = sample.query_points.clone().cpu().numpy()

            pred_visibility = pred_visibility[:, :, :N]
            pred_trajectory = pred_trajectory[:, :, :N]

            gt_tracks = traj.permute(0, 2, 1, 3).cpu().numpy()
            gt_occluded = (
                torch.logical_not(sample.visibility.clone().permute(0, 2, 1))
                .cpu()
                .numpy()
            )

            pred_occluded = (
                torch.logical_not(pred_visibility.clone().permute(0, 2, 1))
                .cpu()
                .numpy()
            )
            pred_tracks = pred_trajectory.permute(0, 2, 1, 3).cpu().numpy()

            # Normalize all coordinates to 256x256 pixel space before metric
            # computation. compute_tapvid_metrics uses fixed pixel thresholds
            # [1, 2, 4, 8, 16] and the standard TAP-Vid protocol assumes a
            # 256x256 reference frame. Without this rescale, datasets loaded
            # at native resolution (e.g. DROID with resize_to=None) get
            # judged against thresholds ~5x stricter than 256-space evals,
            # producing wildly different Jaccard / pts_within numbers for
            # the same predictions. No-op when the input is already 256x256.
            H, W = sample.video.shape[-2:]
            sx = (256 - 1) / max(int(W) - 1, 1)
            sy = (256 - 1) / max(int(H) - 1, 1)
            if sx != 1.0 or sy != 1.0:
                gt_tracks = gt_tracks.copy()
                gt_tracks[..., 0] *= sx
                gt_tracks[..., 1] *= sy
                pred_tracks = pred_tracks.copy()
                pred_tracks[..., 0] *= sx
                pred_tracks[..., 1] *= sy
                # query_points are stored as (t, y, x); rescale y and x.
                query_points = query_points.copy()
                query_points[..., 1] *= sy
                query_points[..., 2] *= sx

            out_metrics = compute_tapvid_metrics(
                query_points,
                gt_occluded,
                gt_tracks,
                pred_occluded,
                pred_tracks,
                query_mode="strided" if "strided" in dataset_name else "first",
            )

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = np.mean(
                    [v[metric_name] for k, v in metrics.items() if k != "avg"]
                )

            logging.info(f"Metrics: {out_metrics}")
            logging.info(f"avg: {metrics['avg']}")
            print("metrics", out_metrics)
            print("avg", metrics["avg"])
        elif dataset_name == "dynamic_replica" or dataset_name == "pointodyssey":
            *_, N, _ = sample.trajectory.shape
            B, T, N = sample.visibility.shape
            H, W = sample.video.shape[-2:]
            device = sample.video.device

            out_metrics = {}

            d_vis_sum = d_occ_sum = d_sum_all = 0.0
            thrs = [1, 2, 4, 8, 16]
            sx_ = (W - 1) / 255.0
            sy_ = (H - 1) / 255.0
            sc_py = np.array([sx_, sy_]).reshape([1, 1, 2])
            sc_pt = torch.from_numpy(sc_py).float().to(device)
            __, first_visible_inds = torch.max(sample.visibility, dim=1)

            frame_ids_tensor = torch.arange(T, device=device)[None, :, None].repeat(
                B, 1, N
            )
            start_tracking_mask = frame_ids_tensor > (first_visible_inds.unsqueeze(1))

            for thr in thrs:
                d_ = (
                    torch.norm(
                        pred_trajectory[..., :2] / sc_pt
                        - sample.trajectory[..., :2] / sc_pt,
                        dim=-1,
                    )
                    < thr
                ).float()  # B,S-1,N
                d_occ = (
                    reduce_masked_mean(
                        d_, (1 - sample.visibility) * start_tracking_mask
                    ).item()
                    * 100.0
                )
                d_occ_sum += d_occ
                out_metrics[f"accuracy_occ_{thr}"] = d_occ

                d_vis = (
                    reduce_masked_mean(
                        d_, sample.visibility * start_tracking_mask
                    ).item()
                    * 100.0
                )
                d_vis_sum += d_vis
                out_metrics[f"accuracy_vis_{thr}"] = d_vis

                d_all = reduce_masked_mean(d_, start_tracking_mask).item() * 100.0
                d_sum_all += d_all
                out_metrics[f"accuracy_{thr}"] = d_all

            d_occ_avg = d_occ_sum / len(thrs)
            d_vis_avg = d_vis_sum / len(thrs)
            d_all_avg = d_sum_all / len(thrs)

            sur_thr = 50
            dists = torch.norm(
                pred_trajectory[..., :2] / sc_pt - sample.trajectory[..., :2] / sc_pt,
                dim=-1,
            )  # B,S,N
            dist_ok = 1 - (dists > sur_thr).float() * sample.visibility  # B,S,N
            survival = torch.cumprod(dist_ok, dim=1)  # B,S,N
            out_metrics["survival"] = torch.mean(survival).item() * 100.0

            out_metrics["accuracy_occ"] = d_occ_avg
            out_metrics["accuracy_vis"] = d_vis_avg
            out_metrics["accuracy"] = d_all_avg

            metrics[sample.seq_name[0]] = out_metrics
            for metric_name in out_metrics.keys():
                if "avg" not in metrics:
                    metrics["avg"] = {}
                metrics["avg"][metric_name] = float(
                    np.mean([v[metric_name] for k, v in metrics.items() if k != "avg"])
                )

            logging.info(f"Metrics: {out_metrics}")
            logging.info(f"avg: {metrics['avg']}")
            print("metrics", out_metrics)
            print("avg", metrics["avg"])

    @torch.inference_mode()
    def evaluate_sequence(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        dataset_name: str,
        train_mode=False,
        visualize_every: int = 50,
        writer: Optional[SummaryWriter] = None,
        step: Optional[int] = 0,
        memory_profile_dir: Optional[str] = None,
    ):
        metrics = {}

        vis = Visualizer(
            save_dir=self.exp_dir,
            fps=7,
        )

        memory_profiled = False

        for ind, sample in enumerate(tqdm(test_dataloader)):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if isinstance(sample, tuple):
                sample, gotit = sample
                if not all(gotit):
                    print("batch is None")
                    continue
            
            # Log the video/sequence name being evaluated
            seq_name = getattr(sample, 'seq_name', None)
            if seq_name is not None:
                seq_name_str = seq_name[0] if isinstance(seq_name, (list, tuple)) else seq_name
                logging.info(f"Evaluating video {ind + 1}/{len(test_dataloader)}: {seq_name_str}")
            else:
                logging.info(f"Evaluating video {ind + 1}/{len(test_dataloader)}")
            
            if torch.cuda.is_available():
                dataclass_to_cuda_(sample)
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            if (
                not train_mode
                and hasattr(model, "sequence_len")
                and (sample.visibility[:, : model.sequence_len].sum() == 0)
            ):
                print(f"skipping batch {ind}")
                continue

            if (
                "tapvid" in dataset_name
                or dataset_name == "robotrack-sim"
                or dataset_name == "robotrack-real"
            ):
                queries = sample.query_points.clone().float()

                queries = torch.stack(
                    [
                        queries[:, :, 0],
                        queries[:, :, 2],
                        queries[:, :, 1],
                    ],
                    dim=2,
                ).to(device)
            else:
                queries = torch.cat(
                    [
                        torch.zeros_like(sample.trajectory[:, 0, :, :1]),
                        sample.trajectory[:, 0],
                    ],
                    dim=2,
                ).to(device)

            # CowTracker wrapper currently supports first-frame anchored queries only.
            # Filter query set in evaluation (instead of model forward) and keep GT aligned.
            if hasattr(model, "model") and hasattr(model.model, "cowtracker"):
                keep_mask = (queries[:, :, 0].long() == 0).all(dim=0)
                if not torch.all(keep_mask):
                    queries = queries[:, keep_mask]
                    if hasattr(sample, "trajectory"):
                        sample.trajectory = sample.trajectory[:, :, keep_mask]
                    if hasattr(sample, "visibility"):
                        sample.visibility = sample.visibility[:, :, keep_mask]
                    if hasattr(sample, "query_points"):
                        sample.query_points = sample.query_points[:, keep_mask]
                if queries.shape[1] == 0:
                    logging.warning(
                        f"Skipping sequence {ind}: no frame-0 queries after filtering for CowTracker."
                    )
                    continue

            # Start memory recording for the first video if profiling requested
            do_mem_profile = (
                memory_profile_dir is not None
                and not memory_profiled
                and torch.cuda.is_available()
            )
            if do_mem_profile:
                torch.cuda.memory._record_memory_history(max_entries=100000)
                torch.cuda.reset_peak_memory_stats()
                logging.info("Memory profiling started for this video")

            pred_tracks = model(sample.video, queries)

            if do_mem_profile:
                # Full CUDA memory summary from the allocator
                logging.info(
                    f"CUDA memory summary after video {ind}:\n"
                    + torch.cuda.memory_summary(abbreviated=False)
                )

                # Dump snapshot with allocation stack traces
                os.makedirs(memory_profile_dir, exist_ok=True)
                snapshot_path = os.path.join(
                    memory_profile_dir,
                    f"memory_snapshot_{dataset_name}_{ind}.pickle",
                )
                try:
                    torch.cuda.memory._dump_snapshot(snapshot_path)
                    logging.info(f"Memory snapshot saved to {snapshot_path}")
                except Exception as e:
                    logging.warning(f"Failed to dump memory snapshot: {e}")

                # Also write a text summary of the snapshot's allocation records
                try:
                    snapshot = torch.cuda.memory._snapshot()
                    if snapshot and "segments" in snapshot:
                        seg_lines = []
                        total_alloc = 0
                        total_reserved = 0
                        block_sizes = []
                        for seg in snapshot["segments"]:
                            seg_size = seg.get("total_size", 0)
                            total_reserved += seg_size
                            for block in seg.get("blocks", []):
                                bsize = block.get("size", 0)
                                state = block.get("state", "unknown")
                                if state == "active_allocated":
                                    total_alloc += bsize
                                    # Get allocation stack frames if available
                                    frames = block.get("frames", [])
                                    loc = ""
                                    for f in frames:
                                        fn = f.get("filename", "")
                                        if "co-tracker" in fn or "cotracker" in fn:
                                            loc = f"{fn}:{f.get('line', '?')}"
                                            break
                                    if not loc and frames:
                                        f = frames[0]
                                        loc = f"{f.get('filename', '?')}:{f.get('line', '?')}"
                                    block_sizes.append((bsize, loc))

                        block_sizes.sort(reverse=True)
                        seg_lines.append(
                            f"  Snapshot: {total_alloc / 1024**3:.2f} GB allocated, "
                            f"{total_reserved / 1024**3:.2f} GB reserved, "
                            f"{len(block_sizes)} active blocks"
                        )
                        seg_lines.append("  Largest allocated blocks:")
                        cumulative = 0
                        for bsize, loc in block_sizes[:50]:
                            cumulative += bsize
                            seg_lines.append(
                                f"    {bsize / 1024**2:>10.1f} MB  {loc}"
                            )
                        seg_lines.append(f"  Top-50 total: {cumulative / 1024**3:.2f} GB")
                        logging.info("\n".join(seg_lines))
                except Exception as e:
                    logging.warning(f"Failed to parse memory snapshot: {e}")

                torch.cuda.memory._record_memory_history(enabled=None)
                memory_profiled = True

            if "strided" in dataset_name:
                inv_video = sample.video.flip(1).clone()
                inv_queries = queries.clone()
                inv_queries[:, :, 0] = inv_video.shape[1] - inv_queries[:, :, 0] - 1

                pred_trj, pred_vsb = pred_tracks
                inv_pred_trj, inv_pred_vsb = model(inv_video, inv_queries)

                inv_pred_trj = inv_pred_trj.flip(1)
                inv_pred_vsb = inv_pred_vsb.flip(1)

                mask = pred_trj == 0

                pred_trj[mask] = inv_pred_trj[mask]
                pred_vsb[mask[:, :, :, 0]] = inv_pred_vsb[mask[:, :, :, 0]]

                pred_tracks = pred_trj, pred_vsb

            if dataset_name == "badja" or dataset_name == "fastcapture":
                seq_name = sample.seq_name[0]
            else:
                seq_name = str(ind)
            if ind % visualize_every == 0:
                if isinstance(pred_tracks, tuple):
                    tracks_to_vis, visibility_to_vis = pred_tracks
                    # Threshold visibility probabilities to binary (0.5 threshold)
                    visibility_to_vis = (visibility_to_vis > 0.5).float()
                else:
                    tracks_to_vis = pred_tracks
                    visibility_to_vis = None
                vis.visualize(
                    sample.video,
                    tracks_to_vis,
                    visibility=visibility_to_vis,
                    filename=dataset_name + "_" + seq_name,
                    writer=writer,
                    step=step,
                )
                if not train_mode and hasattr(sample, "trajectory") and sample.trajectory is not None:
                    vis.visualize(
                        sample.video,
                        sample.trajectory,
                        visibility=None,
                        filename=dataset_name + "_" + seq_name + "_gt",
                        writer=writer,
                        step=step,
                    )
            self.compute_metrics(metrics, sample, pred_tracks, dataset_name)
        return metrics
