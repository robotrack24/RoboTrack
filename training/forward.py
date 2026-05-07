# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Forward batch computation for CoTracker training.
Extracted from train_on_kubric.py for use with new training infrastructure.
"""

from typing import Dict, Any
from dataclasses import dataclass

import torch
from omegaconf import DictConfig

from cotracker.models.core.cotracker.losses import (
    sequence_loss,
    sequence_BCE_loss,
    sequence_prob_loss,
)


@dataclass
class ForwardConfig:
    """Configuration for forward_batch function."""
    train_iters: int = 4
    offline_model: bool = True
    sliding_window_len: int = 16
    query_sampling_method: str = None
    train_only_on_visible: bool = False
    add_huber_loss: bool = True
    flow_loss_weight: float = 0.05
    invisible_loss_weight: float = 0.01


def forward_batch(
    batch,
    model: torch.nn.Module,
    cfg: ForwardConfig,
    queries=None,
) -> Dict[str, Any]:
    """
    Compute forward pass and losses for a batch.

    Args:
        batch: CoTrackerData batch containing video, trajectory, visibility, valid
        model: CoTracker model
        cfg: Forward configuration
        queries: Optional pre-computed query points, shape (B, N, 3) where each
            query is (t, x, y) — frame index and 2D coordinates. If None,
            queries are sampled according to cfg.query_sampling_method.

    Returns:
        Dictionary with loss components and predictions
    """
    video = batch.video
    trajs_g = batch.trajectory
    vis_g = batch.visibility
    valids = batch.valid

    B, T, C, H, W = video.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = video.device

    __, first_positive_inds = torch.max(vis_g, dim=1)

    # Query sampling
    if queries is None:
        if cfg.query_sampling_method == "random":
            queries = _sample_queries_random(trajs_g, vis_g, B, N, D, device)
        else:
            queries = _sample_queries_default(trajs_g, vis_g, first_positive_inds, B, T, N, D, device)
    
    assert queries.shape[:2] == (B, N), f"queries shape {queries.shape} != ({B}, {N}, 3)"

    # Handle invalid samples
    if (
        torch.isnan(queries).any()
        or torch.isnan(trajs_g).any()
        or queries.abs().max() > 1500
    ):
        print("failed_sample")
        print("queries time", queries[..., 0])
        print("queries ", queries[..., 1:])
        queries = torch.ones_like(queries).to(queries.device).float()
        print("new queries", queries)
        valids = torch.zeros_like(valids).to(valids.device).float()
        print("new valids", valids)

    # Forward pass
    model_output = model(
        video=video,
        queries=queries,
        iters=cfg.train_iters,
        is_train=True,
    )

    tracks, visibility, confidence, train_data = model_output
    coord_predictions, vis_predictions, confidence_predictions, valid_mask = train_data

    # Prepare ground truth for loss computation
    vis_gts = []
    invis_gts = []
    traj_gts = []
    valids_gts = []

    if cfg.offline_model:
        S = T
        seq_len = (S // 2) + 1
    else:
        S = cfg.sliding_window_len
        seq_len = T

    for ind in range(0, seq_len - S // 2, S // 2):
        vis_gts.append(vis_g[:, ind : ind + S])
        invis_gts.append(1 - vis_g[:, ind : ind + S])
        traj_gts.append(trajs_g[:, ind : ind + S, :, :2])
        val = valids[:, ind : ind + S]
        if not cfg.offline_model:
            val = val * valid_mask[:, ind : ind + S]
        valids_gts.append(val)

    # if len(vis_gts) > 0:
    #     # CoTracker-style predictions are nested: [chunk][iter] -> Tensor
    #     coord_outer = len(coord_predictions)
    #     coord_inner = len(coord_predictions[0]) if coord_outer > 0 else 0
    #     vis_outer = len(vis_predictions)
    #     vis_inner = len(vis_predictions[0]) if vis_outer > 0 else 0
    #     conf_outer = len(confidence_predictions)
    #     conf_inner = len(confidence_predictions[0]) if conf_outer > 0 else 0
    #     print(
    #         "gt shapes:",
    #         "vis_gts", tuple(torch.stack(vis_gts, dim=0).shape),
    #         "invis_gts", tuple(torch.stack(invis_gts, dim=0).shape),
    #         "traj_gts", tuple(torch.stack(traj_gts, dim=0).shape),
    #         "valids_gts", tuple(torch.stack(valids_gts, dim=0).shape),
    #         "coord_preds_outer_inner", (coord_outer, coord_inner),
    #         "coord_pred_sample",
    #         tuple(coord_predictions[0][0].shape) if coord_outer > 0 and coord_inner > 0 else None,
    #         "vis_preds_outer_inner", (vis_outer, vis_inner),
    #         "vis_pred_sample",
    #         tuple(vis_predictions[0][0].shape) if vis_outer > 0 and vis_inner > 0 else None,
    #         "conf_preds_outer_inner", (conf_outer, conf_inner),
    #         "conf_pred_sample",
    #         tuple(confidence_predictions[0][0].shape) if conf_outer > 0 and conf_inner > 0 else None,
    #         "valid_mask", tuple(valid_mask.shape),
    #     )
    #     print("coord_predictions", coord_predictions[0][0].max(), coord_predictions[0][0].min())
    #     print("vis_predictions", vis_predictions[0][0].max(), vis_predictions[0][0].min())
    #     print("confidence_predictions", confidence_predictions[0][0].max(), confidence_predictions[0][0].min())
    #     print("valid_mask", valid_mask[0].max(), valid_mask[0].min())
    # else:
    #     print("gt shapes: empty gt lists")

    # In Setup A (LightningLite), only model.forward() runs under autocast.
    # All loss computation runs outside autocast in plain fp32.
    # Match that by disabling autocast for the entire loss section.
    with torch.amp.autocast('cuda', enabled=False):
        seq_loss_visible = sequence_loss(
            coord_predictions,
            traj_gts,
            valids_gts,
            vis=vis_gts,
            gamma=0.8,
            add_huber_loss=cfg.add_huber_loss,
            loss_only_for_visible=True,
        )

        confidence_loss = sequence_prob_loss(
            coord_predictions, confidence_predictions, traj_gts, vis_gts
        )
        vis_loss = sequence_BCE_loss(vis_predictions, vis_gts)

        output = {"flow": {"predictions": tracks[0].detach()}}
        output["flow"]["loss"] = seq_loss_visible.mean() * 0.05
        output["flow"]["queries"] = queries.clone()

        output["visibility"] = {
            "loss": vis_loss.mean(),
            "predictions": visibility[0].detach(),
        }
        output["confidence"] = {
            "loss": confidence_loss.mean(),
        }

        if not cfg.train_only_on_visible:
            seq_loss_invisible = sequence_loss(
                coord_predictions,
                traj_gts,
                valids_gts,
                vis=invis_gts,
                gamma=0.8,
                add_huber_loss=False,
                loss_only_for_visible=True,
            )
            output["flow_invisible"] = {"loss": seq_loss_invisible.mean() * 0.01}

    return output


def _sample_queries_random(trajs_g, vis_g, B, N, D, device):
    """Sample queries from random visible frames (vectorized).

    For each (b, n), uniformly samples one frame index from the set of visible
    frames using the Gumbel-max trick: uniform noise masked to visible positions,
    then argmax selects a uniformly random visible frame.
    """
    T = vis_g.shape[1]
    vis_mask = vis_g.bool()
    rand = torch.rand(B, T, N, device=device)
    rand[~vis_mask] = -1.0
    chosen_t = rand.argmax(dim=1)  # (B, N)

    idx = chosen_t.unsqueeze(1).unsqueeze(-1).expand(B, 1, N, D)
    sampled_points = torch.gather(trajs_g, 1, idx).squeeze(1)  # (B, N, D)

    queries = torch.cat([chosen_t.unsqueeze(-1).float(), sampled_points], dim=2)
    return queries


def _sample_queries_default(trajs_g, vis_g, first_positive_inds, B, T, N, D, device):
    """Sample queries mixing random visibility and first frame (vectorized).

    For the first N//4 points, replaces first_positive_inds with a randomly
    sampled visible frame. Remaining points keep their first visible frame.

    Handles padded points (visibility all zeros) by assigning them to frame 0
    with zero coordinates. These are masked out by valid=0 in loss computation.
    """
    N_rand = N // 4

    vis_mask = vis_g.bool()
    rand = torch.rand(B, T, N, device=device)
    has_vis = vis_mask.any(dim=1)  # (B, N)
    rand[~vis_mask] = -1.0
    rand_vis_inds = rand.argmax(dim=1)  # (B, N)
    rand_vis_inds[~has_vis] = 0

    first_positive_inds[:, :N_rand] = rand_vis_inds[:, :N_rand]

    # Verify: sampled query frames have visibility=1 (skip padded points)
    sampled_vis = torch.gather(vis_g, 1, first_positive_inds.unsqueeze(1)).squeeze(1)  # (B, N)
    assert torch.allclose(
        sampled_vis[has_vis],
        torch.ones(1, device=device),
    ), "Query sampled at frame with visibility=0 for a non-padded point"

    idx = first_positive_inds.unsqueeze(1).unsqueeze(-1).expand(B, 1, N, D)
    xys = torch.gather(trajs_g, 1, idx).squeeze(1)  # (B, N, D)

    queries = torch.cat([first_positive_inds[:, :, None].float(), xys[:, :, :2]], dim=2)
    return queries


def compute_total_loss(output: Dict[str, Any]) -> torch.Tensor:
    """Compute total loss from output dictionary."""
    loss = torch.tensor(0.0, device=next(iter(output.values()))["loss"].device)
    for k, v in output.items():
        if "loss" in v:
            loss = loss + v["loss"]
    return loss


def create_forward_config(cfg: DictConfig) -> ForwardConfig:
    """Create ForwardConfig from Hydra config."""
    return ForwardConfig(
        train_iters=cfg.training.train_iters,
        offline_model=cfg.training.offline_model,
        sliding_window_len=cfg.training.sliding_window_len,
        query_sampling_method=cfg.training.query_sampling_method,
        train_only_on_visible=cfg.training.train_only_on_visible,
        add_huber_loss=cfg.training.add_huber_loss,
        flow_loss_weight=cfg.training.get("flow_loss_weight", 0.05),
        invisible_loss_weight=cfg.training.get("invisible_loss_weight", 0.01),
    )
