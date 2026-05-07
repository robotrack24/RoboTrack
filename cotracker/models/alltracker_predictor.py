"""
AllTracker evaluation predictor adapter.

Wraps the AllTracker model (aharley/alltracker) to match the interface expected
by our DROID evaluation pipeline: predictor(video, queries) -> (tracks, visibility).
"""

import sys
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ALLTRACKER_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "alltracker",
)


def build_alltracker(
    checkpoint: str = "./checkpoints/alltracker.pth",
    window_len: int = 16,
    device: str = "cuda",
) -> nn.Module:
    """Load the AllTracker model from checkpoint."""
    if ALLTRACKER_ROOT not in sys.path:
        sys.path.insert(0, ALLTRACKER_ROOT)

    from nets.alltracker import Net

    model = Net(window_len)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    return model


class AllTrackerPredictor(nn.Module):
    """Adapter that makes AllTracker compatible with the CoTracker eval interface.

    Expected call signature:
        pred_tracks, pred_visibility = predictor(video, queries)

    Args:
        video: (B, T, 3, H, W) float tensor, pixel values in [0, 255]
        queries: (B, N, 3) float tensor, each query is [t, x, y]

    Returns:
        pred_tracks: (B, T, N, 2) predicted xy coordinates in original resolution
        pred_visibility: (B, T, N) visibility scores in [0, 1]
    """

    def __init__(
        self,
        model: nn.Module,
        interp_shape: Tuple[int, int] = (384, 512),
        inference_iters: int = 4,
        vis_threshold: float = 0.6,
    ):
        super().__init__()
        self.model = model
        self.interp_shape = interp_shape
        self.inference_iters = inference_iters
        self.vis_threshold = vis_threshold

    @torch.no_grad()
    def forward(self, video: torch.Tensor, queries: torch.Tensor):
        B, T, C, H, W = video.shape
        assert B == 1, "AllTrackerPredictor only supports batch_size=1"
        N = queries.shape[1]
        device = video.device

        # Resize video to interp_shape for AllTracker inference
        ih, iw = self.interp_shape
        video_resized = video.reshape(B * T, C, H, W)
        video_resized = F.interpolate(
            video_resized, (ih, iw), mode="bilinear", align_corners=True
        )
        video_resized = video_resized.reshape(B, T, C, ih, iw)

        # Scale query coordinates to interp_shape
        scale_x = (iw - 1) / max(W - 1, 1)
        scale_y = (ih - 1) / max(H - 1, 1)
        queries_scaled = queries.clone()
        queries_scaled[:, :, 1] *= scale_x  # x
        queries_scaled[:, :, 2] *= scale_y  # y

        # Build output tensors
        pred_tracks = torch.zeros((B, T, N, 2), device=device)
        pred_visconf = torch.zeros((B, T, N, 2), device=device)

        # Grid for converting flow to absolute coords
        grid_xy = torch.stack(
            torch.meshgrid(
                torch.arange(iw, device=device, dtype=torch.float32),
                torch.arange(ih, device=device, dtype=torch.float32),
                indexing="xy",
            ),
            dim=0,
        )  # (2, ih, iw)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(0)  # (1, 1, 2, ih, iw)

        # Group queries by their query frame
        query_frames = queries_scaled[0, :, 0].long()  # (N,)
        unique_frames = torch.unique(query_frames)

        for qf in unique_frames:
            qf_int = qf.item()
            pt_idxs = torch.nonzero(query_frames == qf, as_tuple=False)[:, 0]

            # Get query point positions at the query frame
            qx = queries_scaled[0, pt_idxs, 1].round().long().clamp(0, iw - 1)
            qy = queries_scaled[0, pt_idxs, 2].round().long().clamp(0, ih - 1)

            # At the query frame, tracks are at the query position
            traj_maps = grid_xy.repeat(1, T, 1, 1, 1)  # (1, T, 2, ih, iw)
            visconf_maps = torch.zeros((1, T, 2, ih, iw), device=device)

            if qf_int < T - 1:
                video_from_qf = video_resized[:, qf_int:]
                T_sub = video_from_qf.shape[1]

                if T_sub > 128:
                    flow_e, visconf_e, _, _ = self.model.forward_sliding(
                        video_from_qf, iters=self.inference_iters, sw=None, is_training=False
                    )
                else:
                    flow_e, visconf_e, _, _ = self.model(
                        video_from_qf, iters=self.inference_iters, sw=None, is_training=False
                    )

                # flow_e: (B, T_sub, 2, ih, iw) — displacement from query frame
                forward_traj = flow_e.to(device) + grid_xy  # absolute coords
                traj_maps[:, qf_int:] = forward_traj
                visconf_maps[:, qf_int:] = visconf_e.to(device)

            # Sample tracks at query point locations
            # traj_maps is (1, T, 2, ih, iw), index with [qy, qx]
            tracks_chunk = traj_maps[:, :, :, qy, qx]  # (1, T, 2, K)
            tracks_chunk = tracks_chunk.permute(0, 1, 3, 2)  # (1, T, K, 2)

            visconf_chunk = visconf_maps[:, :, :, qy, qx]  # (1, T, 2, K)
            visconf_chunk = visconf_chunk.permute(0, 1, 3, 2)  # (1, T, K, 2)

            pred_tracks[:, :, pt_idxs] = tracks_chunk
            pred_visconf[:, :, pt_idxs] = visconf_chunk

        # Combine vis and conf: visibility * confidence
        pred_visibility = pred_visconf[:, :, :, 0] * pred_visconf[:, :, :, 1]
        pred_visibility = pred_visibility.clamp(0, 1)

        # Scale tracks back to original resolution
        pred_tracks[:, :, :, 0] *= (W - 1) / max(iw - 1, 1)
        pred_tracks[:, :, :, 1] *= (H - 1) / max(ih - 1, 1)

        return pred_tracks, pred_visibility
