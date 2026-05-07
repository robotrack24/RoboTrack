"""
PyTorch TAPNext predictor for evaluation.

Wraps the PyTorch TAPNext model to match the evaluator's interface:
    model(video, queries) -> (tracks, visibility)

The model runs frame-by-frame (online) and returns per-frame predictions.
"""

import torch
import torch.nn.functional as F


class TAPNextTorchPredictor:
    """
    Evaluation predictor for PyTorch TAPNext.

    Accepts the same (video, queries) interface as EvaluationPredictor
    so it can be plugged directly into the Evaluator.
    """

    def __init__(
        self,
        ckpt_path: str,
        image_size: tuple[int, int] = (256, 256),
        vis_threshold: float = 0.5,
    ):
        from tapnet.tapnext.tapnext_torch import TAPNext

        self.interp_shape = image_size  # (H, W)
        self.vis_threshold = vis_threshold

        self.model = TAPNext(image_size=image_size)

        if ckpt_path.endswith(".npz"):
            from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint
            restore_model_from_jax_checkpoint(self.model, ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.load_state_dict(state_dict)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def cuda(self):
        self.model = self.model.cuda()
        return self

    def to(self, device):
        self.model = self.model.to(device)
        return self

    @torch.inference_mode()
    def __call__(self, video: torch.Tensor, queries: torch.Tensor):
        """
        Args:
            video:   (B, T, C, H, W)  float, [0, 255]
            queries: (B, N, 3) as (t, x, y) in original pixel space

        Returns:
            tracks:     (B, T, N, 2) as (x, y) in original pixel space
            visibility: (B, T, N) float probabilities
        """
        B, T, C, H, W = video.shape
        device = video.device
        ih, iw = self.interp_shape

        # --- preprocess video: (B,T,C,H,W) -> (B,T,ih,iw,3) normalised to [-1,1] ---
        vid = video.reshape(B * T, C, H, W)
        vid = F.interpolate(vid, (ih, iw), mode="bilinear", align_corners=False)
        vid = vid.reshape(B, T, C, ih, iw)
        vid = vid.permute(0, 1, 3, 4, 2)  # (B,T,ih,iw,C)
        vid = vid / 127.5 - 1.0

        # --- preprocess queries: (t,x,y) in orig px -> (t,y,x) in model px ---
        q = queries.clone().float()
        q_t = q[..., 0:1]
        q_x = q[..., 1:2] * (iw / W)  # x in model space
        q_y = q[..., 2:3] * (ih / H)  # y in model space
        model_queries = torch.cat([q_t, q_y, q_x], dim=-1)  # (B, N, 3) as (t, y, x)

        # --- run model frame-by-frame ---
        all_tracks = []
        all_vis = []
        state = None
        for t_idx in range(T):
            frame = vid[:, t_idx : t_idx + 1]  # (B,1,ih,iw,C)
            if state is None:
                tracks_t, _, vis_logits_t, state = self.model(
                    frame, query_points=model_queries
                )
            else:
                tracks_t, _, vis_logits_t, state = self.model(frame, state=state)
            # tracks_t: (B,1,N,2) as (y,x) in model pixel space
            # vis_logits_t: (B,1,N,1) logits
            all_tracks.append(tracks_t)
            all_vis.append(vis_logits_t)

        tracks = torch.cat(all_tracks, dim=1)  # (B,T,N,2) as (y,x)
        vis_logits = torch.cat(all_vis, dim=1)  # (B,T,N,1)

        # --- postprocess: (y,x) model px -> (x,y) original px ---
        tracks_yx = tracks
        tracks_xy = tracks_yx[..., [1, 0]]  # flip to (x,y)
        tracks_xy[..., 0] *= W / iw  # x back to original scale
        tracks_xy[..., 1] *= H / ih  # y back to original scale

        visibility = torch.sigmoid(vis_logits.squeeze(-1))  # (B,T,N)

        return tracks_xy, visibility
