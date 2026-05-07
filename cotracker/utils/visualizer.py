# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import imageio
import torch

from matplotlib import cm
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def read_video_from_path(path):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)


def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None):
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])

    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb


def draw_line(rgb, coord_y, coord_x, color, linewidth):
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb


def add_weighted(rgb, alpha, original, beta, gamma):
    return (rgb * alpha + original * beta + gamma).astype("uint8")


class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: float = 2,
        show_first_frame: int = 10,
        tracks_leave_trace: int = 0,  # -1 for infinite
        max_supersample: int = 8,
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = float(linewidth)
        self.fps = fps

        # PIL drawing primitives only accept integer pixel widths/radii. To
        # support sub-pixel ``linewidth`` we render at a supersampled
        # resolution where the requested width maps to >=1 integer pixel,
        # then downsample with Lanczos to obtain an antialiased thin stroke.
        if self.linewidth < 1:
            self._supersample = min(
                max_supersample, max(1, int(np.ceil(1.0 / self.linewidth)))
            )
        else:
            self._supersample = 1
        self._effective_linewidth = max(
            1, int(round(self.linewidth * self._supersample))
        )

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,T,N,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        filename: str = "video",
        writer=None,  # tensorboard Summary Writer, used for visualization during training
        step: int = 0,
        query_frame=0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
        opacity: float = 1.0,
    ):
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][coords[:, 1], coords[:, 0]].long()

        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )
        color_alpha = int(opacity * 255)
        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        res_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            color_alpha=color_alpha,
        )
        if save_video:
            self.save_video(res_video, filename=filename, writer=writer, step=step)
        return res_video

    def save_video(self, video, filename, writer=None, step=0):
        if writer is not None:
            writer.add_video(
                filename,
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]

            # Prepare the video file path
            save_path = os.path.join(self.save_dir, f"{filename}.mp4")

            # Create a writer object (macro_block_size=1 prevents resizing warnings)
            video_writer = imageio.get_writer(save_path, fps=self.fps, macro_block_size=1)

            # Write frames to the video file
            for frame in wide_list[2:-1]:
                video_writer.append_data(frame)

            video_writer.close()

            print(f"Video saved to {save_path}")

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame=0,
        compensate_for_camera_motion=False,
        color_alpha: int = 255,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].round().long().detach().cpu().numpy()  # S, N, 2
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        if N == 0:
            res_video = [rgb.copy() for rgb in video]
            if self.show_first_frame > 0:
                res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
            return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

        # Upscale the canvas + track coords for sub-pixel linewidth support.
        # All PIL draw calls below operate in this upscaled space; we
        # downsample with Lanczos at the very end for antialiasing.
        ss = self._supersample
        if ss > 1:
            H_ss, W_ss = H * ss, W * ss
            video = np.stack(
                [
                    np.array(
                        Image.fromarray(frame).resize((W_ss, H_ss), Image.NEAREST)
                    )
                    for frame in video
                ]
            )
            tracks = tracks * ss
            if gt_tracks is not None:
                gt_tracks = gt_tracks * ss

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())
        vector_colors = np.zeros((T, N, 3))

        if self.mode == "optical_flow":
            import flow_vis

            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])
        elif segm_mask is None:
            if self.mode == "rainbow":
                if isinstance(query_frame, (torch.Tensor, np.ndarray)):
                    qf_np = query_frame.cpu().numpy().ravel() if isinstance(query_frame, torch.Tensor) else np.asarray(query_frame).ravel()
                    if qf_np.size < N:
                        qf_np = np.pad(qf_np, (0, N - qf_np.size), mode="edge")
                    elif qf_np.size > N:
                        qf_np = qf_np[:N]
                    qf_np = np.clip(qf_np, 0, T - 1)
                    y_vals = np.array(
                        [tracks[int(qf_np[n]), n, 1] for n in range(N)],
                        dtype=np.float64,
                    )
                else:
                    qf = int(np.clip(query_frame, 0, T - 1))
                    y_vals = tracks[qf, :, 1]
                y_min, y_max = float(y_vals.min()), float(y_vals.max())
                if y_max <= y_min:
                    y_max = y_min + 1.0
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if isinstance(query_frame, (torch.Tensor, np.ndarray)):
                        query_frame_ = int(qf_np[n])
                    else:
                        query_frame_ = qf
                    color = self.color_map(norm(tracks[query_frame_, n, 1]))
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # color changes with time
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255

                y_min, y_max = (
                    tracks[0, segm_mask > 0, 1].min(),
                    tracks[0, segm_mask > 0, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if segm_mask[n] > 0:
                        color = self.color_map(norm(tracks[0, n, 1]))
                        color = np.array(color[:3])[None] * 255
                        vector_colors[:, n] = np.repeat(color, T, axis=0)

            else:
                # color changes with segm class
                segm_mask = segm_mask.cpu()
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                vector_colors = np.repeat(color[None], T, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            trace_start = int(query_frame.min()) + 1 if isinstance(query_frame, (torch.Tensor, np.ndarray)) else query_frame + 1
            for t in range(trace_start, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (
                        tracks[first_ind : t + 1, segm_mask <= 0]
                        - tracks[t : t + 1, segm_mask <= 0]
                    ).mean(1)[:, None]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                )
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(
                        res_video[t], gt_tracks[first_ind : t + 1]
                    )

        #  draw points
        for t in range(T):
            img = Image.fromarray(np.uint8(res_video[t]))
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]
                has_coord = visibility is None or (coord[0] != 0 and coord[1] != 0)
                if has_coord:
                    if not compensate_for_camera_motion or (
                        compensate_for_camera_motion and segm_mask[i] > 0
                    ):
                        img = draw_circle(
                            img,
                            coord=coord,
                            radius=int(self._effective_linewidth * 2),
                            color=vector_colors[t, i].astype(int),
                            visible=visibile,
                            color_alpha=color_alpha,
                        )
            res_video[t] = np.array(img)

        # Downsample back to the original resolution. Lanczos gives a clean
        # antialiased thin stroke when ``linewidth < 1``.
        if ss > 1:
            res_video = [
                np.array(Image.fromarray(frame).resize((W, H), Image.LANCZOS))
                for frame in res_video
            ]

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.5,
    ):
        T, N, _ = tracks.shape
        rgb = Image.fromarray(np.uint8(rgb))
        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].astype(int),
                        self._effective_linewidth,
                    )
            if self.tracks_leave_trace > 0:
                rgb = Image.fromarray(
                    np.uint8(
                        add_weighted(
                            np.array(rgb), alpha, np.array(original), 1 - alpha, 0
                        )
                    )
                )
        rgb = np.array(rgb)
        return rgb

    def _draw_gt_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3,
        gt_tracks: np.ndarray,  # T x N x 2
    ):
        T, N, _ = gt_tracks.shape
        color = np.array((211, 0, 0))
        rgb = Image.fromarray(np.uint8(rgb))
        for t in range(T):
            for i in range(N):
                pt = gt_tracks[t][i]
                if pt[0] > 0 and pt[1] > 0:
                    length = self._effective_linewidth * 3
                    coord_y = (int(pt[0]) + length, int(pt[1]) + length)
                    coord_x = (int(pt[0]) - length, int(pt[1]) - length)
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self._effective_linewidth,
                    )
                    coord_y = (int(pt[0]) - length, int(pt[1]) + length)
                    coord_x = (int(pt[0]) + length, int(pt[1]) - length)
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self._effective_linewidth,
                    )
        rgb = np.array(rgb)
        return rgb
