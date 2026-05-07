# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Callable
import collections
from torch import Tensor
from itertools import repeat
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoVideoProcessor
from cotracker.models.core.model_utils import bilinear_sampler

try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False




# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, padding_mode="zeros"
        )
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

class VJEPA2Encoder(nn.Module):
    """
    Wrapper for V-JEPA 2 to produce per-frame features for CoTracker3.

    V-JEPA 2 processes video as spatiotemporal tubelets (patch_size x patch_size x tubelet_size).
    For the ViT-g/16_384 checkpoint: patch_size=16, tubelet_size=2, hidden_size=1408.

    Since tubelet_size=2 compresses every 2 frames into 1 temporal token, we use trilinear
    interpolation to map T/2 temporal tokens back to T per-frame feature maps.
    """

    def __init__(
        self,
        output_dim: int = 128,
        stride: int = 4,
        model_name: str = "facebook/vjepa2-vitg-fpc64-384",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.stride = stride
        self.freeze = freeze

        if pretrained:
            self.model = AutoModel.from_pretrained(
                model_name,
                attn_implementation="sdpa",
                dtype=torch.float32,
            )
        else:
            from transformers import VJEPA2Config
            self.model = AutoModel.from_config(VJEPA2Config()).to(torch.bfloat16)

        self.processor = AutoVideoProcessor.from_pretrained(model_name)

        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size
        self.tubelet_size = self.model.config.tubelet_size
        self.crop_size = self.model.config.crop_size

        self.proj = nn.Conv2d(self.hidden_size, output_dim, kernel_size=1)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for n, p in self.model.named_parameters():
                if "encoder.embeddings" in n:
                    p.requires_grad = False
                elif "encoder.layer." in n:
                    layer_num = int(n.split(".")[2])
                    if layer_num < 12:
                        p.requires_grad = False
                    else: p.requires_grad = True
                else:
                    p.requires_grad = True

    @property
    def supports_video_input(self) -> bool:
        return True

    def preprocess_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess using the official VJEPA2VideoProcessor."""
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, S, C, H, W = x.shape

        if x.max() > 1.0:
            x = x / 255.0

        # Processor expects list of (T, C, H, W) tensors
        videos = [x[i] for i in range(B)]
        processed = self.processor(videos, return_tensors="pt", do_rescale=False)
        pixel_values = processed["pixel_values_videos"].to(x.device, x.dtype)

        return pixel_values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_is_4d = x.dim() == 4
        if input_is_4d:
            B, C, H, W = x.shape
            S = 1
        else:
            B, S, C, H, W = x.shape

        preprocessed = self.preprocess_tensor(x)
        S_padded = preprocessed.shape[1]

        with torch.set_grad_enabled(not self.freeze):
            features = self.model.get_vision_features(preprocessed.to(torch.bfloat16))

        T_tokens = S_padded // self.tubelet_size
        H_pre, W_pre = preprocessed.shape[3], preprocessed.shape[4]
        H_patches = H_pre // self.patch_size
        W_patches = W_pre // self.patch_size
        # Token sequence -> spatiotemporal grid -> per-frame via interpolation
        features = features.reshape(B, T_tokens, H_patches, W_patches, self.hidden_size)
        features = features.permute(0, 4, 1, 2, 3).float()  # (B, D, T_tok, H_p, W_p)

        if T_tokens != S:
            features = F.interpolate(
                features, size=(S, H_patches, W_patches),
                mode="trilinear", align_corners=False,
            )

        features = features.reshape(B * S, self.hidden_size, H_patches, W_patches)
        features = self.proj(features)

        target_h = H // self.stride
        target_w = W // self.stride
        if features.shape[-2:] != (target_h, target_w):
            features = F.interpolate(
                features, size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            )

        if input_is_4d:
            return features
        return features.view(B, S, self.output_dim, target_h, target_w)


class VGGTEncoder(nn.Module):
    """
    VGGT Aggregator + DPT feature head for CoTracker3 (no ResNet side branch).

    Loads the pretrained ``Aggregator`` and, when ``dpt_features=128`` and
    ``feature_down_ratio=2``, the DPT weights from VGGT's ``track_head.feature_extractor``.
    The encoder returns the native dense feature map produced by FeatureExtractor,
    without an extra resize step. Downstream tracking code should map image
    coordinates into feature coordinates using ``feature_down_ratio``.
    """

    _AGG_EMBED_DIM = 1024
    _AGG_PATCH_EMBED = "dinov2_vitl14_reg"
    _AGG_DEPTH = 24

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = True,
        img_size: int = 518,
        patch_size: int = 14,
        dpt_features: int = 128,
        feature_down_ratio: int = 2,
    ):
        super().__init__()
        self.freeze = freeze
        self.img_size = img_size
        self.patch_size = patch_size
        self.feature_down_ratio = feature_down_ratio

        import cowtracker.thirdparty  # noqa: F401
        from vggt.models.aggregator import Aggregator
        from cowtracker.heads.feature_extractor import FeatureExtractor

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=self._AGG_EMBED_DIM,
            patch_embed=self._AGG_PATCH_EMBED,
            depth=self._AGG_DEPTH,
        )

        vggt_full = None
        if pretrained:
            from vggt.models.vggt import VGGT
            vggt_full = VGGT.from_pretrained("facebook/VGGT-1B")
            self.aggregator.load_state_dict(vggt_full.aggregator.state_dict())

        self.feature_extractor = FeatureExtractor(
            features=dpt_features,
            down_ratio=feature_down_ratio,
            use_side_resnet=False,
        )
        self.output_dim = self.feature_extractor.out_dim

        if vggt_full is not None:
            if dpt_features == 128 and feature_down_ratio == 2:
                prefix = "track_head.feature_extractor."
                dpt_sd = {
                    k[len(prefix):]: v
                    for k, v in vggt_full.state_dict().items()
                    if k.startswith(prefix)
                }
                if dpt_sd:
                    self.feature_extractor.dpt_head.load_state_dict(dpt_sd, strict=True)
                    print("Loaded DPT weights from VGGT checkpoint")
                else:
                    import warnings
                    warnings.warn(
                        "VGGT checkpoint has no track_head.feature_extractor.* weights; "
                        "DPT head stays randomly initialized.",
                        stacklevel=2,
                    )
            del vggt_full

        self.aggregator = self.aggregator.to(torch.bfloat16)

        if freeze:
            for n, p in self.aggregator.named_parameters():
                if "patch_embed." in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

    @property
    def supports_video_input(self) -> bool:
        return True

    def preprocess_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)

        B, S, C, H, W = x.shape

        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = x.view(B * S, C, H, W)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            x = x.view(B, S, C, H + pad_h, W + pad_w)

        x_min = x.min()
        x_max = x.max()
        if x_max > 1.0:
            x = x / 255.0
        elif x_min < 0.0:
            x = (x + 1.0) / 2.0

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_is_4d = x.dim() == 4
        if input_is_4d:
            B, C, H, W = x.shape
            S = 1
        else:
            B, S, C, H, W = x.shape

        preprocessed = self.preprocess_tensor(x)

        if self.training and preprocessed.requires_grad:
            def _aggregator_tokens(inp: torch.Tensor):
                token_list, _ = self.aggregator(inp)
                return tuple(token_list)

            aggregated_tokens_list = list(
                checkpoint(_aggregator_tokens, preprocessed, use_reentrant=False)
            )
            ps_idx = self.aggregator.patch_start_idx
        else:
            aggregated_tokens_list, ps_idx = self.aggregator(preprocessed)

        preprocessed_fe = preprocessed.float()
        features = self.feature_extractor(
            aggregated_tokens_list, preprocessed_fe, ps_idx
        )

        # features is already at the intended spatial resolution:
        # [B, S, C_out, Hf, Wf], where Hf/Wf are determined by feature_down_ratio
        if input_is_4d:
            return features.squeeze(1).contiguous()

        return features.contiguous()

class PerceptionEncoder(nn.Module):
    """
    Wrapper for Perception Encoder to produce image features for Cotracker3
    """

    def __init__(
        self,
        output_dim=128,
        stride: int = 4,
        model_name: str = "PE-Spatial-G14-448",
        pretrained: bool = True,
        freeze: bool = True
    ):
        super(PerceptionEncoder, self).__init__()
        self.output_dim = output_dim
        self.model_name = model_name
        self.freeze = freeze
        
        import perception_models.core.vision_encoder.pe as pe
        from perception_models.core.vision_encoder.config import PE_VISION_CONFIG

        if self.model_name not in PE_VISION_CONFIG:
            available = list(PE_VISION_CONFIG.keys())
            raise ValueError(f"Model name {self.model_name} not found in available models: {available}")

        self.model = pe.VisionTransformer.from_config(self.model_name, pretrained=pretrained)
        self.image_size = self.model.image_size
        
        # Keep the requested stride (not patch_size) for output compatibility with CoTracker
        self.stride = stride
        
        self.proj = nn.Conv2d(self.model.output_dim, output_dim, kernel_size=1)
        
        # Convert to bfloat16 for mixed precision training
        self.model = self.model.to(torch.bfloat16)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def preprocess_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess tensor input for the vision transformer.
        Expects input in range [0, 255] or [0, 1], outputs normalized tensor.
        
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Preprocessed tensor of shape (B, C, image_size, image_size)
        """
        # Resize to model's expected size
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # Normalize: if input is in [0, 255], convert to [0, 1] first
        if x.max() > 1.0:
            x = x / 255.0
        
        # Normalize to [-1, 1] (same as transforms: mean=0.5, std=0.5)
        x = (x - 0.5) / 0.5
        
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        with torch.set_grad_enabled(not self.freeze):
            preprocessed = self.preprocess_tensor(x)
            features = self.model.forward_features(preprocessed, strip_cls_token=True)
        
        # Reshape from (B, num_patches, dim) to (B, h_patches, w_patches, dim)
        # Use preprocessed image_size for patch calculation (not original H, W)
        h_patches = self.image_size // self.model.patch_size
        w_patches = self.image_size // self.model.patch_size
        features = features.reshape(B, h_patches, w_patches, -1)
        features = features.permute(0, 3, 1, 2)  # (B, dim, h_patches, w_patches)

    
        features = self.proj(features)  # (B, 128, h_patches, w_patches)

        # Resize features to match expected output size (H/stride, W/stride)
        target_h = H // self.stride
        target_w = W // self.stride
        if features.shape[-2:] != (target_h, target_w):
            features = F.interpolate(features, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return features


class DINOv3Encoder(nn.Module):
    """
    Wraps a pretrained DINOv3 ViT backbone (HuggingFace transformers) for CoTracker3.

    DINOv3 ViT-B/16 produces patch tokens at stride 16 with hidden_dim=768.
    We project 768 -> output_dim with a 1x1 conv and bilinearly upsample
    from stride 16 to stride 4, matching BasicEncoder's output contract.

    This encoder handles its own ImageNet normalization internally, so
    the model config should set normalize_input=false.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        output_dim: int = 128,
        stride: int = 4,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        pretrained: bool = True,
        freeze: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.stride = stride

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                model_name, attn_implementation="sdpa",
            )
        else:
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
            self.backbone = DINOv3ViTModel(DINOv3ViTConfig())

        self.patch_size = self.backbone.config.patch_size
        hidden_size = self.backbone.config.hidden_size
        self.proj = nn.Conv2d(hidden_size, output_dim, kernel_size=1)

        self.register_buffer(
            "pixel_mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        if freeze:
            # Freeze patch embeddings + position embeddings (well-learned, fragile)
            # but keep transformer blocks trainable for task adaptation.
            # Mirrors the VGGT encoder's freeze strategy.
            frozen_prefixes = ("embeddings.",)
            for n, p in self.backbone.named_parameters():
                if any(n.startswith(pfx) for pfx in frozen_prefixes):
                    p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Normalize: [0, 255] -> [0, 1] -> ImageNet normalize
        x = x / 255.0 if x.max() > 1.0 else x
        x = (x - self.pixel_mean) / self.pixel_std

        outputs = self.backbone(pixel_values=x)
        tokens = outputs.last_hidden_state  # (B, num_patches, hidden_size)

        h_patches = H // self.patch_size
        w_patches = W // self.patch_size
        features = tokens[:, :h_patches * w_patches]  # strip any CLS / register tokens
        features = features.reshape(B, h_patches, w_patches, -1)
        features = features.permute(0, 3, 1, 2)  # (B, hidden_size, h_patches, w_patches)

        features = self.proj(features)  # (B, output_dim, h_patches, w_patches)

        target_h = H // self.stride
        target_w = W // self.stride
        if features.shape[-2:] != (target_h, target_w):
            features = F.interpolate(
                features, size=(target_h, target_w),
                mode="bilinear", align_corners=False,
            )

        return features


class BasicEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=128, stride=4):
        super().__init__()
        self.stride = stride
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2
        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)

        self.conv2 = nn.Conv2d(
            output_dim * 3 + output_dim // 4,
            output_dim * 2,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        d = self.layer4(c)

        def _bilinear_intepolate(x):
            return F.interpolate(
                x,
                (H // self.stride, W // self.stride),
                mode="bilinear",
                align_corners=True,
            )

        a = _bilinear_intepolate(a)
        b = _bilinear_intepolate(b)
        c = _bilinear_intepolate(c)
        d = _bilinear_intepolate(d)

        x = self.conv2(torch.cat([a, b, c, d], dim=1))
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class EfficientCorrBlock:
    def __init__(
        self,
        fmaps,
        num_levels=4,
        radius=4,
        padding_mode="zeros",
    ):
        B, S, C, H, W = fmaps.shape
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords, target):
        r = self.radius
        device = coords.device
        B, S, N, D = coords.shape
        assert D == 2

        target = target.permute(0, 1, 3, 2).unsqueeze(-1)

        out_pyramid = []
        for i in range(self.num_levels):
            pyramid = self.fmaps_pyramid[i]
            C, H, W = pyramid.shape[2:]
            centroid_lvl = (
                torch.cat(
                    [torch.zeros_like(coords[..., :1], device=device), coords], dim=-1
                ).reshape(B * S, N, 1, 1, 3)
                / 2**i
            )

            dx = torch.linspace(-r, r, 2 * r + 1, device=device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=device)

            xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
            zgrid = torch.zeros_like(xgrid, device=device)
            delta = torch.stack([zgrid, xgrid, ygrid], axis=-1)
            delta_lvl = delta.view(1, 1, 2 * r + 1, 2 * r + 1, 3)
            coords_lvl = centroid_lvl + delta_lvl
            pyramid_sample = bilinear_sampler(
                pyramid.reshape(B * S, C, 1, H, W), coords_lvl
            )

            corr = torch.sum(target * pyramid_sample.reshape(B, S, C, N, -1), dim=2)
            corr = corr / torch.sqrt(torch.tensor(C).float())
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        out = out.permute(0, 2, 1, 3).contiguous().view(B * N, S, -1).float()
        return out


class CorrBlock:
    def __init__(
        self,
        fmaps,
        num_levels=4,
        radius=4,
        multiple_track_feats=False,
        padding_mode="zeros",
    ):
        B, S, C, H, W = fmaps.shape
        self.S, self.C, self.H, self.W = S, C, H, W
        self.padding_mode = padding_mode
        self.num_levels = num_levels
        self.radius = radius
        self.fmaps_pyramid = []
        self.multiple_track_feats = multiple_track_feats

        self.fmaps_pyramid.append(fmaps)
        for i in range(self.num_levels - 1):
            fmaps_ = fmaps.reshape(B * S, C, H, W)
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            _, _, H, W = fmaps_.shape
            fmaps = fmaps_.reshape(B, S, C, H, W)
            self.fmaps_pyramid.append(fmaps)

    def sample(self, coords):
        r = self.radius
        B, S, N, D = coords.shape
        assert D == 2

        H, W = self.H, self.W
        out_pyramid = []
        for i in range(self.num_levels):
            corrs = self.corrs_pyramid[i]  # B, S, N, H, W
            *_, H, W = corrs.shape

            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).to(
                coords.device
            )

            centroid_lvl = coords.reshape(B * S * N, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corrs = bilinear_sampler(
                corrs.reshape(B * S * N, 1, H, W),
                coords_lvl,
                padding_mode=self.padding_mode,
            )
            corrs = corrs.view(B, S, N, -1)
            out_pyramid.append(corrs)

        out = torch.cat(out_pyramid, dim=-1)  # B, S, N, LRR*2
        out = out.permute(0, 2, 1, 3).contiguous().view(B * N, S, -1).float()
        return out

    def corr(self, targets):
        B, S, N, C = targets.shape
        if self.multiple_track_feats:
            targets_split = targets.split(C // self.num_levels, dim=-1)
            B, S, N, C = targets_split[0].shape

        assert C == self.C
        assert S == self.S

        fmap1 = targets

        self.corrs_pyramid = []
        for i, fmaps in enumerate(self.fmaps_pyramid):
            *_, H, W = fmaps.shape
            fmap2s = fmaps.view(B, S, C, H * W)  # B S C H W ->  B S C (H W)
            if self.multiple_track_feats:
                fmap1 = targets_split[i]
            corrs = torch.matmul(fmap1, fmap2s)
            corrs = corrs.view(B, S, N, H, W)  # B S N (H W) -> B S N H W
            corrs = corrs / torch.sqrt(torch.tensor(C).float())
            self.corrs_pyramid.append(corrs)


def _rotate_half(x):
    """Rotates half the hidden dims of the input for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to Q and K tensors.

    Args:
        q, k: (B, H, T, D) tensors after projection
        cos, sin: (1, 1, T, D) precomputed rotation components
    """
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    """Precomputes and caches rotary position embedding sin/cos tables."""

    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len):
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len],
            self.sin_cached[:, :, :seq_len],
        )


class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_bias=None, rope=None):
        B, N1, C = x.shape
        h = self.heads

        q = self.to_q(x).reshape(B, N1, h, C // h).permute(0, 2, 1, 3)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        N2 = context.shape[1]
        k = k.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)
        v = v.reshape(B, N2, h, C // h).permute(0, 2, 1, 3)

        if rope is not None:
            cos, sin = rope
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        sim = (q @ k.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            sim = sim + attn_bias
        attn = sim.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        return self.to_out(x)

class AttentionTorch(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, num_heads=8, dim_head=48, qkv_bias=False
    ):
        super().__init__()
        inner_dim = dim_head * num_heads
        context_dim = default(context_dim, query_dim)

        self.heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = inner_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=qkv_bias)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, attn_bias=None, rope=None):
        """
        x:        (B, N1, query_dim)
        context:  (B, N2, context_dim) or None (self-attn)
        attn_bias: additive bias or mask broadcastable to (B, heads, N1, N2)
                  - float: added to attention logits (e.g. 0 or -inf)
                  - bool:  True = keep, False = mask out (per PyTorch SDPA semantics)
        rope:     optional (cos, sin) tuple from RotaryEmbedding, each (1, 1, T, d)
        """
        B, N1, _ = x.shape
        h, d = self.heads, self.dim_head

        context = default(context, x)
        N2 = context.shape[1]

        q = self.to_q(x)                      # (B, N1, h*d)
        k, v = self.to_kv(context).chunk(2, dim=-1)  # each (B, N2, h*d)

        q = q.view(B, N1, h, d).transpose(1, 2)  # (B, h, N1, d)
        k = k.view(B, N2, h, d).transpose(1, 2)  # (B, h, N2, d)
        v = v.view(B, N2, h, d).transpose(1, 2)  # (B, h, N2, d)

        if rope is not None:
            cos, sin = rope
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, N1, h * d)

        return self.to_out(out)

class AttnBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        attn_class: Callable[..., nn.Module] = AttentionTorch,
        mlp_ratio=4.0,
        use_rope=False,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_class(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )

        self.rope_emb = None
        if use_rope:
            dim_head = getattr(self.attn, 'dim_head', hidden_size // num_heads)
            self.rope_emb = RotaryEmbedding(dim_head)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, mask=None):
        attn_bias = mask
        if mask is not None:
            mask = (
                (mask[:, None] * mask[:, :, None])
                .unsqueeze(1)
                .expand(-1, self.attn.heads, -1, -1)
            )
            max_neg_value = -torch.finfo(x.dtype).max
            attn_bias = (~mask) * max_neg_value

        rope = None
        if self.rope_emb is not None:
            rope = self.rope_emb(x.shape[1])

        x = x + self.attn(self.norm1(x), attn_bias=attn_bias, rope=rope)
        x = x + self.mlp(self.norm2(x))
        return x


class SSMBlock(nn.Module):
    """Bidirectional Mamba-2 SSM block with the same interface as AttnBlock.

    Runs a forward and (optionally) reverse Mamba-2 scan, sums them,
    then applies a residual MLP -- mirrors the pre-norm residual
    pattern of AttnBlock so it can be used as a drop-in temporal block.

    Input / output shape: (batch, seq_len, hidden_size).
    """

    def __init__(
        self,
        hidden_size,
        ssm_state_dim=128,
        ssm_conv_dim=4,
        ssm_expand=2,
        ssm_headdim=64,
        mlp_ratio=4.0,
        bidirectional=True,
    ):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm is required for SSMBlock. "
                "Install it with: pip install mamba-ssm"
            )

        # Mamba 2 requires d_inner = expand * hidden_size to be divisible
        # by headdim. Validate here so the error is clear.
        d_inner = ssm_expand * hidden_size
        if d_inner % ssm_headdim != 0:
            raise ValueError(
                f"Mamba-2 requires ssm_expand*hidden_size ({d_inner}) to be "
                f"divisible by ssm_headdim ({ssm_headdim})."
            )

        self.bidirectional = bidirectional
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ssm_fwd = Mamba2(
            d_model=hidden_size,
            d_state=ssm_state_dim,
            d_conv=ssm_conv_dim,
            expand=ssm_expand,
            headdim=ssm_headdim,
        )
        if bidirectional:
            self.ssm_rev = Mamba2(
                d_model=hidden_size,
                d_state=ssm_state_dim,
                d_conv=ssm_conv_dim,
                expand=ssm_expand,
                headdim=ssm_headdim,
            )

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, mask=None):
        # mask is accepted for interface compatibility but unused by SSM
        normed = self.norm1(x)
        h = self.ssm_fwd(normed)
        if self.bidirectional:
            # .flip creates a non-contiguous view; Mamba's CUDA kernel
            # expects a contiguous (B, L, D) layout.
            rev_input = normed.flip(1).contiguous()
            h = h + self.ssm_rev(rev_input).flip(1)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x
