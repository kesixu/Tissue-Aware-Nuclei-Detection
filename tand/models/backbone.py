"""DINOv3-ConvNeXt UNet backbone for nuclei detection and classification.

Provides a UNet-style architecture with a ConvNeXt encoder (loaded via timm or
HuggingFace transformers) and multi-head decoder outputs for heatmap prediction,
class logits, and optional contrastive embeddings.
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import timm
except ImportError:
    raise ImportError("timm is required. Install via `pip install timm`.")

try:
    from huggingface_hub import hf_hub_download

    _HAS_HF = True
except ImportError:
    _HAS_HF = False

# Optional: use transformers AutoModel to load official HF pretrained DINOv3 ConvNeXt
try:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


# -----------------------------
# Basic Conv Block for Decoder
# -----------------------------
class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + GELU activation block."""

    def __init__(
        self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, groups: int = 1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """Two consecutive ConvBNAct blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1, 1),
            ConvBNAct(out_ch, out_ch, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample -> concat skip -> DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, mode: str = "upsample"):
        super().__init__()
        if mode == "convtrans":
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            up_out_ch = in_ch // 2
        else:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            up_out_ch = in_ch
        self.fuse = DoubleConv(up_out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Align spatial if odd rounding occurs
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.fuse(x)


# -----------------------------
# DINOv3-ConvNeXt Encoder wrapper (timm)
# -----------------------------
class DinoConvNeXtEncoder(nn.Module):
    """ConvNeXt backbone configured with features_only=True to expose stage outputs.

    Returns list of features at reductions ~[1/4, 1/8, 1/16, 1/32].
    The last stage output is used as the UNet bottleneck.
    """

    def __init__(
        self,
        variant: str = "convnext_small",
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        load_hf: bool = False,
        hf_repo: str = "facebook/dinov3-convnext-small-pretrain-lvd1689m",
        hf_filenames: Tuple[str, ...] = (
            "pytorch_model.bin",
            "model.safetensors",
            "checkpoint.pth",
        ),
    ):
        super().__init__()
        self.out_indices = out_indices
        self.hf_model = None
        self.backbone = None
        self.channels = None

        # Prefer official HF pretrained weights via transformers when requested and available
        if load_hf and _HAS_TRANSFORMERS:
            try:
                cfg = AutoConfig.from_pretrained(hf_repo)
                cfg.output_hidden_states = True
                self.hf_model = AutoModel.from_pretrained(hf_repo, config=cfg)
                # Optional normalization from HF image processor
                try:
                    proc = AutoImageProcessor.from_pretrained(hf_repo)
                    mean = torch.tensor(proc.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
                    std = torch.tensor(proc.image_std, dtype=torch.float32).view(1, -1, 1, 1)
                    self.register_buffer("hf_mean", mean, persistent=False)
                    self.register_buffer("hf_std", std, persistent=False)
                except ImportError:
                    logger.warning("Could not load HF image processor; skipping normalization.")
                    self.hf_mean = None
                    self.hf_std = None
                # Infer channels from config if available
                if hasattr(cfg, "hidden_sizes") and cfg.hidden_sizes:
                    self.channels = list(cfg.hidden_sizes)
                else:
                    # Fallback: run one tiny forward to infer channels
                    with torch.no_grad():
                        dummy = torch.zeros(1, 3, 64, 64)
                        out = self.hf_model(
                            pixel_values=dummy, output_hidden_states=True, return_dict=True
                        )
                        hs = out.hidden_states[-4:]
                        self.channels = [t.shape[1] for t in hs]
            except ImportError:
                logger.warning(
                    "Failed to load HF model for %s; falling back to timm.", hf_repo
                )
                self.hf_model = None

        # TIMM fallback (or default if HF not requested)
        if self.hf_model is None:
            self.backbone = timm.create_model(
                variant, features_only=True, out_indices=out_indices, pretrained=True
            )
            try:
                self.channels = list(self.backbone.feature_info.channels())
            except AttributeError:
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 64, 64)
                    feats = self.backbone(dummy)
                    self.channels = [f.shape[1] for f in feats]
            # Optional best-effort: load HF state dict into timm backbone
            if load_hf and _HAS_HF:
                state_dict = None
                for fname in hf_filenames:
                    try:
                        ckpt = hf_hub_download(repo_id=hf_repo, filename=fname)
                        sd = torch.load(ckpt, map_location="cpu")
                        if isinstance(sd, dict) and "state_dict" in sd:
                            sd = sd["state_dict"]
                        state_dict = sd
                        break
                    except Exception:  # noqa: BLE001
                        continue
                if state_dict is not None:
                    missing, unexpected = self.backbone.load_state_dict(
                        state_dict, strict=False
                    )
                    if missing:
                        logger.warning(
                            "HF state dict loaded with %d missing keys.", len(missing)
                        )

    def forward(self, x: torch.Tensor):
        """Extract multi-scale features.

        Returns
        -------
        list[torch.Tensor]
            Feature maps at ~[1/4, 1/8, 1/16, 1/32] resolution.
        """
        # Use transformers HF model path
        if self.hf_model is not None:
            # Normalize as per HF processor if available
            if (
                getattr(self, "hf_mean", None) is not None
                and getattr(self, "hf_std", None) is not None
            ):
                x = (x - self.hf_mean) / self.hf_std
            out = self.hf_model(
                pixel_values=x, output_hidden_states=True, return_dict=True
            )
            # Take last 4 stage outputs as features at ~[1/4,1/8,1/16,1/32]
            hs = list(out.hidden_states)[-4:]
            return hs
        # Otherwise, use timm features_only path
        return self.backbone(x)


# -----------------------------
# UNet-style decoder + multi-head outputs
# -----------------------------
class DINOv3ConvNeXtUNet(nn.Module):
    """DINOv3-ConvNeXt UNet with multi-head outputs for detection and classification.

    Parameters
    ----------
    variant : str
        timm ConvNeXt model variant name.
    num_classes : int
        Number of cell type classes.
    in_ch : int
        Number of input image channels.
    emb_dim : int
        Dimensionality of the contrastive embedding head.
    up_mode : str
        Upsampling mode for decoder: 'upsample' (bilinear) or 'convtrans'.
    load_hf_encoder : bool
        Whether to attempt loading HuggingFace pretrained encoder weights.
    """

    def __init__(
        self,
        variant: str = "convnext_small",
        num_classes: int = 10,
        in_ch: int = 3,
        emb_dim: int = 64,
        up_mode: str = "upsample",
        load_hf_encoder: bool = False,
    ):
        super().__init__()
        self.in_ch = in_ch
        # Store number of classes for external metric code (training scripts expect model.num_classes)
        self.num_classes = num_classes

        self.needs_adapter = in_ch != 3
        self.input_adapter = None
        if self.needs_adapter:
            # Map arbitrary input channels to 3 for ConvNeXt
            self.input_adapter = nn.Conv2d(in_ch, 3, kernel_size=1, bias=False)

        self.encoder = DinoConvNeXtEncoder(variant=variant, load_hf=load_hf_encoder)
        c1, c2, c3, c4 = self.encoder.channels  # 4 stages

        # Decoder: 3 up blocks to go from c4 -> c1 resolution
        self.up3 = UpBlock(in_ch=c4, skip_ch=c3, out_ch=c3, mode=up_mode)
        self.up2 = UpBlock(in_ch=c3, skip_ch=c2, out_ch=c2, mode=up_mode)
        self.up1 = UpBlock(in_ch=c2, skip_ch=c1, out_ch=c1, mode=up_mode)

        # Heads on final feature map (channel = c1)
        self.head_heatmap = nn.Conv2d(c1, 1, kernel_size=1)
        self.head_class = nn.Conv2d(c1, num_classes, kernel_size=1)
        self.head_embed = nn.Sequential(
            nn.Conv2d(c1, emb_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim, emb_dim, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the full UNet.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor [B, C, H, W].

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with keys: 'feat', 'heatmap_logits', 'class_logits', 'embedding'.
        """
        # Preserve original spatial size for final outputs
        orig_h, orig_w = x.shape[-2], x.shape[-1]

        if self.input_adapter is not None:
            x = self.input_adapter(x)

        feats = self.encoder(x)  # list of 4 tensors
        f1, f2, f3, f4 = feats  # low->high depth

        # Decoder upsamples back to 1/4 resolution (ConvNeXt stage 1 stride is 4)
        x = self.up3(f4, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)

        # Produce heads at 1/4 resolution
        heatmap_logits_1_4 = self.head_heatmap(x)
        class_logits_1_4 = self.head_class(x)
        embedding_1_4 = self.head_embed(x)

        # Upsample all outputs to match the original input spatial size
        heatmap_logits = F.interpolate(
            heatmap_logits_1_4, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        class_logits = F.interpolate(
            class_logits_1_4, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )
        # Normalize embedding after upsampling
        embedding = F.normalize(
            F.interpolate(
                embedding_1_4, size=(orig_h, orig_w), mode="bilinear", align_corners=False
            ),
            dim=1,
        )

        return {
            "feat": x,
            "heatmap_logits": heatmap_logits,
            "class_logits": class_logits,
            "embedding": embedding,
        }


def build_model(
    variant: str = "convnext_small",
    num_classes: int = 10,
    in_ch: int = 3,
    emb_dim: int = 64,
    up_mode: str = "upsample",
    load_hf_encoder: bool = False,
) -> DINOv3ConvNeXtUNet:
    """Convenience factory for building a DINOv3ConvNeXtUNet model.

    Parameters
    ----------
    variant : str
        timm ConvNeXt model variant name.
    num_classes : int
        Number of cell type classes.
    in_ch : int
        Number of input image channels.
    emb_dim : int
        Dimensionality of the contrastive embedding head.
    up_mode : str
        Upsampling mode: 'upsample' (bilinear) or 'convtrans'.
    load_hf_encoder : bool
        Whether to attempt loading HuggingFace pretrained encoder weights.

    Returns
    -------
    DINOv3ConvNeXtUNet
        Constructed model instance.
    """
    return DINOv3ConvNeXtUNet(
        variant=variant,
        num_classes=num_classes,
        in_ch=in_ch,
        emb_dim=emb_dim,
        up_mode=up_mode,
        load_hf_encoder=load_hf_encoder,
    )
