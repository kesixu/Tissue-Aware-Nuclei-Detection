"""TAND fused network: DINOv3-ConvNeXt UNet + Virchow tissue segmentation via Spatial FiLM.

This module implements the main TAND architecture that fuses a DINOv3-ConvNeXt UNet
for nuclei detection/classification with a Virchow2 tissue encoder via Spatial FiLM
modulation layers and optional Bayesian tissue-prior logit bias.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tand.models.backbone import DINOv3ConvNeXtUNet
from tand.models.virchow2.encoder import VirchowEncoder
from tand.models.virchow2.seg_head import LinearSegHead
from tand.modules.film import SpatialFiLM, compute_logit_bias

logger = logging.getLogger(__name__)


class DINOv3VirchowFused(nn.Module):
    """DINOv3-ConvNeXt UNet detection fused with Virchow tissue segmentation via Spatial FiLM.

    - Input: full image (e.g., 1024x1024)
    - Virchow branch: internally resizes to 224x224, outputs tissue logits at 16x16 and 224x224
    - Detection: DINOv3ConvNeXtUNet; classification path modulated by FiLM residuals;
      detection head unaffected
    - Output keys: heatmap_logits, class_logits, tissue_logits_224, tissue_logits_16
    """

    def __init__(
        self,
        num_cell_classes: int,
        num_tissue: int = 6,
        film_limit: float = 0.5,
        dino_variant: str = "convnext_small",
        dino_up_mode: str = "upsample",
        dino_load_hf: bool = False,
        lam_bias: float = 0.8,
        conf_thr: float = 0.7,
        tau: float = 1.0,
        prior_path: str | None = None,
        film_scales: str = "16,8,4",
    ):
        super().__init__()
        # Parse active FiLM scales (e.g. "16,8,4" or "16,8" or "4")
        if isinstance(film_scales, str) and film_scales.strip():
            self.active_film_scales = {int(s.strip()) for s in film_scales.split(",")}
        else:
            self.active_film_scales = {16, 8, 4}
        self.lam_bias = float(lam_bias)
        self.conf_thr = float(conf_thr)
        self.tau = float(tau)
        self.prior_path = prior_path
        self.register_buffer("log_pc_given_t", None, persistent=False)
        if self.prior_path and os.path.isfile(self.prior_path):
            logp = np.load(self.prior_path)  # [C,T]
            self.set_tissue_prior(torch.tensor(logp, dtype=torch.float32))
        elif self.prior_path:
            logger.warning(
                "Prior file not found at %s; using uniform fallback until built.",
                self.prior_path,
            )

        # Detection backbone/decoder (DINOv3 ConvNeXt UNet)
        self.det = DINOv3ConvNeXtUNet(
            variant=dino_variant,
            num_classes=num_cell_classes,
            up_mode=dino_up_mode,
            load_hf_encoder=dino_load_hf,
        )

        # Virchow branch (frozen by default at trainer time)
        self.vir = VirchowEncoder(backbone="virchow2", freeze=True)
        self.seg = LinearSegHead(embed_dim=self.vir.embed_dim, num_classes=num_tissue, grid=16)

        # FiLM on classification path at 1/16, 1/8, 1/4
        c1, c2, c3, _ = self.det.encoder.channels  # 1/4, 1/8, 1/16, 1/32
        self.film16 = SpatialFiLM(
            feat_ch=c3, tissue_ch=num_tissue, hidden=128, limit=film_limit
        )  # 1/16
        self.film8 = SpatialFiLM(
            feat_ch=c2, tissue_ch=num_tissue, hidden=128, limit=film_limit
        )  # 1/8
        self.film4 = SpatialFiLM(
            feat_ch=c1, tissue_ch=num_tissue, hidden=64, limit=film_limit
        )  # 1/4

        # Project FiLM residuals from 1/16 and 1/8 to 1/4 channel dimension for safe summation
        self.proj16_to_4 = nn.Conv2d(c3, c1, kernel_size=1, bias=False)
        self.proj8_to_4 = nn.Conv2d(c2, c1, kernel_size=1, bias=False)

        # Initialize projections to zero so fused path starts as identity
        nn.init.zeros_(self.proj16_to_4.weight)
        nn.init.zeros_(self.proj8_to_4.weight)

    def freeze_virchow(self, freeze: bool = True):
        """Freeze or unfreeze the Virchow encoder and segmentation head."""
        for p in self.vir.parameters():
            p.requires_grad = not freeze
        for p in self.seg.parameters():
            p.requires_grad = not freeze

    def set_tissue_prior(self, log_pc_given_t: torch.Tensor):
        """Set the tissue-class prior matrix.

        Parameters
        ----------
        log_pc_given_t : torch.Tensor or None
            [C, T] log-probability matrix, or None to clear.
        """
        if log_pc_given_t is None:
            self.log_pc_given_t = None
            return
        tensor = log_pc_given_t.detach()
        device = None
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # Parameters not yet registered (called during __init__);
            # will migrate when model moves to device
            device = None
        if device is not None:
            tensor = tensor.to(device)
        self.log_pc_given_t = tensor

    @torch.no_grad()
    def _vir_forward(self, img_full: torch.Tensor):
        """Run Virchow encoder + segmentation head at 224x224."""
        # Resize input to 224x224 for Virchow
        x224 = F.interpolate(img_full, size=(224, 224), mode="bilinear", align_corners=False)
        tokens = self.vir.forward_features(x224)  # [B,256,1280]
        logits224, logits16 = self.seg(tokens)
        return logits16, logits224

    def forward(
        self, img_full: torch.Tensor, use_film: bool = True, use_bias: bool = False
    ) -> dict:
        """Forward pass through the fused TAND network.

        Parameters
        ----------
        img_full : torch.Tensor
            Input image [B, C, H, W] at full resolution.
        use_film : bool
            Whether to apply FiLM modulation on the classification path.
        use_bias : bool
            Whether to apply tissue-prior logit bias.

        Returns
        -------
        dict
            Dictionary with keys: 'heatmap_logits', 'class_logits',
            'tissue_logits_224', 'tissue_logits_16'.
        """
        B, C, H, W = img_full.shape

        # Virchow tissue predictions
        tissue16, tissue224 = self._vir_forward(img_full)
        tissue_full = F.interpolate(tissue224, size=(H, W), mode="bilinear", align_corners=False)

        # DINOv3 features and decoder path (replicate UNet forward while exposing stages)
        x = img_full
        if getattr(self.det, "needs_adapter", False) and self.det.input_adapter is not None:
            x = self.det.input_adapter(x)
        f1, f2, f3, f4 = self.det.encoder(x)  # 1/4,1/8,1/16,1/32
        x16 = self.det.up3(f4, f3)  # 1/16
        x8 = self.det.up2(x16, f2)  # 1/8
        x4 = self.det.up1(x8, f1)  # 1/4

        # Detection head (unmodulated)
        heat_1_4 = self.det.head_heatmap(x4)

        # Classification head with FiLM residuals
        if use_film:
            p16 = torch.softmax(tissue16, dim=1)  # [B,T,16,16]
            scales = self.active_film_scales

            x_cls = x4
            if 4 in scales:
                s4 = F.interpolate(
                    p16, size=x4.shape[-2:], mode="bilinear", align_corners=False
                )
                x_cls = x_cls + (self.film4(x4, s4) - x4)
            if 8 in scales:
                s8 = F.interpolate(
                    p16, size=x8.shape[-2:], mode="bilinear", align_corners=False
                )
                d8 = self.film8(x8, s8) - x8
                d8_up = F.interpolate(
                    d8, size=x4.shape[-2:], mode="bilinear", align_corners=False
                )
                x_cls = x_cls + self.proj8_to_4(d8_up)
            if 16 in scales:
                s16 = F.interpolate(
                    p16, size=x16.shape[-2:], mode="bilinear", align_corners=False
                )
                d16 = self.film16(x16, s16) - x16
                d16_up = F.interpolate(
                    d16, size=x4.shape[-2:], mode="bilinear", align_corners=False
                )
                x_cls = x_cls + self.proj16_to_4(d16_up)
        else:
            x_cls = x4

        cls_1_4 = self.det.head_class(x_cls)

        if use_bias and (self.lam_bias > 0):
            tissue_prob_224 = torch.softmax(tissue224 / self.tau, dim=1)
            tissue_prob = F.interpolate(
                tissue_prob_224, size=cls_1_4.shape[-2:], mode="bilinear", align_corners=False
            )

            log_pc_ct = self.log_pc_given_t
            if log_pc_ct is None or log_pc_ct.numel() == 1:
                C = cls_1_4.shape[1]
                T = tissue_prob.shape[1]
                uniform = torch.full(
                    (C, T), 1.0 / max(C, 1), device=cls_1_4.device, dtype=cls_1_4.dtype
                )
                log_pc_ct = torch.log(uniform.clamp_min(1e-8))
            else:
                log_pc_ct = log_pc_ct.to(cls_1_4.device, dtype=cls_1_4.dtype)

            bias = compute_logit_bias(
                tissue_prob,
                log_pc_ct,
                lam=self.lam_bias,
                conf_thr=self.conf_thr,
            )
            cls_1_4 = cls_1_4 + bias

        # Upsample to original size and return
        heat_out = F.interpolate(heat_1_4, size=(H, W), mode="bilinear", align_corners=False)
        cls_out = F.interpolate(cls_1_4, size=(H, W), mode="bilinear", align_corners=False)

        return {
            "heatmap_logits": heat_out,
            "class_logits": cls_out,
            "tissue_logits_224": tissue224,
            "tissue_logits_16": tissue16,
        }
