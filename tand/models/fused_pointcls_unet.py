"""EfficientNet-B0 UNet with FiLM-based tissue conditioning for nuclei detection/classification.

Provides an EfficientNet-B0 encoder with MBConv decoder blocks, Spatial FiLM modulation
conditioned on tissue probability maps, and optional Bayesian logit bias from tissue priors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.efficientnet import (
    EfficientNet_B0_Weights,
    MBConv,
    MBConvConfig,
    efficientnet_b0,
)
from torchvision.models.feature_extraction import create_feature_extractor

from tand.models.virchow2.encoder import VirchowEncoder
from tand.models.virchow2.seg_head import LinearSegHead
from tand.modules.film import SpatialFiLM, compute_logit_bias, make_tissue_pyramid

# MBConv decoder block configurations
MBConv1_conf = MBConvConfig(
    expand_ratio=1, kernel=3, stride=1, input_channels=1280, out_channels=320, num_layers=1
)
MBConv2_conf = MBConvConfig(
    expand_ratio=6, kernel=5, stride=1, input_channels=432, out_channels=192, num_layers=4
)
MBConv3_conf = MBConvConfig(
    expand_ratio=6, kernel=5, stride=1, input_channels=192, out_channels=112, num_layers=3
)
MBConv4_conf = MBConvConfig(
    expand_ratio=6, kernel=3, stride=1, input_channels=152, out_channels=80, num_layers=3
)
MBConv5_conf = MBConvConfig(
    expand_ratio=6, kernel=5, stride=1, input_channels=104, out_channels=40, num_layers=2
)
MBConv6_conf = MBConvConfig(
    expand_ratio=6, kernel=3, stride=1, input_channels=56, out_channels=24, num_layers=2
)
MBConv7_conf = MBConvConfig(
    expand_ratio=1, kernel=3, stride=1, input_channels=24, out_channels=16, num_layers=2
)


class EfficientUnet_MBConv_PointCLS_Fused(nn.Module):
    """EfficientUNet decoder with FiLM on classification path and optional logit bias."""

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int = 10,
        num_tissue: int = 6,
        film_limit: float = 0.5,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_tissue = int(num_tissue)
        self.encoder = encoder
        self.skip = create_feature_extractor(
            self.encoder,
            return_nodes={
                "1.0.block.2": "layer0",
                "2.1.add": "layer1",
                "3.1.add": "layer2",
                "5.1.add": "layer3",
                "8": "encoder_output",
            },
        )

        self.block2_upsample = nn.ConvTranspose2d(320, 320, kernel_size=2, stride=2)  # 7->14
        self.block4_upsample = nn.ConvTranspose2d(112, 112, kernel_size=2, stride=2)  # 14->28
        self.block5_upsample = nn.ConvTranspose2d(80, 80, kernel_size=2, stride=2)  # 28->56
        self.block6_upsample = nn.ConvTranspose2d(40, 40, kernel_size=2, stride=2)  # 56->112
        self.block7_upsample = nn.ConvTranspose2d(24, 24, kernel_size=2, stride=2)  # 112->224

        self.MBConv1 = MBConv(MBConv1_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
        self.MBConv2 = MBConv(MBConv2_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
        self.MBConv3 = MBConv(MBConv3_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
        self.MBConv4 = MBConv(MBConv4_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
        self.MBConv5 = MBConv(MBConv5_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
        self.MBConv6 = MBConv(MBConv6_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
        self.MBConv7 = MBConv(MBConv7_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)

        # FiLM modulators for classification path
        self.film_p4 = SpatialFiLM(112, num_tissue, hidden=128, limit=film_limit)  # 14x14
        self.film_p3 = SpatialFiLM(80, num_tissue, hidden=128, limit=film_limit)  # 28x28
        self.film_p2 = SpatialFiLM(40, num_tissue, hidden=64, limit=film_limit)  # 56x56
        self.film_p1 = SpatialFiLM(24, num_tissue, hidden=32, limit=film_limit)  # 112x112
        self.film_p0 = SpatialFiLM(16, num_tissue, hidden=32, limit=film_limit)  # 224x224

        self.head_heatmap = nn.Conv2d(16, 1, 1)
        self.head_class = nn.Conv2d(16, self.num_classes, 1)

        self.register_buffer(
            "log_pc_given_t", torch.zeros(self.num_classes, self.num_tissue), persistent=False
        )
        self.lam_bias = 1.0
        self.conf_thr = 0.0

    def set_prior(self, log_pc_given_t: torch.Tensor, lam: float = 1.0, conf_thr: float = 0.0):
        """Set the tissue-class prior for logit bias.

        Parameters
        ----------
        log_pc_given_t : torch.Tensor
            [C, T] log-probability matrix.
        lam : float
            Bias strength scaling factor.
        conf_thr : float
            Minimum tissue confidence threshold for applying bias.
        """
        self.log_pc_given_t.copy_(log_pc_given_t)
        self.lam_bias = float(lam)
        self.conf_thr = float(conf_thr)

    def forward(
        self,
        img: torch.Tensor,
        tissue_logits_16: torch.Tensor | None = None,
        tissue_logits_224: torch.Tensor | None = None,
        use_film: bool = True,
        use_bias: bool = True,
    ):
        """Forward pass through the EfficientUNet decoder.

        Parameters
        ----------
        img : torch.Tensor
            Input image [B, 3, H, W].
        tissue_logits_16 : torch.Tensor or None
            Tissue logits at 16x16 from Virchow segmentation head.
        tissue_logits_224 : torch.Tensor or None
            Tissue logits at 224x224 from Virchow segmentation head.
        use_film : bool
            Whether to apply FiLM modulation on the classification path.
        use_bias : bool
            Whether to apply tissue-prior logit bias.

        Returns
        -------
        dict
            Dictionary with keys: 'heatmap_logits', 'class_logits'.
        """
        _ = self.encoder(img)
        feats = self.skip(img)
        enc_out = feats["encoder_output"]

        x7 = self.MBConv1(enc_out)

        x14 = self.block2_upsample(x7)
        x14 = torch.cat([x14, feats["layer3"]], dim=1)
        x14 = self.MBConv2(x14)
        x14 = self.MBConv3(x14)

        x28 = self.block4_upsample(x14)
        x28 = torch.cat([x28, feats["layer2"]], dim=1)
        x28 = self.MBConv4(x28)

        x56 = self.block5_upsample(x28)
        x56 = torch.cat([x56, feats["layer1"]], dim=1)
        x56 = self.MBConv5(x56)

        x112 = self.block6_upsample(x56)
        x112 = torch.cat([x112, feats["layer0"]], dim=1)
        x112 = self.MBConv6(x112)

        x224 = self.block7_upsample(x112)
        x224 = self.MBConv7(x224)

        heatmap_logits = self.head_heatmap(x224)

        x_cls = x224
        if (tissue_logits_16 is not None) and use_film:
            # Dynamically upsample tissue maps to match current feature sizes
            p16 = F.softmax(tissue_logits_16, dim=1)  # [B,T,16,16]
            s14 = F.interpolate(p16, size=x14.shape[-2:], mode="bilinear", align_corners=False)
            s28 = F.interpolate(p16, size=x28.shape[-2:], mode="bilinear", align_corners=False)
            s56 = F.interpolate(p16, size=x56.shape[-2:], mode="bilinear", align_corners=False)
            s112 = F.interpolate(
                p16, size=x112.shape[-2:], mode="bilinear", align_corners=False
            )
            # Prefer the higher-res 224 logits for the final 1x scale, resized to x224
            if tissue_logits_224 is not None:
                s224 = F.softmax(tissue_logits_224, dim=1)
                s224 = F.interpolate(
                    s224, size=x224.shape[-2:], mode="bilinear", align_corners=False
                )
            else:
                s224 = F.interpolate(
                    p16, size=x224.shape[-2:], mode="bilinear", align_corners=False
                )

            x14 = self.film_p4(x14, s14)
            x28 = self.film_p3(x28, s28)
            x56 = self.film_p2(x56, s56)
            x112 = self.film_p1(x112, s112)
            x_cls = self.film_p0(x224, s224)

        class_logits = self.head_class(x_cls)

        if (
            (self.log_pc_given_t.abs().sum() > 0)
            and (tissue_logits_224 is not None)
            and use_bias
        ):
            s_full = F.softmax(tissue_logits_224, dim=1)
            s_full = F.interpolate(
                s_full, size=class_logits.shape[-2:], mode="bilinear", align_corners=False
            )
            bias = compute_logit_bias(
                s_full, self.log_pc_given_t, lam=self.lam_bias, conf_thr=self.conf_thr
            )
            class_logits = class_logits + bias

        return {"heatmap_logits": heatmap_logits, "class_logits": class_logits}


class VirchowFusedNet(nn.Module):
    """Container for Virchow encoder, tissue head, and fused detection UNet.

    Combines the Virchow2 tissue segmentation branch with an EfficientNet-B0 UNet
    for tissue-aware nuclei detection and classification.
    """

    def __init__(
        self,
        num_classes: int,
        num_tissue: int,
        film_limit: float = 0.5,
        pretrained: bool = True,
    ):
        super().__init__()
        enc_effi = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None
        ).features
        self.det = EfficientUnet_MBConv_PointCLS_Fused(
            enc_effi, num_classes=num_classes, num_tissue=num_tissue, film_limit=film_limit
        )
        self.vir = VirchowEncoder(backbone="virchow2", freeze=True, pretrained=pretrained)
        self.seg = LinearSegHead(embed_dim=1280, num_classes=num_tissue, grid=16)

    def forward(self, img: torch.Tensor, use_film: bool = True, use_bias: bool = True):
        """Forward pass through the full Virchow-fused network.

        Parameters
        ----------
        img : torch.Tensor
            Input image [B, 3, H, W].
        use_film : bool
            Whether to apply FiLM modulation.
        use_bias : bool
            Whether to apply tissue-prior logit bias.

        Returns
        -------
        dict
            Dictionary with keys: 'heatmap_logits', 'class_logits',
            'tissue_logits_224', 'tissue_logits_16'.
        """
        # Always run Virchow branch at 224x224 to ensure 16x16 token grid
        with torch.no_grad():
            x224 = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
            tokens = self.vir.forward_features(x224)
            tissue224, tissue16 = self.seg(tokens)
        out = self.det(
            img,
            tissue_logits_16=tissue16,
            tissue_logits_224=tissue224,
            use_film=use_film,
            use_bias=use_bias,
        )
        return {**out, "tissue_logits_224": tissue224, "tissue_logits_16": tissue16}
