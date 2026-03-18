"""EfficientNet-B0 UNet with MBConv decoder for nuclei detection and classification.

Standalone EfficientNet-B0 encoder with MBConv-based UNet decoder, producing dual-head
outputs (heatmap for detection, class logits for classification) and optional contrastive
embeddings.
"""

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

# MBConv decoder block configurations (matching EfficientNet-B0 feature dimensions)
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


class EfficientUnet_MBConv_PointCLS(nn.Module):
    """EfficientNet-B0 UNet with dual-head outputs for detection and classification.

    Outputs:
      - head_heatmap: 1-channel center probability logits (detection)
      - head_class: C-channel class logits (classification)
      - head_embed: Optional embedding for contrastive learning
    """

    def __init__(self, encoder, num_classes=10, emb_dim=64):
        super().__init__()
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

        # Upsample + MBConv decoder blocks
        self.block2_upsample = nn.ConvTranspose2d(320, 320, kernel_size=2, stride=2)
        self.block4_upsample = nn.ConvTranspose2d(112, 112, kernel_size=2, stride=2)
        self.block5_upsample = nn.ConvTranspose2d(80, 80, kernel_size=2, stride=2)
        self.block6_upsample = nn.ConvTranspose2d(40, 40, kernel_size=2, stride=2)
        self.block7_upsample = nn.ConvTranspose2d(24, 24, kernel_size=2, stride=2)

        self.MBConv1 = MBConv(MBConv1_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
        self.MBConv2 = MBConv(MBConv2_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
        self.MBConv3 = MBConv(MBConv3_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
        self.MBConv4 = MBConv(MBConv4_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
        self.MBConv5 = MBConv(MBConv5_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
        self.MBConv6 = MBConv(MBConv6_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
        self.MBConv7 = MBConv(MBConv7_conf, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)

        # Detection head (1 channel heatmap logits)
        self.head_heatmap = nn.Conv2d(16, 1, kernel_size=1)
        # Classification head (C channel class logits)
        self.head_class = nn.Conv2d(16, num_classes, kernel_size=1)
        # Optional contrastive embedding head
        self.head_embed = nn.Sequential(
            nn.Conv2d(16, emb_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim, emb_dim, kernel_size=1, bias=False),
        )

    def forward(self, x):
        """Forward pass through the EfficientUNet.

        Parameters
        ----------
        x : torch.Tensor
            Input image [B, 3, H, W].

        Returns
        -------
        dict
            Dictionary with keys: 'feat', 'heatmap_logits', 'class_logits', 'embedding'.
        """
        # EfficientNet features and skip connections
        feats = self.skip(x)
        enc_out = feats["encoder_output"]  # (B, 1280, 7, 7) for B0

        # Decode block 1
        x = self.MBConv1(enc_out)  # (B, 320, 7, 7)

        # Decode block 2
        x = self.block2_upsample(x)  # -> (B, 320, 14, 14)
        x = torch.cat([x, feats["layer3"]], dim=1)  # 320 + 112 = 432
        x = self.MBConv2(x)  # -> (B, 192, 14, 14)

        # Decode block 3
        x = self.MBConv3(x)  # -> (B, 112, 14, 14)

        # Decode block 4
        x = self.block4_upsample(x)  # -> (B, 112, 28, 28)
        x = torch.cat([x, feats["layer2"]], dim=1)  # 112 + 40 = 152
        x = self.MBConv4(x)  # -> (B, 80, 28, 28)

        # Decode block 5
        x = self.block5_upsample(x)  # -> (B, 80, 56, 56)
        x = torch.cat([x, feats["layer1"]], dim=1)  # 80 + 24 = 104
        x = self.MBConv5(x)  # -> (B, 40, 56, 56)

        # Decode block 6
        x = self.block6_upsample(x)  # -> (B, 40, 112, 112)
        x = torch.cat([x, feats["layer0"]], dim=1)  # 40 + 16 = 56
        x = self.MBConv6(x)  # -> (B, 24, 112, 112)

        # Decode block 7
        x = self.block7_upsample(x)  # -> (B, 24, 224, 224)
        x = self.MBConv7(x)  # -> (B, 16, 224, 224)

        heatmap_logits = self.head_heatmap(x)
        class_logits = self.head_class(x)
        embedding = F.normalize(self.head_embed(x), dim=1)

        return {
            "feat": x,
            "heatmap_logits": heatmap_logits,
            "class_logits": class_logits,
            "embedding": embedding,
        }


def get_efficientunet_b0(num_classes=10, pretrained=True, emb_dim=64):
    """Build an EfficientNet-B0 based UNet for nuclei detection/classification.

    Parameters
    ----------
    num_classes : int
        Number of cell type classes.
    pretrained : bool
        Whether to use ImageNet-pretrained encoder weights.
    emb_dim : int
        Dimensionality of the contrastive embedding head.

    Returns
    -------
    EfficientUnet_MBConv_PointCLS
        Constructed model instance.
    """
    enc = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    encoder = enc.features
    model = EfficientUnet_MBConv_PointCLS(encoder, num_classes=num_classes, emb_dim=emb_dim)
    return model
