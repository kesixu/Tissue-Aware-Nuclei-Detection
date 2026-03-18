"""Linear segmentation head for tissue classification from Virchow tokens.

Projects Virchow/Virchow2 patch token embeddings to tissue class logits at both
the native grid resolution (16x16) and upsampled to 224x224 for dense prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearSegHead(nn.Module):
    """Linear projection head mapping Virchow tokens to tissue logits.

    Expects Virchow/Virchow2 patch tokens [B, 256, embed_dim] arranged on a 16x16 grid.

    Parameters
    ----------
    embed_dim : int
        Token embedding dimensionality (1280 for Virchow2).
    num_classes : int
        Number of tissue classes (default: 6 for PUMA).
    grid : int
        Spatial grid size of the patch tokens (default: 16).
    dropout : float
        Spatial dropout rate before projection.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        - logits_224: [B, T, 224, 224] upsampled tissue logits.
        - logits_grid: [B, T, grid, grid] native-resolution tissue logits.
    """

    def __init__(
        self, embed_dim: int = 1280, num_classes: int = 6, grid: int = 16, dropout: float = 0.0
    ):
        super().__init__()
        self.grid = int(grid)
        self.num_classes = int(num_classes)
        self.proj = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.drop = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, patch_tokens: torch.Tensor):
        """Project patch tokens to tissue logits.

        Parameters
        ----------
        patch_tokens : torch.Tensor
            [B, N, D] patch token embeddings from Virchow encoder.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (logits_224, logits_grid) at 224x224 and native grid resolution.
        """
        B, N, D = patch_tokens.shape
        g = self.grid
        x = patch_tokens.transpose(1, 2).reshape(B, D, g, g)
        x = self.drop(x)
        logits_g = self.proj(x)  # [B, T, 16, 16]
        logits_224 = F.interpolate(logits_g, size=(224, 224), mode="bilinear", align_corners=False)
        return logits_224, logits_g
