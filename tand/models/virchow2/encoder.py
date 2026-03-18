"""Virchow / Virchow2 vision transformer encoder wrapper.

Loads pre-trained Virchow or Virchow2 foundation models from HuggingFace Hub
via timm and extracts patch-level token embeddings for downstream tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VirchowEncoder(nn.Module):
    """Wrapper around timm Virchow/Virchow2 CLIP-like encoders.

    Loads from HF hub via timm: hf-hub:paige-ai/Virchow or Virchow2.
    Default is Virchow2. Optionally unfreezes last N blocks for fine-tuning.

    Parameters
    ----------
    backbone : str
        Model variant: 'virchow' or 'virchow2'.
    freeze : bool
        Whether to freeze all encoder parameters.
    unfreeze_last_n : int
        Number of final transformer blocks to unfreeze (overrides freeze for those blocks).
    pretrained : bool
        Whether to load pretrained weights from HuggingFace Hub.
    """

    def __init__(
        self,
        backbone: str = "virchow2",
        freeze: bool = True,
        unfreeze_last_n: int = 0,
        pretrained: bool = True,
    ):
        super().__init__()
        try:
            import timm
            from timm.layers import SwiGLUPacked
        except ImportError as e:
            raise ImportError(
                "timm (and possibly huggingface_hub) required for Virchow2: "
                "pip install timm huggingface_hub"
            ) from e

        name = "Virchow2" if backbone.lower() == "virchow2" else "Virchow"
        self.is_v2 = name == "Virchow2"
        self.backbone = timm.create_model(
            f"hf-hub:paige-ai/{name}",
            pretrained=pretrained,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
        )
        self.embed_dim = 1280
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if unfreeze_last_n and unfreeze_last_n > 0:
            blocks = getattr(self.backbone, "blocks", None) or getattr(
                self.backbone, "stages", None
            )
            if blocks is not None:
                for blk in blocks[-unfreeze_last_n:]:
                    for p in blk.parameters():
                        p.requires_grad = True

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from Virchow/Virchow2 outputs.

        Parameters
        ----------
        x : torch.Tensor
            Input image [B, 3, 224, 224].

        Returns
        -------
        torch.Tensor
            Patch tokens [B, 256, 1280] (excluding CLS and register tokens).
        """
        out = self.backbone(x)
        if self.is_v2:
            patch = out[:, 5:, :]
        else:
            patch = out[:, 1:, :]
        return patch
