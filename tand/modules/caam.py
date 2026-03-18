"""Context-Aware Adaptive Modulation (CAAM).

Combines AdaLN-Zero gating (Peebles & Xie, ICCV 2023) with spatial FiLM-style
affine modulation.  Key improvements over vanilla Spatial-FiLM:

1. **Learnable gate**: a tanh-bounded gate (zero-initialised) that starts as an
   identity mapping and gradually opens during training.  This resolves the
   suppressive-gate-initialisation problem documented in earlier TAND variants.
2. **Pre-modulation normalisation**: GroupNorm before the affine transform
   stabilises the conditioning signal and decouples it from feature magnitudes.
3. **SiLU adapter**: smoother gradient flow than ReLU in the conditioning path.

Ablation flags (for factorial experiments):
- ``use_gate=False``: disables the learnable residual gate (falls back to
  standard affine modulation without residual gating).
- ``use_norm=False``: disables GroupNorm pre-modulation (applies gamma/beta
  directly to the input feature map, i.e. standard Spatial-FiLM behaviour).
Setting both ``use_gate=False, use_norm=False`` recovers standard Spatial-FiLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CAAM(nn.Module):
    """Context-Aware Adaptive Modulation.

    Parameters
    ----------
    feat_ch : int
        Number of channels in the feature map to be modulated.
    tissue_ch : int
        Number of tissue probability channels (default: 6 for PUMA).
    hidden : int
        Hidden dimension of the conditioning adapter.
    limit : float
        Tanh scaling bound for gamma and beta.
    use_gate : bool
        Enable learnable residual gate (AdaLN-Zero style). Default True.
    use_norm : bool
        Enable GroupNorm pre-modulation. Default True.
    """

    def __init__(
        self,
        feat_ch: int,
        tissue_ch: int = 6,
        hidden: int = 128,
        limit: float = 0.5,
        use_gate: bool = True,
        use_norm: bool = True,
    ):
        super().__init__()
        self.limit = float(limit)
        self.use_gate = bool(use_gate)
        self.use_norm = bool(use_norm)

        # GroupNorm before modulation (AdaLN-style); only built when use_norm=True
        if self.use_norm:
            num_groups = min(32, feat_ch)
            while feat_ch % num_groups != 0:
                num_groups -= 1
            self.norm = nn.GroupNorm(num_groups, feat_ch)
        else:
            self.norm = None

        # Adapter output channels: 3C (gamma+beta+gate) when use_gate, else 2C (gamma+beta)
        out_ch = 3 * feat_ch if self.use_gate else 2 * feat_ch
        self.adapter = nn.Sequential(
            nn.Conv2d(tissue_ch, hidden, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, out_ch, 3, padding=1),
        )

        # Zero-init gate slice so gate = tanh(0) = 0 at start (identity residual)
        last_conv = self.adapter[-1]
        C = feat_ch
        if self.use_gate:
            # Weight slices: gamma [0:C], beta [C:2C], gate [2C:3C]
            nn.init.zeros_(last_conv.weight[2 * C :])
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias[2 * C :])
                nn.init.zeros_(last_conv.bias[: 2 * C])
        else:
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias[: 2 * C])

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Apply context-aware modulation.

        Parameters
        ----------
        x : torch.Tensor
            [B, C, H, W] feature map.
        s : torch.Tensor
            [B, T, H, W] tissue probability map (softmaxed).

        Returns
        -------
        torch.Tensor
            Modulated feature map with same shape as x.
        """
        mod = self.adapter(s)

        if self.use_gate:
            gamma, beta, gate = mod.chunk(3, dim=1)
            gate = torch.tanh(gate)  # 0 at init => identity
        else:
            gamma, beta = mod.chunk(2, dim=1)

        gamma = torch.tanh(gamma) * self.limit  # [-limit, +limit]
        beta = torch.tanh(beta) * (self.limit * 0.5)

        x_in = self.norm(x) if self.use_norm else x
        modulated = (1 + gamma) * x_in + beta

        if self.use_gate:
            return x + gate * modulated
        else:
            return modulated
