"""Spatial FiLM (Feature-wise Linear Modulation) for tissue-conditioned classification.

Provides affine modulation of feature maps conditioned on tissue probability maps,
along with utilities for constructing multi-scale tissue pyramids and computing
Bayesian logit biases from tissue priors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFiLM(nn.Module):
    """Spatial FiLM modulation conditioned on tissue logits/prob maps.

    x: [B,C,H,W] features; s: [B,T,H,W] tissue maps (logits or prob)
    Produces gamma,beta via a small conv adapter and applies: x*(1+gamma)+beta
    """

    def __init__(self, feat_ch: int, tissue_ch: int = 6, hidden: int = 0, limit: float = 0.5):
        super().__init__()
        self.limit = float(limit)
        if hidden and hidden > 0:
            self.adapter = nn.Sequential(
                nn.Conv2d(tissue_ch, hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, 2 * feat_ch, 3, padding=1),
            )
        else:
            self.adapter = nn.Conv2d(tissue_ch, 2 * feat_ch, 1)
        # zero-init last conv
        last = self.adapter[-1] if isinstance(self.adapter, nn.Sequential) else self.adapter
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        mod = self.adapter(s)
        gamma, beta = mod.chunk(2, dim=1)
        gamma = torch.tanh(gamma) * self.limit
        beta = torch.tanh(beta) * (self.limit * 0.5)
        return x * (1.0 + gamma) + beta


def make_tissue_pyramid(tissue_logits_16: torch.Tensor, sizes):
    """Upsample tissue logits to multiple spatial scales as probs.

    Returns list of prob maps matching sizes in order.
    """
    probs = F.softmax(tissue_logits_16, dim=1)
    outs = []
    for (H, W) in sizes:
        if (H, W) == tissue_logits_16.shape[-2:]:
            outs.append(probs)
        else:
            outs.append(F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False))
    return outs


def compute_logit_bias(s_full, log_pc_given_t, lam=1.0, conf_thr=0.7):
    """Compute tissue-prior logit bias for class predictions.

    Parameters
    ----------
    s_full : torch.Tensor
        [B, T, H, W] tissue probabilities.
    log_pc_given_t : torch.Tensor
        [C, T] log P(c|t) prior matrix.
    lam : float
        Bias strength scaling factor.
    conf_thr : float
        Only apply bias where max tissue probability >= conf_thr.

    Returns
    -------
    torch.Tensor
        [B, C, H, W] bias to add to class logits.
    """
    conf = s_full.max(dim=1).values  # [B,H,W]
    mask = (conf >= conf_thr).float().unsqueeze(1)  # [B,1,H,W]
    bias = torch.einsum("b t h w, c t -> b c h w", s_full, log_pc_given_t)  # [B,C,H,W]
    return lam * bias * mask
