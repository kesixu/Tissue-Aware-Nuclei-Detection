"""HTCNet-style distribution consistency loss.

Aligns the predicted per-tissue class distribution P_hat(c|t) with a prior
P(c|t) via mean-squared error.  This variant accepts either soft tissue
probability maps or hard integer tissue label maps.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def distribution_consistency_loss(
    class_logits: torch.Tensor,
    tissue_map_or_probs: torch.Tensor,
    prior_ct: torch.Tensor,
    conf_thr: float = 0.7,
    tau: float = 1.0,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """HTCNet Eq.(4) style distribution consistency loss aligning P_hat(c|t) with priors.

    Args:
        class_logits: Tensor [B, C, H, W] raw logits.
        tissue_map_or_probs: Either soft probabilities/logits [B, T, H, W] or an
            integer tissue label map [B, H, W] / [B, 1, H, W].
        prior_ct: Tensor [C, T] with probabilities P(c|t).
        conf_thr: Mask low-confidence tissue predictions.
        tau: Temperature for logits-to-probabilities conversion.
        eps: Numerical stability epsilon.
        reduction: ``"mean"`` or ``"sum"``.

    Returns:
        loss: Scalar tensor.
        P_hat: Detached tensor [C, T] estimated from predictions.
    """
    B, C, H, W = class_logits.shape
    p: torch.Tensor = F.softmax(class_logits, dim=1)  # [B, C, H, W]

    # Build tissue probabilities s: [B, T, H, W]
    if tissue_map_or_probs.dim() == 4 and tissue_map_or_probs.size(1) > 1:
        x: torch.Tensor = tissue_map_or_probs
        # Values > 1.5 indicate unnormalized logits (raw network output)
        # rather than probabilities already in [0, 1], so we apply softmax.
        if x.max() > 1.5:
            s: torch.Tensor = F.softmax(x / max(tau, eps), dim=1)
        else:
            s = x
        conf: torch.Tensor = s.max(dim=1).values
        mask: torch.Tensor = (conf >= conf_thr).float()
    else:
        if tissue_map_or_probs.dim() == 4 and tissue_map_or_probs.size(1) == 1:
            tm: torch.Tensor = tissue_map_or_probs[:, 0]
        else:
            tm = tissue_map_or_probs
        T: int = int(tm.max().item() + 1) if tm.numel() > 0 else 1
        s = F.one_hot(tm.long(), num_classes=T).permute(0, 3, 1, 2).float()
        mask = torch.ones((B, H, W), device=s.device, dtype=s.dtype)

    mask = mask.to(p.device)
    s = s.to(p.device)
    prior: torch.Tensor = prior_ct.to(p.device, dtype=p.dtype).clamp_min(eps)

    num: torch.Tensor = torch.einsum(
        "b t h w, b c h w, b h w -> b c t", s, p, mask
    )  # [B, C, T]
    den: torch.Tensor = torch.einsum(
        "b t h w, b h w -> b t", s, mask
    ).clamp_min(eps)  # [B, T]
    P_hat: torch.Tensor = (num / den.unsqueeze(1)).mean(dim=0)  # [C, T]

    loss: torch.Tensor = (P_hat - prior).pow(2).sum()
    if reduction == "mean":
        loss = loss / (C * P_hat.shape[1])
    return loss, P_hat.detach()
