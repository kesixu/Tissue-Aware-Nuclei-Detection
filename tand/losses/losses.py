"""Loss functions for point-supervised cell detection and classification."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract_point_logits(
    class_logits: torch.Tensor,
    points_xy: List[torch.Tensor],
    labels: List[torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract per-point logits from spatial class maps. Shared by CE and focal variants."""
    B, C, H, W = class_logits.shape
    all_logits, all_targets = [], []
    for b in range(B):
        pts = points_xy[b]
        if pts.numel() == 0:
            continue
        gx = (pts[:, 0].float() / (W - 1)) * 2 - 1
        gy = (pts[:, 1].float() / (H - 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=1).view(1, -1, 1, 2).to(class_logits.device)
        sampled = F.grid_sample(
            class_logits[b : b + 1], grid, mode="bilinear", align_corners=True
        )
        sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)  # (Ni, C)
        all_logits.append(sampled)
        all_targets.append(labels[b].to(class_logits.device).long())
    if not all_logits:
        return None, None
    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def pointwise_classification_loss(
    class_logits: torch.Tensor,
    points_xy: List[torch.Tensor],
    labels: List[torch.Tensor],
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy loss at annotated point locations.

    Args:
        class_logits: (B, C, H, W) classification logits.
        points_xy: List of (N_i, 2) tensors with (x, y) coordinates.
        labels: List of (N_i,) tensors with class indices.
        class_weights: Optional (C,) tensor of per-class weights.
        label_smoothing: Label smoothing factor.

    Returns:
        Scalar loss tensor.
    """
    logits, targets = _extract_point_logits(class_logits, points_xy, labels)
    if logits is None:
        return class_logits.sum() * 0
    weight = class_weights.to(logits.device) if class_weights is not None else None
    return F.cross_entropy(logits, targets, weight=weight, label_smoothing=label_smoothing)


def pointwise_focal_loss(
    class_logits: torch.Tensor,
    points_xy: List[torch.Tensor],
    labels: List[torch.Tensor],
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal loss variant of pointwise classification loss for handling class imbalance."""
    logits, targets = _extract_point_logits(class_logits, points_xy, labels)
    if logits is None:
        return class_logits.sum() * 0
    # Compute per-sample CE (unreduced)
    ce = F.cross_entropy(
        logits, targets, weight=None, reduction="none", label_smoothing=label_smoothing
    )
    p_t = torch.exp(-ce)  # probability of the true class
    focal_weight = (1.0 - p_t) ** gamma
    # Apply class weights if provided
    if class_weights is not None:
        w = class_weights.to(logits.device)
        per_sample_w = w[targets]
        focal_weight = focal_weight * per_sample_w
    return (focal_weight * ce).mean()


def bce_on_classmaps(
    class_logits: torch.Tensor,
    class_maps: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """Binary cross-entropy on dense class maps (Gaussian supervision)."""
    target = (class_maps > 0).float().to(class_logits.device)
    if pos_weight is None:
        weight = None
    else:
        weight = torch.as_tensor(
            pos_weight, device=class_logits.device, dtype=class_logits.dtype
        )
    return F.binary_cross_entropy_with_logits(
        class_logits, target, pos_weight=weight, reduction="mean"
    )


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    points_xy: List[torch.Tensor],
    labels: List[torch.Tensor],
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss at annotated point locations.

    Pulls embeddings of same-class points together and pushes different-class
    embeddings apart.

    Args:
        embeddings: (B, D, H, W) feature maps.
        points_xy: List of (N_i, 2) tensors with (x, y) coordinates.
        labels: List of (N_i,) tensors with class indices.
        temperature: Temperature scaling for the contrastive similarity.

    Returns:
        Scalar loss tensor.
    """
    B, D, H, W = embeddings.shape
    zs = []
    ys = []
    for b in range(B):
        pts = points_xy[b]
        if pts.numel() == 0:
            continue
        gx = (pts[:, 0].float() / (W - 1)) * 2 - 1
        gy = (pts[:, 1].float() / (H - 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=1).view(1, -1, 1, 2).to(embeddings.device)
        sampled = F.grid_sample(
            embeddings[b : b + 1], grid, mode="bilinear", align_corners=True
        )
        sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
        zs.append(sampled)
        ys.append(labels[b].long())
    if len(zs) == 0:
        return embeddings.sum() * 0
    z = torch.cat(zs, dim=0)
    y = torch.cat(ys, dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    labels_equal = (y[:, None] == y[None, :]).float()
    mask = 1 - torch.eye(z.size(0), device=z.device)
    logits = sim - 1e9 * (1 - mask)
    logsumexp = torch.logsumexp(logits, dim=1)
    pos_mask = labels_equal * mask
    pos_logits = sim - 1e9 * (1 - pos_mask)
    pos_logsumexp = torch.logsumexp(pos_logits, dim=1)
    valid = (pos_mask.sum(dim=1) > 0).float()
    loss = -(pos_logsumexp - logsumexp) * valid
    denom = valid.sum().clamp(min=1.0)
    return loss.sum() / denom


class SoftDiceLoss(nn.Module):
    """Soft Dice loss on (B, C, H, W) logits vs targets in [0, 1].

    Computes Dice per channel aggregated over batch and spatial dims, then averages.
    ``loss = 1 - (2*sum(p*t) + eps) / (sum(p) + sum(t) + eps)``, where
    ``p = sigmoid(logits)``.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W); targets: (B, C, H, W) in [0, 1]
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)  # aggregate over batch and spatial, keep channels
        intersection = (probs * targets).sum(dims)
        denom = probs.sum(dims) + targets.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


class WeightedSoftDiceLoss(nn.Module):
    """Per-channel weighted Soft Dice loss on (B, C, H, W).

    Args:
        smooth: Numerical stability constant.

    Forward args:
        logits: (B, C, H, W) raw logits.
        targets: (B, C, H, W) in [0, 1] (one-hot for multiclass).
        weights: Optional (C,) non-negative weights. If None, uniform.

    Returns:
        Scalar weighted average of per-class dice losses.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dims)
        denom = probs.sum(dims) + targets.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)  # (C,)
        loss_per_class = 1.0 - dice  # (C,)
        if weights is None:
            return loss_per_class.mean()
        w = weights.to(logits.device)
        w = torch.clamp(w, min=0)
        if torch.sum(w) <= 0:
            return loss_per_class.mean()
        return (loss_per_class * w).sum() / (w.sum())


def distribution_consistency_loss(
    class_logits: torch.Tensor,
    tissue_logits: torch.Tensor,
    prior_log_pc_t: torch.Tensor,
    conf_thr: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Encourage per-tissue class distributions to match the prior P(c|t).

    Args:
        class_logits: [B, C, H, W] classification logits.
        tissue_logits: [B, T, Ht, Wt] tissue logits (e.g. from Virchow head at
            224x224).
        prior_log_pc_t: [C, T] log probability matrix.
        conf_thr: Gate low-confidence tissue regions.
        eps: Numerical stability constant.

    Returns:
        Scalar tensor (mean squared difference between predicted and prior
        distributions).
    """
    if prior_log_pc_t is None:
        return class_logits.sum() * 0

    # Softmax over classes/tissues
    p_cls = torch.softmax(class_logits, dim=1)  # [B, C, H, W]
    p_tis = torch.softmax(tissue_logits, dim=1)  # [B, T, Ht, Wt]
    p_tis = F.interpolate(
        p_tis, size=class_logits.shape[-2:], mode="bilinear", align_corners=False
    )

    if conf_thr > 0:
        gate = (p_tis.max(dim=1, keepdim=True).values >= conf_thr).float()
        p_tis = p_tis * gate

    sums = p_tis.sum(dim=(2, 3), keepdim=True).clamp_min(eps)  # [B, T, 1, 1]
    # Aggregate predicted class distribution conditioned on tissue
    agg_cls = torch.einsum("bchw,bthw->bct", p_cls, p_tis)
    agg_cls = agg_cls / sums.squeeze(-1).squeeze(-1).unsqueeze(1)

    prior = prior_log_pc_t.to(class_logits.device).exp()
    prior = prior / prior.sum(dim=0, keepdim=True).clamp_min(eps)
    prior = prior.unsqueeze(0).expand_as(agg_cls)

    loss = ((agg_cls - prior) ** 2).mean()
    return loss
