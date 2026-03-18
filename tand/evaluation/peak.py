"""Peak detection utilities for heatmap-based cell detection."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def nms_heatmap(heat: torch.Tensor, nms_radius: int = 3) -> torch.Tensor:
    """Non-maximum suppression on a single heatmap tensor.

    Retains only pixels that are equal to the local maximum within a square
    window of size ``2 * nms_radius + 1``.

    Args:
        heat: 2D tensor of shape (H, W) with detection scores.
        nms_radius: Radius of the square max-pooling window.

    Returns:
        Boolean tensor of shape (H, W) where ``True`` marks local maxima.
    """
    H, W = heat.shape
    heat4d = heat.view(1, 1, H, W)
    maxpooled = F.max_pool2d(
        heat4d, kernel_size=2 * nms_radius + 1, stride=1, padding=nms_radius
    )
    local_max = heat4d == maxpooled
    return local_max.view(H, W)


def detect_peaks(
    heat: torch.Tensor,
    thresh: float = 0.3,
    nms_radius: int = 3,
    topk: Optional[int] = None,
) -> List[Tuple[int, int, float]]:
    """Detect peaks in a heatmap via NMS and thresholding.

    Args:
        heat: 2D tensor of shape (H, W) with probabilities in [0, 1].
        thresh: Minimum score to consider a peak.
        nms_radius: Radius for non-maximum suppression.
        topk: If set, keep only the top-k scoring peaks.

    Returns:
        List of ``(x, y, score)`` tuples for each detected peak.
    """
    mask_local_max = nms_heatmap(heat, nms_radius)
    mask_thresh = heat >= thresh
    mask = mask_local_max & mask_thresh
    ys, xs = torch.where(mask)
    scores = heat[ys, xs]
    if topk is not None and scores.numel() > topk:
        s, idx = torch.topk(scores, topk)
        xs = xs[idx]
        ys = ys[idx]
        scores = s
    return [
        (int(x.item()), int(y.item()), float(s.item()))
        for x, y, s in zip(xs, ys, scores)
    ]
