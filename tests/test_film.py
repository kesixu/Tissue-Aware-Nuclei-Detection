"""Basic smoke tests for the Spatial-FiLM module."""

import torch
import pytest


def test_spatial_film_identity_at_init():
    """SpatialFiLM should produce near-identity output when freshly initialized (zero-init)."""
    from tand.modules.film import SpatialFiLM

    film = SpatialFiLM(feat_ch=64, tissue_ch=6, hidden=32, limit=0.5)
    x = torch.randn(2, 64, 16, 16)
    s = torch.randn(2, 6, 16, 16)
    out = film(x, s)
    # At initialization, gamma=beta=0 due to zero-init, so output ≈ input
    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-5), "FiLM should be near-identity at initialization"


def test_spatial_film_bounded_modulation():
    """Gamma and beta should be bounded by the limit parameter."""
    from tand.modules.film import SpatialFiLM

    limit = 0.3
    film = SpatialFiLM(feat_ch=32, tissue_ch=6, limit=limit)
    # Force large adapter weights to test bounding
    with torch.no_grad():
        for p in film.adapter.parameters():
            p.fill_(10.0)
    x = torch.ones(1, 32, 8, 8)
    s = torch.ones(1, 6, 8, 8)
    out = film(x, s)
    # Output should be bounded: x * (1 + gamma) + beta where |gamma| <= limit, |beta| <= limit/2
    max_possible = 1.0 * (1 + limit) + limit / 2
    min_possible = 1.0 * (1 - limit) - limit / 2
    assert out.max().item() <= max_possible + 1e-5
    assert out.min().item() >= min_possible - 1e-5


def test_peak_detection():
    """detect_peaks should find local maxima above threshold."""
    from tand.evaluation.peak import detect_peaks

    heat = torch.zeros(32, 32)
    heat[10, 15] = 0.9
    heat[25, 5] = 0.7
    heat[3, 3] = 0.2  # below default threshold
    peaks = detect_peaks(heat, thresh=0.3, nms_radius=3)
    assert len(peaks) == 2
    coords = {(x, y) for x, y, _ in peaks}
    assert (15, 10) in coords
    assert (5, 25) in coords


def test_greedy_point_match():
    """greedy_point_match should correctly match nearby points."""
    import numpy as np
    from tand.evaluation.metrics import greedy_point_match

    pred = np.array([[10.0, 10.0], [20.0, 20.0], [50.0, 50.0]])
    pred_labels = np.array([0, 1, 0])
    gt = np.array([[11.0, 10.0], [20.0, 21.0]])
    gt_labels = np.array([0, 1])
    result = greedy_point_match(pred, pred_labels, gt, gt_labels, radius=5.0, num_classes=2)
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["cls_correct"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
