"""Point-matching metrics for cell detection and classification evaluation."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def greedy_point_match(
    pred_points: np.ndarray,
    pred_labels: np.ndarray,
    gt_points: np.ndarray,
    gt_labels: np.ndarray,
    radius: float = 5.0,
    num_classes: int = 2,
) -> Dict[str, Any]:
    """Greedily match predicted points to ground-truth points and compute metrics.

    Each predicted point is matched to the nearest unmatched ground-truth point
    within ``radius`` pixels.  Both detection metrics (TP, FP, FN) and
    classification accuracy are computed.

    Args:
        pred_points: Predicted point coordinates, shape (N, 2).
        pred_labels: Predicted point labels, shape (N,).
        gt_points: Ground-truth point coordinates, shape (M, 2).
        gt_labels: Ground-truth point labels, shape (M,).
        radius: Maximum matching distance in pixels.
        num_classes: Number of cell classes.

    Returns:
        Dictionary containing:
          - ``tp``, ``fp``, ``fn``: Detection counts.
          - ``cls_correct``, ``cls_total``: Classification accuracy numerator/denominator.
          - ``matches``: List of (pred_idx, gt_idx) pairs.
          - Per-class breakdowns: ``tp_per_class``, ``fn_per_class``,
            ``gt_count_per_class``, ``cls_correct_per_class``,
            ``cls_total_per_class``, ``fp_per_pred_class``,
            ``pred_count_per_class``.
    """
    if len(pred_points) == 0 and len(gt_points) == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "cls_correct": 0,
            "cls_total": 0,
            "matches": [],
            # Per-class containers (zeros)
            "tp_per_class": [0] * num_classes,
            "fn_per_class": [0] * num_classes,
            "gt_count_per_class": [0] * num_classes,
            "cls_correct_per_class": [0] * num_classes,
            "cls_total_per_class": [0] * num_classes,
            "fp_per_pred_class": [0] * num_classes,
            "pred_count_per_class": [0] * num_classes,
        }

    if len(pred_points) == 0:
        # Only GT exists: all are FN, per-class GT counts available
        gt_count: List[int] = [0] * num_classes
        for g in gt_labels:
            if 0 <= int(g) < num_classes:
                gt_count[int(g)] += 1
        return {
            "tp": 0,
            "fp": 0,
            "fn": int(len(gt_points)),
            "cls_correct": 0,
            "cls_total": 0,
            "matches": [],
            "tp_per_class": [0] * num_classes,
            "fn_per_class": gt_count.copy(),
            "gt_count_per_class": gt_count.copy(),
            "cls_correct_per_class": [0] * num_classes,
            "cls_total_per_class": [0] * num_classes,
            "fp_per_pred_class": [0] * num_classes,
            "pred_count_per_class": [0] * num_classes,
        }

    if len(gt_points) == 0:
        # Only predictions exist: all are FP, per-pred-class counts available
        pred_count: List[int] = [0] * num_classes
        for p in pred_labels:
            if 0 <= int(p) < num_classes:
                pred_count[int(p)] += 1
        return {
            "tp": 0,
            "fp": int(len(pred_points)),
            "fn": 0,
            "cls_correct": 0,
            "cls_total": 0,
            "matches": [],
            "tp_per_class": [0] * num_classes,
            "fn_per_class": [0] * num_classes,
            "gt_count_per_class": [0] * num_classes,
            "cls_correct_per_class": [0] * num_classes,
            "cls_total_per_class": [0] * num_classes,
            "fp_per_pred_class": pred_count.copy(),
            "pred_count_per_class": pred_count.copy(),
        }

    # Compute pairwise distance matrix
    pred_points = pred_points.reshape(-1, 1, 2)  # (N, 1, 2)
    gt_points = gt_points.reshape(1, -1, 2)  # (1, M, 2)
    distances: np.ndarray = np.sqrt(
        np.sum((pred_points - gt_points) ** 2, axis=2)
    )  # (N, M)

    # Greedy matching
    matches: List[tuple] = []
    used_gt: set = set()

    for pred_idx in range(len(distances)):
        # Find the nearest unused GT point
        valid_mask = np.ones(distances.shape[1], dtype=bool)
        for gt_idx in used_gt:
            valid_mask[gt_idx] = False

        if not valid_mask.any():
            continue

        valid_distances = distances[pred_idx].copy()
        valid_distances[~valid_mask] = np.inf

        min_dist_idx: int = int(np.argmin(valid_distances))
        min_dist: float = valid_distances[min_dist_idx]

        if min_dist <= radius:
            matches.append((pred_idx, min_dist_idx))
            used_gt.add(min_dist_idx)

    # Compute metrics
    tp: int = len(matches)
    fp: int = int(len(pred_points.reshape(-1, 2))) - tp
    fn: int = int(len(gt_points.reshape(-1, 2))) - tp

    # Per-class aggregations
    gt_count_per_class: List[int] = [0] * num_classes
    for g in gt_labels:
        if 0 <= int(g) < num_classes:
            gt_count_per_class[int(g)] += 1
    tp_per_class: List[int] = [0] * num_classes
    fn_per_class: List[int] = gt_count_per_class.copy()
    cls_correct: int = 0
    cls_total: int = tp
    cls_correct_per_class: List[int] = [0] * num_classes
    cls_total_per_class: List[int] = [0] * num_classes

    matched_pred_indices: set = set()
    for pred_idx, gt_idx in matches:
        matched_pred_indices.add(pred_idx)
        g = int(gt_labels[gt_idx]) if gt_idx is not None else -1
        p = int(pred_labels[pred_idx]) if pred_idx is not None else -1
        if 0 <= g < num_classes:
            tp_per_class[g] += 1
            fn_per_class[g] -= 1
            cls_total_per_class[g] += 1
        if (0 <= p < num_classes) and (0 <= g < num_classes) and (p == g):
            cls_correct += 1
            cls_correct_per_class[g] += 1

    pred_count_per_class: List[int] = [0] * num_classes
    for p in pred_labels:
        if 0 <= int(p) < num_classes:
            pred_count_per_class[int(p)] += 1
    fp_per_pred_class: List[int] = pred_count_per_class.copy()
    # Subtract matched predictions from FP per predicted class
    for pred_idx, gt_idx in matches:
        p = int(pred_labels[pred_idx]) if pred_idx is not None else -1
        if 0 <= p < num_classes:
            fp_per_pred_class[p] -= 1

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "cls_correct": cls_correct,
        "cls_total": cls_total,
        "matches": matches,
        "tp_per_class": tp_per_class,
        "fn_per_class": fn_per_class,
        "gt_count_per_class": gt_count_per_class,
        "cls_correct_per_class": cls_correct_per_class,
        "cls_total_per_class": cls_total_per_class,
        "fp_per_pred_class": fp_per_pred_class,
        "pred_count_per_class": pred_count_per_class,
    }


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-sample metrics into dataset-level metrics.

    Args:
        metrics_list: List of per-sample metric dicts as returned by
            :func:`greedy_point_match`.

    Returns:
        Dictionary with aggregated detection precision/recall/F1, classification
        accuracy, and per-class breakdowns.
    """
    total_tp: int = sum(m["tp"] for m in metrics_list)
    total_fp: int = sum(m["fp"] for m in metrics_list)
    total_fn: int = sum(m["fn"] for m in metrics_list)
    total_cls_correct: int = sum(m["cls_correct"] for m in metrics_list)
    total_cls_total: int = sum(m["cls_total"] for m in metrics_list)

    # Detection metrics
    precision: float = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    recall: float = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
    f1: float = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Classification accuracy
    cls_acc: float = (
        total_cls_correct / total_cls_total if total_cls_total > 0 else 0.0
    )

    # Per-class aggregations
    def _sum_array(key: str) -> List[int] | None:
        if len(metrics_list) == 0 or key not in metrics_list[0]:
            return None
        return [
            sum(m[key][i] for m in metrics_list)
            for i in range(len(metrics_list[0][key]))
        ]

    tp_pc = _sum_array("tp_per_class")
    fn_pc = _sum_array("fn_per_class")
    gt_pc = _sum_array("gt_count_per_class")
    cls_cor_pc = _sum_array("cls_correct_per_class")
    cls_tot_pc = _sum_array("cls_total_per_class")
    fp_pred_pc = _sum_array("fp_per_pred_class")
    pred_cnt_pc = _sum_array("pred_count_per_class")

    cls_acc_pc: List[float] | None = None
    det_recall_pc: List[float] | None = None
    if cls_cor_pc is not None and cls_tot_pc is not None:
        cls_acc_pc = [
            (cls_cor_pc[i] / cls_tot_pc[i]) if cls_tot_pc[i] > 0 else 0.0
            for i in range(len(cls_tot_pc))
        ]
    if tp_pc is not None and gt_pc is not None:
        det_recall_pc = [
            (tp_pc[i] / gt_pc[i]) if gt_pc[i] > 0 else 0.0
            for i in range(len(gt_pc))
        ]

    out: Dict[str, Any] = {
        "overall_precision": precision,
        "overall_recall": recall,
        "overall_f1": f1,
        "cls_acc": cls_acc,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }
    if cls_acc_pc is not None:
        out["cls_acc_per_class"] = cls_acc_pc
        out["cls_correct_per_class"] = cls_cor_pc
        out["cls_total_per_class"] = cls_tot_pc
    if det_recall_pc is not None:
        out["det_recall_per_class"] = det_recall_pc
        out["tp_per_class"] = tp_pc
        out["fn_per_class"] = fn_pc
        out["gt_count_per_class"] = gt_pc
    if fp_pred_pc is not None:
        out["fp_per_pred_class"] = fp_pred_pc
        out["pred_count_per_class"] = pred_cnt_pc

    return out
