#!/usr/bin/env python3
"""Evaluate point-based nuclei predictions with Track2 nuclei metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def load_class_names(meta_path: Path) -> List[str]:
    meta = json.loads(meta_path.read_text())
    names = meta.get("class_names", [])
    if not isinstance(names, list) or not names:
        names = [f"class_{i}" for i in range(10)]
    return names


def load_gt_features(gt_path: Path, class_names: List[str]) -> List[Dict[str, object]]:
    data = json.loads(gt_path.read_text())
    feats = []
    for center in data.get("centers", []):
        cls_idx = int(center.get("cls", 0))
        name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
        feats.append(
            {
                "filename": gt_path.name,
                "category": name,
                "centroid": [float(center.get("x", 0.0)), float(center.get("y", 0.0))],
                "score": 1.0,
            }
        )
    return feats


def load_pred_features(pred_path: Path, class_names: List[str]) -> List[Dict[str, object]]:
    data = json.loads(pred_path.read_text())
    pts = data.get("points", [])
    feats = []
    for pt in pts:
        cls_idx = int(pt.get("cls_id", pt.get("cls", 0)))
        name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
        feats.append(
            {
                "filename": pred_path.name,
                "category": name,
                "centroid": [float(pt.get("x", 0.0)), float(pt.get("y", 0.0))],
                "score": float(pt.get("score", 1.0)),
            }
        )
    return feats


def aggregate_counts(
    per_image_metrics: Iterable[Dict[str, Dict[str, float]]],
    class_names: List[str],
) -> Tuple[Dict[str, Dict[str, float]], float, float, float, float, float, float]:
    per_class_counts: Dict[str, Dict[str, float]] = {name: {"TP": 0.0, "FP": 0.0, "FN": 0.0} for name in class_names}
    for metrics in per_image_metrics:
        for cls_name, values in metrics.items():
            if cls_name in ("micro", "macro") or cls_name not in per_class_counts:
                continue
            per_class_counts[cls_name]["TP"] += float(values.get("TP", 0.0))
            per_class_counts[cls_name]["FP"] += float(values.get("FP", 0.0))
            per_class_counts[cls_name]["FN"] += float(values.get("FN", 0.0))

    micro_tp = sum(per_class_counts[name]["TP"] for name in class_names)
    micro_fp = sum(per_class_counts[name]["FP"] for name in class_names)
    micro_fn = sum(per_class_counts[name]["FN"] for name in class_names)
    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    f1_vals = []
    for name in class_names:
        counts = per_class_counts[name]
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        counts["precision"] = prec
        counts["recall"] = rec
        counts["f1"] = f1
        f1_vals.append(f1)
    macro_f1 = float(np.mean(f1_vals)) if f1_vals else 0.0
    return per_class_counts, micro_prec, micro_rec, micro_f1, macro_f1, micro_tp, micro_fp, micro_fn


def evaluate_fold(
    fold_idx: int,
    pred_dir: Path,
    ann_dir: Path,
    class_names: List[str],
    calc_distance,
    calc_metrics,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    per_image_metrics = []
    per_image_records: Dict[str, Dict[str, Dict[str, float]]] = {}

    for pred_path in sorted(pred_dir.glob("*.json")):
        stem = pred_path.stem
        gt_path = ann_dir / f"{stem}.json"
        if not gt_path.exists():
            continue
        pred_feats = load_pred_features(pred_path, class_names)
        gt_feats = load_gt_features(gt_path, class_names)
        matches = calc_distance(gt_feats, pred_feats)
        metrics = calc_metrics(matches, gt_feats, pred_feats)
        per_image_metrics.append(metrics)
        per_image_records[stem] = metrics

    per_class_counts, micro_prec, micro_rec, micro_f1, macro_f1, micro_tp, micro_fp, micro_fn = aggregate_counts(per_image_metrics, class_names)

    micro_summary = {
        "precision": micro_prec,
        "recall": micro_rec,
        "f1": micro_f1,
        "TP": micro_tp,
        "FP": micro_fp,
        "FN": micro_fn,
    }

    summary = {
        "micro": micro_summary,
        "macro_f1": macro_f1,
        "per_class": per_class_counts,
        "num_samples": len(per_image_records),
    }
    return summary, per_image_records, per_class_counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate point predictions with Track2 nuclei metrics")
    ap.add_argument("--pred-root", type=Path, required=True, help="Root directory containing fold subdirectories")
    ap.add_argument("--puma-root", type=Path, required=True, help="PUMA dataset root")
    ap.add_argument("--track2-root", type=Path, default=Path("PUMA-challenge-eval-track2-main"), help="Track2 evaluator repo path")
    ap.add_argument("--pred-subdir", type=str, default="", help="Optional subdirectory inside each fold with predictions")
    ap.add_argument("--fold-glob", type=str, default="fold*", help="Glob pattern for fold directories")
    ap.add_argument("--out", type=Path, help="Output summary JSON path")
    ap.add_argument("--model-name", type=str, default="model", help="Name of the model (for metadata)")
    ap.add_argument("--save-per-image", action="store_true", help="Include per-image metrics in summary")
    args = ap.parse_args()

    if str(args.track2_root) not in sys.path:
        sys.path.insert(0, str(args.track2_root))
    from eval_nuclei import calculate_centroid_distance, calculate_classification_metrics  # type: ignore

    fold_dirs = sorted([p for p in args.pred_root.glob(args.fold_glob) if p.is_dir()], key=lambda p: p.name)
    if not fold_dirs:
        raise RuntimeError(f"No fold directories matching pattern {args.fold_glob} under {args.pred_root}")

    overall_counts: Dict[str, Dict[str, float]] = {}
    overall_micro_tp = overall_micro_fp = overall_micro_fn = 0.0
    per_fold_summaries = []
    class_name_order: Optional[List[str]] = None

    for fold_dir in fold_dirs:
        name = fold_dir.name
        digits = ''.join(ch for ch in name if ch.isdigit())
        if not digits:
            continue
        fold_idx = int(digits)
        pred_dir = fold_dir / args.pred_subdir if args.pred_subdir else fold_dir
        if not pred_dir.exists():
            print(f"[WARN] Fold {fold_idx}: prediction dir {pred_dir} missing, skipped")
            continue
        ann_dir = args.puma_root / f"fold_{fold_idx}" / "test" / "ann"
        meta_path = args.puma_root / f"fold_{fold_idx}" / "meta.json"
        class_names = load_class_names(meta_path)

        summary, per_image_records, per_class_counts = evaluate_fold(
            fold_idx,
            pred_dir,
            ann_dir,
            class_names,
            calculate_centroid_distance,
            calculate_classification_metrics,
        )

        if class_name_order is None:
            class_name_order = class_names
        else:
            if class_name_order != class_names:
                print(f"[WARN] Fold {fold_idx}: class name list differs from previous folds")

        for cls_name in class_names:
            counts = per_class_counts.get(cls_name, {"TP": 0.0, "FP": 0.0, "FN": 0.0})
            overall_counts.setdefault(cls_name, {"TP": 0.0, "FP": 0.0, "FN": 0.0})
            overall_counts[cls_name]["TP"] += counts.get("TP", 0.0)
            overall_counts[cls_name]["FP"] += counts.get("FP", 0.0)
            overall_counts[cls_name]["FN"] += counts.get("FN", 0.0)
        overall_micro_tp += summary["micro"]["TP"]
        overall_micro_fp += summary["micro"]["FP"]
        overall_micro_fn += summary["micro"]["FN"]

        fold_entry = {
            "fold": fold_idx,
            "micro": summary["micro"],
            "macro_f1": summary["macro_f1"],
            "per_class": summary["per_class"],
            "num_samples": summary["num_samples"],
        }
        if args.save_per_image:
            fold_entry["per_image"] = per_image_records
        per_fold_summaries.append(fold_entry)

    sorted_class_names = class_name_order or sorted(overall_counts.keys())
    overall_per_class = {}
    f1_vals = []
    for cls_name in sorted_class_names:
        counts = overall_counts[cls_name]
        tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        counts.update({"precision": prec, "recall": rec, "f1": f1})
        overall_per_class[cls_name] = counts
        f1_vals.append(f1)

    overall_micro_prec = overall_micro_tp / (overall_micro_tp + overall_micro_fp) if (overall_micro_tp + overall_micro_fp) > 0 else 0.0
    overall_micro_rec = overall_micro_tp / (overall_micro_tp + overall_micro_fn) if (overall_micro_tp + overall_micro_fn) > 0 else 0.0
    overall_micro_f1 = 2 * overall_micro_prec * overall_micro_rec / (overall_micro_prec + overall_micro_rec) if (overall_micro_prec + overall_micro_rec) > 0 else 0.0
    overall_macro_f1 = float(np.mean(f1_vals)) if f1_vals else 0.0

    summary_payload = {
        "model": args.model_name,
        "evaluation": {
            "match_radius": 15.0,
            "evaluator": "eval_nuclei.py",
        },
        "per_fold": sorted(per_fold_summaries, key=lambda x: x["fold"]),
        "overall": {
            "micro": {
                "precision": overall_micro_prec,
                "recall": overall_micro_rec,
                "f1": overall_micro_f1,
                "TP": overall_micro_tp,
                "FP": overall_micro_fp,
                "FN": overall_micro_fn,
            },
            "macro_f1": overall_macro_f1,
            "per_class": overall_per_class,
        },
    }

    out_path = args.out or (args.pred_root / "track2_eval_summary.json")

    # Convert numpy types to native Python for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    out_path.write_text(json.dumps(_convert(summary_payload), indent=2), encoding="utf-8")
    print(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
