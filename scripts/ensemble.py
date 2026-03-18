#!/usr/bin/env python3
"""Ensemble predictions from multiple enhanced inference runs.

Averages class logits across models before argmax, optionally applying tissue prior.

Usage:
    python ensemble.py \\
        --model_roots outputs/inference_enhanced_v4b outputs/inference_enhanced_v4c outputs/inference_enhanced_v4d \\
        --output_root outputs/inference_ensemble_v4bcd \\
        --prior_path outputs/Analysis/log_pc_given_t.npy \\
        --alphas 0 0.5 1.0 2.0 \\
        --eval --puma_root puma_coco_folds
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_enhanced_predictions(model_roots: List[Path], fold: int) -> Dict[str, List[Dict]]:
    """Load enhanced predictions from all models for one fold.
    Returns {stem: [model1_points, model2_points, ...]}"""
    all_preds = {}
    for root in model_roots:
        enh_dir = root / f"fold_{fold}" / "enhanced"
        if not enh_dir.exists():
            print(f"  [WARN] {enh_dir} missing")
            continue
        for jf in sorted(enh_dir.glob("*.json")):
            stem = jf.stem
            data = json.loads(jf.read_text())
            points = data.get("points", [])
            all_preds.setdefault(stem, []).append(points)
    return all_preds


def match_points_across_models(model_points_list: List[List[Dict]], max_dist: float = 3.0):
    """Match points across models by spatial proximity.
    Uses model 0 (primary) as the anchor, matches others to it.
    Returns list of matched point groups."""
    if not model_points_list or not model_points_list[0]:
        return []

    primary = model_points_list[0]
    n_models = len(model_points_list)
    matched_groups = []

    for p0 in primary:
        group = [p0]
        x0, y0 = p0["x"], p0["y"]

        for mi in range(1, n_models):
            best_pt = None
            best_dist = max_dist + 1
            for pt in model_points_list[mi]:
                d = ((pt["x"] - x0) ** 2 + (pt["y"] - y0) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_pt = pt
            group.append(best_pt if best_dist <= max_dist else None)

        matched_groups.append(group)
    return matched_groups


def ensemble_point(group: List[Optional[Dict]], log_pc_given_t=None, alpha=0.0):
    """Ensemble logits from matched points, optionally with tissue prior."""
    logits_list = []
    tissue_probs_list = []

    for pt in group:
        if pt is None or pt.get("class_logits") is None:
            continue
        logits_list.append(np.array(pt["class_logits"], dtype=np.float64))
        if pt.get("tissue_probs") is not None:
            tissue_probs_list.append(np.array(pt["tissue_probs"], dtype=np.float64))

    if not logits_list:
        p0 = group[0]
        return p0["cls_id"] if p0 else 0

    # Average logits
    avg_logits = np.mean(logits_list, axis=0)

    # Add tissue prior if requested
    if alpha > 0 and log_pc_given_t is not None and tissue_probs_list:
        avg_tissue = np.mean(tissue_probs_list, axis=0)
        bias = log_pc_given_t @ avg_tissue
        avg_logits = avg_logits + alpha * bias

    return int(np.argmax(avg_logits))


def process_fold(model_roots, fold, output_dir, log_pc_given_t=None, alpha=0.0):
    """Process one fold: ensemble + optional tissue prior."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_preds = load_enhanced_predictions(model_roots, fold)
    total, changed = 0, 0

    for stem, model_points_list in sorted(all_preds.items()):
        if len(model_points_list) < 2:
            # Only one model has predictions -- use as-is
            pts = model_points_list[0]
            out_pts = []
            for pt in pts:
                if log_pc_given_t is not None and alpha > 0 and pt.get("class_logits"):
                    cls_id = ensemble_point([pt], log_pc_given_t, alpha)
                else:
                    cls_id = pt["cls_id"]
                total += 1
                if cls_id != pt["cls_id"]:
                    changed += 1
                out_pts.append({"x": pt["x"], "y": pt["y"], "cls": cls_id,
                                "cls_id": cls_id, "score": pt["score"]})
        else:
            matched = match_points_across_models(model_points_list, max_dist=5.0)
            out_pts = []
            for group in matched:
                p0 = group[0]
                cls_id = ensemble_point(group, log_pc_given_t, alpha)
                total += 1
                if cls_id != p0["cls_id"]:
                    changed += 1
                out_pts.append({"x": p0["x"], "y": p0["y"], "cls": cls_id,
                                "cls_id": cls_id, "score": p0["score"]})

        (output_dir / f"{stem}.json").write_text(json.dumps({"points": out_pts}, indent=2))

    return total, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_roots", type=Path, nargs="+", required=True)
    ap.add_argument("--output_root", type=Path, required=True)
    ap.add_argument("--prior_path", type=Path, default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0])
    ap.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--puma_root", type=Path, default=Path("puma_coco_folds"))
    ap.add_argument("--evaluator_path", type=str, default=None,
                    help="Path to evaluate.py script (default: auto-detect from repo root)")
    args = ap.parse_args()

    log_pc_given_t = None
    if args.prior_path and args.prior_path.exists():
        log_pc_given_t = np.load(args.prior_path)
        print(f"Loaded prior: {log_pc_given_t.shape}")

    print(f"Models: {[str(r) for r in args.model_roots]}")

    # Resolve evaluator path
    if args.evaluator_path is not None:
        evaluator_path = args.evaluator_path
    else:
        repo_root = Path(__file__).resolve().parents[1]
        evaluator_path = str(repo_root / "scripts" / "evaluate.py")

    for alpha in args.alphas:
        tag = f"alpha{alpha}"
        print(f"\n=== Ensemble + {tag} ===")
        out_root = args.output_root / tag

        for fold in args.folds:
            fold_out = out_root / f"fold_{fold}"
            total, changed = process_fold(args.model_roots, fold, fold_out,
                                          log_pc_given_t, alpha)
            pct = 100.0 * changed / total if total > 0 else 0
            print(f"  fold {fold}: {total} points, {changed} changed ({pct:.1f}%)")

        if args.eval:
            eval_out = args.output_root / f"track2_ensemble_{tag}.json"
            cmd = [
                sys.executable, evaluator_path,
                "--pred-root", str(out_root),
                "--puma-root", str(args.puma_root),
                "--track2-root", "PUMA-challenge-eval-track2-main",
                "--out", str(eval_out),
                "--model-name", f"ensemble_{tag}",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True,
                               cwd=str(Path(__file__).resolve().parent))
            if r.returncode == 0:
                d = json.loads(eval_out.read_text())
                mf1 = d["overall"]["macro_f1"]
                print(f"  >>> macro-F1 = {mf1:.4f}")
            else:
                print(f"  [ERROR] {r.stderr[-300:]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
