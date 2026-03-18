#!/usr/bin/env python3
"""Enhanced 5-fold inference saving per-point raw logits and tissue probabilities."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to Python path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tand.data.dataset import ShapesPointDataset, collate_point_batches
from tand.models.efficientunet import get_efficientunet_b0
from tand.models.fused_pointcls_unet import VirchowFusedNet
from tand.evaluation.peak import detect_peaks


def _load_model(weights_path: Path, num_classes: int, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
    is_fused = any(k.startswith("det.") for k in state_dict.keys())
    if is_fused:
        model = VirchowFusedNet(num_classes=num_classes, num_tissue=6, film_limit=0.5, pretrained=False)
        model.load_state_dict(state_dict)
    else:
        model = get_efficientunet_b0(num_classes=num_classes, pretrained=False)
        model.load_state_dict(state_dict)
    return model.to(device).eval(), is_fused


def _predict_batch_enhanced(model, images, stems, threshold, nms_radius, device, is_fused):
    sigmoid = nn.Sigmoid()
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)

    heatmaps = sigmoid(outputs["heatmap_logits"]).squeeze(1)
    class_logits_map = outputs.get("class_logits")

    tissue_probs_map = None
    if is_fused:
        tissue_logits_224 = outputs.get("tissue_logits_224")
        if tissue_logits_224 is not None:
            tissue_probs_map = F.softmax(tissue_logits_224, dim=1)
            if tissue_probs_map.shape[-2:] != heatmaps.shape[-2:]:
                tissue_probs_map = F.interpolate(
                    tissue_probs_map, size=heatmaps.shape[-2:],
                    mode="bilinear", align_corners=False,
                )

    standard_batch = []
    enhanced_batch = []

    for idx, stem in enumerate(stems):
        peaks = detect_peaks(heatmaps[idx], thresh=threshold, nms_radius=nms_radius)
        std_preds = []
        enh_points = []

        for x, y, det_score in peaks:
            if class_logits_map is not None:
                raw_logits = class_logits_map[idx, :, y, x]
                cls_id = int(torch.argmax(raw_logits).item())
                cls_score = float(torch.softmax(raw_logits, dim=0)[cls_id].item())
                class_logits_list = raw_logits.cpu().float().tolist()
            else:
                cls_id, cls_score, class_logits_list = -1, 0.0, None

            tissue_probs_list = None
            if tissue_probs_map is not None:
                tissue_probs_list = tissue_probs_map[idx, :, y, x].cpu().float().tolist()

            std_preds.append({
                "x": int(x), "y": int(y), "cls": cls_id, "cls_id": cls_id,
                "score": float(det_score), "cls_score": cls_score,
            })
            enh_points.append({
                "x": int(x), "y": int(y), "cls_id": cls_id,
                "score": float(det_score),
                "class_logits": class_logits_list,
                "tissue_probs": tissue_probs_list,
            })

        standard_batch.append({"stem": stem, "predictions": std_preds})
        enhanced_batch.append({"stem": stem, "points": enh_points})

    return standard_batch, enhanced_batch


def run_fold(data_root, weights_path, output_dir, split="test", batch_size=4,
             num_workers=0, det_threshold=0.35, nms_radius=3, device_str="cuda"):
    output_dir = Path(output_dir)
    enhanced_dir = output_dir / "enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    enhanced_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    dataset = ShapesPointDataset(data_root, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_point_batches)
    model, is_fused = _load_model(weights_path, dataset.num_classes, device)
    print(f"  Loaded {'VirchowFusedNet' if is_fused else 'EfficientUNet'} from {weights_path}")

    for batch in dataloader:
        std_batch, enh_batch = _predict_batch_enhanced(
            model, batch["image"], batch["stem"],
            det_threshold, nms_radius, device, is_fused,
        )
        for std, enh in zip(std_batch, enh_batch):
            stem = std["stem"]
            std_pts = [{"x": p["x"], "y": p["y"], "cls": p["cls"],
                        "cls_id": p["cls_id"], "score": p["score"]}
                       for p in std["predictions"]]
            (output_dir / f"{stem}.json").write_text(json.dumps({"points": std_pts}, indent=2))
            (enhanced_dir / f"{stem}.json").write_text(json.dumps(enh, indent=2))

    return len(list(output_dir.glob("*.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--weights_root", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/inference_enhanced"))
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--det_thresh", type=float, default=0.35)
    ap.add_argument("--nms_radius", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    for fold in args.folds:
        wt = args.weights_root / f"fold_{fold}" / "best.pt"
        if not wt.exists():
            print(f"[WARN] Skipping fold {fold}: {wt} not found")
            continue
        fold_data = args.data_root / f"fold_{fold}"
        fold_out = args.output_dir / f"fold_{fold}"
        print(f"\n[fold {fold}] data={fold_data}  weights={wt}")
        n = run_fold(fold_data, wt, fold_out, batch_size=args.batch_size,
                     det_threshold=args.det_thresh, nms_radius=args.nms_radius,
                     device_str=args.device)
        print(f"  Done -- {n} images")
    print(f"\nAll folds complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
