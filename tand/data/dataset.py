"""Dataset helpers for point-level cell detection and classification."""

from __future__ import annotations

import json
import logging
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFile
import torch
from torch.utils.data import Dataset

# Allow loading truncated images to avoid dataloader crashes on partially written PNGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gaussian heatmap generation
# ---------------------------------------------------------------------------


def gaussian_2d(
    height: int, width: int, cx: int, cy: int, sigma: float = 2.0
) -> np.ndarray:
    """Return a 2D Gaussian centered at (cx, cy).

    Args:
        height: Height of the output array.
        width: Width of the output array.
        cx: X coordinate of the Gaussian center.
        cy: Y coordinate of the Gaussian center.
        sigma: Standard deviation of the Gaussian.

    Returns:
        A float64 array of shape (height, width) with values in [0, 1].
    """
    y = np.arange(0, height).reshape(-1, 1)
    x = np.arange(0, width).reshape(1, -1)
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))


def centers_to_heatmap(
    height: int,
    width: int,
    centers: List[Dict[str, int]],
    sigma: float = 2.0,
) -> np.ndarray:
    """Render detections as a max-pooled Gaussian heatmap.

    Each center dict must contain ``"x"`` and ``"y"`` keys.  The output is the
    element-wise maximum over individual Gaussians, producing a single-channel
    probability-like map in [0, 1].

    Args:
        height: Height of the output heatmap.
        width: Width of the output heatmap.
        centers: List of dicts with ``"x"`` and ``"y"`` keys.
        sigma: Standard deviation of each Gaussian.

    Returns:
        A float32 array of shape (height, width).
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    for center in centers:
        heatmap = np.maximum(
            heatmap, gaussian_2d(height, width, center["x"], center["y"], sigma)
        )
    return heatmap


# ---------------------------------------------------------------------------
# Data augmentation for point-supervised cell detection
# ---------------------------------------------------------------------------


def augment_sample(
    image_arr: np.ndarray,
    heatmap: np.ndarray,
    points: np.ndarray,
    H: int,
    W: int,
    tissue_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, Optional[np.ndarray]]:
    """Apply random geometric and color augmentation to a sample.

    All inputs are numpy arrays.  ``points`` is (N, 2) with columns (x, y).
    Returns augmented copies; originals are not modified.

    Geometric transforms applied (each with p=0.5):
      - Horizontal flip
      - Vertical flip
      - Random 90-degree rotation (0/90/180/270)

    Color transforms:
      - Brightness jitter  +/-10%
      - Contrast jitter    +/-10%
    """
    image_arr = image_arr.copy()
    heatmap = heatmap.copy()
    points = points.copy() if len(points) > 0 else points
    tissue_mask = tissue_mask.copy() if tissue_mask is not None else None

    # --- Horizontal flip ---
    if random.random() < 0.5:
        image_arr = image_arr[:, ::-1, :].copy()
        heatmap = heatmap[:, ::-1].copy()
        if len(points) > 0:
            points[:, 0] = W - 1 - points[:, 0]
        if tissue_mask is not None:
            tissue_mask = tissue_mask[:, ::-1].copy()

    # --- Vertical flip ---
    if random.random() < 0.5:
        image_arr = image_arr[::-1, :, :].copy()
        heatmap = heatmap[::-1, :].copy()
        if len(points) > 0:
            points[:, 1] = H - 1 - points[:, 1]
        if tissue_mask is not None:
            tissue_mask = tissue_mask[::-1, :].copy()

    # --- Random 90-degree rotation ---
    k = random.randint(0, 3)  # 0=no rotation, 1=90, 2=180, 3=270
    if k > 0:
        image_arr = np.rot90(image_arr, k, axes=(0, 1)).copy()
        heatmap = np.rot90(heatmap, k, axes=(0, 1)).copy()
        if tissue_mask is not None:
            tissue_mask = np.rot90(tissue_mask, k, axes=(0, 1)).copy()
        if len(points) > 0:
            for _ in range(k):
                # 90-degree counter-clockwise: (x, y) -> (y, W-1-x)
                # np.rot90 with k=1 rotates CCW, so new coords:
                new_x = points[:, 1].copy()
                new_y = W - 1 - points[:, 0]
                points[:, 0] = new_x
                points[:, 1] = new_y
                # After rotation, H and W swap
                H, W = W, H
        else:
            if k % 2 == 1:
                H, W = W, H

    # --- Color jitter (only on image, not heatmap/points) ---
    # Brightness
    factor = 1.0 + random.uniform(-0.1, 0.1)
    image_arr = np.clip(image_arr * factor, 0.0, 1.0)
    # Contrast
    mean = image_arr.mean()
    factor = 1.0 + random.uniform(-0.1, 0.1)
    image_arr = np.clip((image_arr - mean) * factor + mean, 0.0, 1.0)

    return image_arr, heatmap, points, H, W, tissue_mask


# ---------------------------------------------------------------------------
# Oversampling weights
# ---------------------------------------------------------------------------


def compute_sample_weights(
    dataset: Dataset, rare_boost: float = 3.0
) -> Tuple[List[float], Counter]:
    """Compute per-sample weights for ``WeightedRandomSampler``.

    Scans all annotations and computes per-class frequency.  Samples containing
    cells from the bottom-30% frequency classes get ``rare_boost`` weight;
    others get weight 1.0.

    Args:
        dataset: A ``PatchesDataset`` or ``ShapesPointDataset`` instance.
        rare_boost: Weight assigned to samples with rare-class cells.

    Returns:
        weights: List of floats, one per sample.
        class_counts: Counter of total cells per class.
    """
    class_counts: Counter = Counter()
    sample_classes: List[set] = []

    for idx in range(len(dataset)):
        stem = dataset.items[idx]
        ann_path = dataset.ann_dir / f"{stem}.json"
        ann = json.loads(ann_path.read_text())
        centers = ann.get("centers", [])
        classes_in_sample: set = set()
        for c in centers:
            cls = c.get("cls", -1)
            class_counts[cls] += 1
            classes_in_sample.add(cls)
        sample_classes.append(classes_in_sample)

    if not class_counts:
        return [1.0] * len(dataset), class_counts

    # Identify rare classes: bottom 30% by frequency
    sorted_counts = sorted(class_counts.values())
    threshold_idx = max(0, int(len(sorted_counts) * 0.3) - 1)
    rare_threshold = sorted_counts[threshold_idx]
    rare_classes = {cls for cls, cnt in class_counts.items() if cnt <= rare_threshold}

    weights: List[float] = []
    for classes_in_sample in sample_classes:
        if classes_in_sample & rare_classes:
            weights.append(rare_boost)
        else:
            weights.append(1.0)

    return weights, class_counts


# ---------------------------------------------------------------------------
# PatchesDataset — loads pre-cut 224x224 patches
# ---------------------------------------------------------------------------


class PatchesDataset(Dataset):
    """Dataset for pre-cut 224x224 patches with point annotations.

    Loads images and JSON annotation files from a directory tree::

        root_dir/
          {split}/
            images/{stem}.png
            ann/{stem}.json
            tissue_masks/{stem}.png   (optional)
            heatmaps/{stem}.npy       (optional cache)
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        sigma: float = 2.0,
        with_tissue_mask: bool = True,
        load_cached_heatmaps: bool = True,
        heatmap_subdir: Optional[str] = None,
        save_missing_heatmaps: bool = False,
        augment: bool = False,
    ):
        """
        Args:
            root_dir: Root directory of pre-processed patches
                (e.g. ``puma_coco_folds_224x224_patches``).
            split: ``"train"`` or ``"test"``.
            sigma: Gaussian sigma for heatmap generation.
            with_tissue_mask: Whether to load tissue mask labels.
            load_cached_heatmaps: Try to load pre-computed heatmaps from disk.
            heatmap_subdir: Explicit subdirectory name for cached heatmaps.
            save_missing_heatmaps: Save heatmaps to disk when computed on the fly.
            augment: Enable data augmentation (only applied when split is
                ``"train"``).
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.sigma = sigma
        self.augment = augment and (split == "train")
        self.img_dir = self.root_dir / split / "images"
        self.ann_dir = self.root_dir / split / "ann"

        # Collect all patch stems
        self.items = sorted([p.stem for p in self.ann_dir.glob("*.json")])

        # Read metadata
        meta_path = self.root_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.class_names = meta["class_names"]
            self.num_classes = len(self.class_names)
        else:
            # Fallback defaults
            self.class_names = ["background", "cell"]
            self.num_classes = 2

        logger.info("Loaded %s dataset: %d patches", split, len(self.items))

        # Optional tissue masks directory
        self.tissue_dir = (
            (self.root_dir / split / "tissue_masks") if with_tissue_mask else None
        )

        # Heatmap cache directory
        cand_subdirs: List[str] = []
        if heatmap_subdir:
            cand_subdirs.append(heatmap_subdir)
        cand_subdirs.append("heatmaps")
        seen: set = set()
        self._heatmap_candidates: List[str] = []
        for sub in cand_subdirs:
            if sub and sub not in seen:
                self._heatmap_candidates.append(sub)
                seen.add(sub)
        if not self._heatmap_candidates:
            self._heatmap_candidates = ["heatmaps"]
        self._heatmap_save_dir = self.root_dir / split / self._heatmap_candidates[0]
        self._heatmap_load_dir: Optional[Path] = None
        self._load_cached_heatmaps = False
        if load_cached_heatmaps:
            for sub in self._heatmap_candidates:
                path = self.root_dir / split / sub
                if path.exists():
                    self._heatmap_load_dir = path
                    self._load_cached_heatmaps = True
                    break
        self._save_missing_heatmaps = bool(save_missing_heatmaps)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem = self.items[idx]

        # Load image
        img = Image.open(self.img_dir / f"{stem}.png").convert("RGB")

        # Load annotation
        ann = json.loads((self.ann_dir / f"{stem}.json").read_text())
        centers = ann["centers"]

        # Convert to numpy array
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        H, W = arr.shape[:2]

        # Generate heatmap (try cached first)
        hm: Optional[np.ndarray] = None
        if self._load_cached_heatmaps:
            heat_path = self._heatmap_load_dir / f"{stem}.npy"
            if heat_path.exists():
                try:
                    hm = np.load(heat_path).astype(np.float32, copy=False)
                    if hm.shape != (H, W):
                        hm = None
                except Exception:
                    hm = None
        if hm is None:
            hm = centers_to_heatmap(H, W, centers, sigma=self.sigma)
            if self._save_missing_heatmaps:
                try:
                    self._heatmap_save_dir.mkdir(parents=True, exist_ok=True)
                    np.save(
                        self._heatmap_save_dir / f"{stem}.npy",
                        hm.astype(np.float16),
                    )
                except Exception:
                    pass

        # Process points and labels
        pts = (
            np.array([[c["x"], c["y"]] for c in centers], dtype=np.float32)
            if len(centers) > 0
            else np.zeros((0, 2), dtype=np.float32)
        )
        labs = (
            np.array([c["cls"] for c in centers], dtype=np.int64)
            if len(centers) > 0
            else np.zeros((0,), dtype=np.int64)
        )

        # Read optional tissue mask (integer label map, HxW)
        tm_arr: Optional[np.ndarray] = None
        if self.tissue_dir is not None:
            mask_path_png = self.tissue_dir / f"{stem}.png"
            mask_path_npy = self.tissue_dir / f"{stem}.npy"
            if mask_path_png.exists():
                try:
                    m = Image.open(mask_path_png)
                    tm_arr = np.array(m)
                except Exception:
                    tm_arr = None
            elif mask_path_npy.exists():
                try:
                    tm_arr = np.load(mask_path_npy)
                except Exception:
                    tm_arr = None
            if tm_arr is not None:
                if tm_arr.ndim == 3:
                    try:
                        tm_arr = np.argmax(tm_arr, axis=2)
                    except Exception:
                        tm_arr = tm_arr[..., 0]
                if tm_arr.shape[0] != H or tm_arr.shape[1] != W:
                    m_img = Image.fromarray(tm_arr.astype(np.int32, copy=False))
                    m_img = m_img.resize((W, H), resample=Image.NEAREST)
                    tm_arr = np.array(m_img)
                tm_arr = tm_arr.astype(np.int64)

        # --- Apply augmentation (training only) ---
        if self.augment:
            arr, hm, pts, H, W, tm_arr = augment_sample(arr, hm, pts, H, W, tm_arr)

        tissue_mask_t = torch.from_numpy(tm_arr).long() if tm_arr is not None else None

        # Convert to tensors
        img_t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)  # (3,H,W)
        hm_t = torch.from_numpy(np.ascontiguousarray(hm)).unsqueeze(0)  # (1,H,W)
        pts_t = (
            torch.from_numpy(np.ascontiguousarray(pts))
            if len(pts) > 0
            else torch.zeros((0, 2), dtype=torch.float32)
        )
        labs_t = torch.from_numpy(labs)  # (N,)

        sample: Dict[str, Any] = {
            "image": img_t,
            "heatmap": hm_t,
            "points": pts_t,
            "labels": labs_t,
            "stem": stem,
            # Original image info (for downstream patch reassembly if needed)
            "original_image": ann.get("original_image", stem),
            "window_index": ann.get("window_index", 0),
            "window_position": ann.get("window_position", {"y": 0, "x": 0}),
        }
        if tissue_mask_t is not None:
            sample["tissue_mask"] = tissue_mask_t
        return sample


def collate_patches(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching patches.

    Stacks images and heatmaps into tensors; keeps variable-length point
    annotations as lists.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    heatmaps = torch.stack([b["heatmap"] for b in batch], dim=0)
    points = [b["points"] for b in batch]
    labels = [b["labels"] for b in batch]
    stems = [b["stem"] for b in batch]
    original_images = [b["original_image"] for b in batch]
    window_indices = [b["window_index"] for b in batch]
    window_positions = [b["window_position"] for b in batch]

    out: Dict[str, Any] = {
        "image": images,
        "heatmap": heatmaps,
        "points": points,
        "labels": labels,
        "stem": stems,
        "original_image": original_images,
        "window_index": window_indices,
        "window_position": window_positions,
    }
    # Optional tissue masks (keep as list to preserve optionality)
    if all("tissue_mask" in b for b in batch):
        out["tissue_mask"] = [b["tissue_mask"] for b in batch]
    return out


# ---------------------------------------------------------------------------
# ShapesPointDataset — full-size images with optional resize
# ---------------------------------------------------------------------------


class ShapesPointDataset(Dataset):
    """Dataset for full-size images with point annotations.

    Supports optional resizing, tissue masks from an external directory,
    heatmap caching, and data augmentation.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        img_size: int = 256,
        sigma: float = 2.0,
        resize: Optional[int] = None,
        with_tissue_mask: bool = False,
        tissue_mask_root: Optional[str] = None,
        load_cached_heatmaps: bool = True,
        heatmap_subdir: Optional[str] = None,
        save_missing_heatmaps: bool = False,
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.sigma = sigma
        self.resize = resize
        self.augment = augment and (split == "train")
        self.img_dir = self.root_dir / split / "images"
        self.ann_dir = self.root_dir / split / "ann"
        self.items = sorted([p.stem for p in self.ann_dir.glob("*.json")])
        self.with_tissue_mask = bool(with_tissue_mask)
        self.tissue_mask_root = (
            Path(tissue_mask_root) if (with_tissue_mask and tissue_mask_root) else None
        )
        meta_path = self.root_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.class_names = meta.get("class_names", ["background", "cell"])
            self.num_classes = len(self.class_names)
        else:
            # Fallback defaults
            self.class_names = ["background", "cell"]
            self.num_classes = 2

        # Heatmap cache handling
        cand_subdirs: List[str] = []
        if heatmap_subdir:
            cand_subdirs.append(heatmap_subdir)
        if resize:
            cand_subdirs.append(f"heatmaps_{int(resize)}")
        cand_subdirs.append("heatmaps")
        seen: set = set()
        self._heatmap_candidates: List[str] = []
        for sub in cand_subdirs:
            if sub and sub not in seen:
                self._heatmap_candidates.append(sub)
                seen.add(sub)
        if not self._heatmap_candidates:
            self._heatmap_candidates = ["heatmaps"]
        self._heatmap_save_dir = self.root_dir / split / self._heatmap_candidates[0]
        self._heatmap_load_dir: Optional[Path] = None
        self._load_cached_heatmaps = False
        if load_cached_heatmaps:
            for sub in self._heatmap_candidates:
                path = self.root_dir / split / sub
                if path.exists():
                    self._heatmap_load_dir = path
                    self._load_cached_heatmaps = True
                    break
        self._save_missing_heatmaps = bool(save_missing_heatmaps)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem = self.items[idx]
        img = Image.open(self.img_dir / f"{stem}.png").convert("RGB")
        ann = json.loads((self.ann_dir / f"{stem}.json").read_text())
        centers = ann["centers"]

        # Original size (PIL size = (W, H))
        W0, H0 = img.size

        # Optional external tissue mask (full-size). Support .png or .npy
        tissue_mask_t = None
        m_arr0: Optional[np.ndarray] = None
        if self.with_tissue_mask and self.tissue_mask_root is not None:
            mpath_png = self.tissue_mask_root / f"{stem}.png"
            mpath_npy = self.tissue_mask_root / f"{stem}.npy"
            if mpath_png.exists():
                try:
                    m = Image.open(mpath_png)
                    m_arr0 = np.array(m)
                except Exception:
                    m_arr0 = None
            elif mpath_npy.exists():
                try:
                    m_arr0 = np.load(mpath_npy)
                except Exception:
                    m_arr0 = None
            # If mask loaded with unexpected shape, reduce to single channel
            if m_arr0 is not None and m_arr0.ndim == 3:
                try:
                    # Prefer argmax over channels if one-hot
                    m_arr0 = np.argmax(m_arr0, axis=2)
                except Exception:
                    # Fall back to first channel
                    m_arr0 = m_arr0[..., 0]
            if m_arr0 is not None:
                # Ensure integer array for resizing
                m_arr0 = m_arr0.astype(np.int32, copy=False)

        # Optional resize keeping square target
        if (
            self.resize is not None
            and int(self.resize) > 0
            and (H0 != self.resize or W0 != self.resize)
        ):
            scale_x = self.resize / float(W0)
            scale_y = self.resize / float(H0)
            # Scale centers
            if len(centers) > 0:
                centers = [
                    {
                        "x": int(round(c["x"] * scale_x)),
                        "y": int(round(c["y"] * scale_y)),
                        "cls": c["cls"],
                    }
                    for c in centers
                ]
            img = img.resize((self.resize, self.resize), resample=Image.BILINEAR)

        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        H, W = arr.shape[:2]

        hm: Optional[np.ndarray] = None
        if self._load_cached_heatmaps and self._heatmap_load_dir is not None:
            candidate = self._heatmap_load_dir / f"{stem}.npy"
            if candidate.exists():
                try:
                    hm_loaded = np.load(candidate).astype(np.float32, copy=False)
                    if hm_loaded.shape == (H, W):
                        hm = hm_loaded
                except Exception:
                    hm = None
        if hm is None:
            hm = centers_to_heatmap(H, W, centers, sigma=self.sigma)
            if self._save_missing_heatmaps:
                try:
                    self._heatmap_save_dir.mkdir(parents=True, exist_ok=True)
                    np.save(
                        self._heatmap_save_dir / f"{stem}.npy",
                        hm.astype(np.float16),
                    )
                except Exception:
                    pass

        pts = (
            np.array([[c["x"], c["y"]] for c in centers], dtype=np.float32)
            if len(centers) > 0
            else np.zeros((0, 2), dtype=np.float32)
        )
        labs = (
            np.array([c["cls"] for c in centers], dtype=np.int64)
            if len(centers) > 0
            else np.zeros((0,), dtype=np.int64)
        )

        # Finalize tissue mask to current HxW
        tm_arr: Optional[np.ndarray] = None
        if m_arr0 is not None:
            m_img = Image.fromarray(m_arr0)
            if m_img.size != (W, H):
                m_img = m_img.resize((W, H), resample=Image.NEAREST)
            tm_arr = np.array(m_img, dtype=np.int64)

        # --- Apply augmentation (training only) ---
        if self.augment:
            arr, hm, pts, H, W, tm_arr = augment_sample(arr, hm, pts, H, W, tm_arr)

        tissue_mask_t = torch.from_numpy(tm_arr).long() if tm_arr is not None else None

        img_t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
        hm_t = torch.from_numpy(np.ascontiguousarray(hm)).unsqueeze(0)
        pts_t = (
            torch.from_numpy(np.ascontiguousarray(pts))
            if len(pts) > 0
            else torch.zeros((0, 2), dtype=torch.float32)
        )
        labs_t = torch.from_numpy(labs)

        sample: Dict[str, Any] = {
            "image": img_t,
            "heatmap": hm_t,
            "points": pts_t,
            "labels": labs_t,
            "stem": stem,
        }
        if tissue_mask_t is not None:
            sample["tissue_mask"] = tissue_mask_t
        return sample


def collate_point_batches(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function that keeps variable-length point annotations."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    heatmaps = torch.stack([b["heatmap"] for b in batch], dim=0)
    points = [b["points"] for b in batch]
    labels = [b["labels"] for b in batch]
    stems = [b["stem"] for b in batch]
    out: Dict[str, Any] = {
        "image": images,
        "heatmap": heatmaps,
        "points": points,
        "labels": labels,
        "stem": stems,
    }
    if all("tissue_mask" in b for b in batch):
        out["tissue_mask"] = [b["tissue_mask"] for b in batch]
    return out
