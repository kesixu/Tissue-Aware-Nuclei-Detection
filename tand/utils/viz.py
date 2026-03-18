"""Visualization utilities for cell detection results."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def to_uint8_rgb(img_tensor) -> Image.Image:
    """Convert a tensor [3, H, W] in [0, 1] to a PIL RGB image."""
    if hasattr(img_tensor, "detach"):
        arr = img_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(img_tensor)
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def overlay_heatmap(
    rgb: Image.Image, heat_prob: np.ndarray, alpha: float = 0.5
) -> Image.Image:
    """Overlay a heat probability map (H, W) on an RGB PIL image in red colormap."""
    heat = np.clip(heat_prob, 0.0, 1.0)
    H, W = heat.shape
    # Create red overlay
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[..., 0] = (heat * 255).astype(np.uint8)
    base = np.array(rgb).astype(np.float32)
    out = (base * (1 - alpha) + overlay.astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def draw_points(
    rgb: Image.Image,
    points: np.ndarray,
    labels: np.ndarray,
    colors: List[Tuple[int, int, int]],
    radius: int = 3,
) -> Image.Image:
    """Draw points as small filled circles with class-specific colors."""
    vis = rgb.copy()
    draw = ImageDraw.Draw(vis, "RGBA")
    for (x, y), c in zip(points, labels):
        c_idx = int(c)
        if c_idx < 0 or c_idx >= len(colors):
            color = (255, 255, 255, 200)
        else:
            r, g, b = colors[c_idx]
            color = (r, g, b, 200)
        x0, y0 = int(x - radius), int(y - radius)
        x1, y1 = int(x + radius), int(y + radius)
        draw.ellipse((x0, y0, x1, y1), fill=color, outline=None)
    return vis


def overlay_segmentation(
    rgb: Image.Image,
    seg_map: np.ndarray,
    colors: List[Tuple[int, int, int]],
    alpha: float = 0.35,
) -> Image.Image:
    """Overlay a HxW integer segmentation map on RGB image using a color palette."""
    H, W = seg_map.shape
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_idx in range(len(colors)):
        mask = seg_map == cls_idx
        overlay[mask] = colors[cls_idx]
    base = np.array(rgb).astype(np.float32)
    out = (base * (1 - alpha) + overlay.astype(np.float32) * alpha).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def save_legend(
    path: str,
    names: List[str],
    colors: List[Tuple[int, int, int]],
    title: Optional[str] = None,
) -> None:
    """Save a small legend image mapping names to colors."""
    H = 20 * (len(names) + (1 if title else 0)) + 10
    W = max(280, max((len(n) for n in names), default=10) * 10 + 100)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 10
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if title:
        draw.text((10, y), title, fill=(0, 0, 0), font=font)
        y += 20
    for i, (name, color) in enumerate(zip(names, colors)):
        draw.rectangle((10, y + 4, 30, y + 16), fill=color, outline=(0, 0, 0))
        draw.text((40, y), f"{i}: {name}", fill=(0, 0, 0), font=font)
        y += 20
    img.save(path)


def default_class_colors(n: int) -> List[Tuple[int, int, int]]:
    """Return *n* distinct-ish RGB colors (deterministic)."""
    preset = [
        (230, 25, 75),  # red
        (60, 180, 75),  # green
        (0, 130, 200),  # blue
        (245, 130, 48),  # orange
        (145, 30, 180),  # purple
        (70, 240, 240),  # cyan
        (240, 50, 230),  # magenta
        (210, 245, 60),  # lime
        (250, 190, 190),  # pink
        (0, 128, 128),  # teal
        (230, 190, 255),  # lavender
        (170, 110, 40),  # brown
        (255, 250, 200),  # beige
        (128, 0, 0),  # maroon
        (170, 255, 195),  # mint
        (128, 128, 0),  # olive
        (255, 215, 180),  # peach
        (0, 0, 128),  # navy
        (128, 128, 128),  # gray
        (255, 255, 255),  # white
    ]
    if n <= len(preset):
        return preset[:n]
    # Simple extension if needed
    out = preset.copy()
    rng = np.random.default_rng(42)
    while len(out) < n:
        out.append(tuple(int(x) for x in rng.integers(0, 255, size=3)))
    return out
