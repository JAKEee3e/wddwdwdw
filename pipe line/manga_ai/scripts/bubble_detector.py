from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from .storyboard import DialogueLine, PanelSpec


BBoxPx = Tuple[int, int, int, int]


@dataclass
class SAMBubbleBackend:
    mask_generator: SamAutomaticMaskGenerator


def load_sam(
    checkpoint_path: str | Path,
    model_type: str = "vit_h",
    device: Optional[str] = None,
) -> SAMBubbleBackend:
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = os.environ.get("MANGA_AI_SAM_DEVICE")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    try:
        sam.to(device=device)
    except torch.OutOfMemoryError:
        if device != "cpu":
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            sam.to(device="cpu")
            device = "cpu"
        else:
            raise

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=512,
    )

    return SAMBubbleBackend(mask_generator=mask_generator)


def _image_to_np_rgb(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def _union_occupancy(masks: List[dict], h: int, w: int) -> np.ndarray:
    occ = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        seg = m.get("segmentation")
        if seg is None:
            continue
        area = int(m.get("area", 0))
        if area < 1024:
            continue
        occ |= seg.astype(np.uint8)
    return occ


def _edge_density(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 80, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return (edges > 0).astype(np.uint8)


def _best_bubble_rect(
    occ: np.ndarray,
    edge: np.ndarray,
    bubble_w: int,
    bubble_h: int,
    prefer_top: bool = True,
) -> BBoxPx:
    h, w = occ.shape

    stride = max(16, min(bubble_w, bubble_h) // 6)

    x_candidates = list(range(0, max(1, w - bubble_w), stride))
    y_candidates = list(range(0, max(1, h - bubble_h), stride))

    if prefer_top:
        y_candidates = sorted(y_candidates, key=lambda y: y)
    else:
        y_candidates = sorted(y_candidates, key=lambda y: abs(y - h // 2))

    best = (1e9, 0, 0)
    best_xy = (max(0, w - bubble_w), 0)

    for y in y_candidates:
        for x in x_candidates:
            roi_occ = occ[y : y + bubble_h, x : x + bubble_w]
            roi_edge = edge[y : y + bubble_h, x : x + bubble_w]

            occ_score = float(roi_occ.mean())
            edge_score = float(roi_edge.mean())
            score = occ_score * 2.0 + edge_score

            tie = abs(x - (w * 0.75)) / w
            key = (score, tie, y)
            if key < best:
                best = key
                best_xy = (x, y)

        if best[0] < 0.03:
            break

    x, y = best_xy
    return (int(x), int(y), int(bubble_w), int(bubble_h))


def suggest_bubble_bboxes(
    panel_image: Image.Image,
    dialogue: List[DialogueLine],
    backend: SAMBubbleBackend,
) -> List[BBoxPx]:
    rgb = _image_to_np_rgb(panel_image)
    h, w, _ = rgb.shape

    masks = backend.mask_generator.generate(rgb)
    occ = _union_occupancy(masks, h=h, w=w)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge = _edge_density(gray)

    n = max(1, len(dialogue))
    bubble_w = int(w * 0.48)
    bubble_h_each = int(h * 0.16)
    bubble_h_each = max(64, min(bubble_h_each, h // 3))

    bboxes: List[BBoxPx] = []
    used = np.zeros_like(occ)

    for i in range(n):
        rect = _best_bubble_rect(
            occ=np.clip(occ + used, 0, 1),
            edge=edge,
            bubble_w=bubble_w,
            bubble_h=bubble_h_each,
            prefer_top=True,
        )
        x, y, bw, bh = rect
        used[y : y + bh, x : x + bw] = 1
        bboxes.append(rect)

    return bboxes


def bubble_mask_from_bboxes(
    image_size: tuple[int, int],
    bboxes: List[BBoxPx],
    pad_px: int = 10,
) -> Image.Image:
    w, h = image_size
    mask = np.zeros((h, w), dtype=np.uint8)

    for (x, y, bw, bh) in bboxes:
        x0 = max(0, x + pad_px)
        y0 = max(0, y + pad_px)
        x1 = min(w - 1, x + bw - pad_px)
        y1 = min(h - 1, y + bh - pad_px)

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        rx = max(8, (x1 - x0) // 2)
        ry = max(8, (y1 - y0) // 2)

        cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 255, -1)

    return Image.fromarray(mask, mode="L")


def bubble_masks_from_bboxes(
    image_size: tuple[int, int],
    bboxes: List[BBoxPx],
    pad_px: int = 10,
) -> List[Image.Image]:
    w, h = image_size
    masks: List[Image.Image] = []

    for (x, y, bw, bh) in bboxes:
        mask = np.zeros((h, w), dtype=np.uint8)

        x0 = max(0, x + pad_px)
        y0 = max(0, y + pad_px)
        x1 = min(w - 1, x + bw - pad_px)
        y1 = min(h - 1, y + bh - pad_px)

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        rx = max(8, (x1 - x0) // 2)
        ry = max(8, (y1 - y0) // 2)

        cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 255, -1)
        masks.append(Image.fromarray(mask, mode="L"))

    return masks
