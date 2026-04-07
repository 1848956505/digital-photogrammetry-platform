"""DOM 融合处理。"""
from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _prepare_weight(mask: np.ndarray, feather_radius: int = 15) -> np.ndarray:
    weight = mask.astype(np.float32) / 255.0
    if feather_radius > 1:
        sigma = max(1.0, feather_radius / 3.0)
        weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(weight, 0.0, 1.0)


def compose_layers(layers: List[dict], method: str = "feather", feather_radius: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    if not layers:
        raise ValueError("没有可融合的图层")

    canvas_shape = layers[0]["image"].shape
    if len(canvas_shape) == 2:
        canvas_h, canvas_w = canvas_shape
        canvas_c = 1
    else:
        canvas_h, canvas_w, canvas_c = canvas_shape

    accum = np.zeros((canvas_h, canvas_w, canvas_c), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for layer in layers:
        image = layer["image"]
        mask = layer["mask"]
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        weight = _prepare_weight(mask, feather_radius if method == "feather" else 1)
        accum += image.astype(np.float32) * weight[:, :, None]
        weight_sum += weight

    denom = np.where(weight_sum > 1e-6, weight_sum, 1.0).astype(np.float32)
    result = accum / denom[:, :, None]
    result[weight_sum <= 1e-6] = 0
    result = _ensure_uint8(result)
    final_mask = (weight_sum > 0).astype(np.uint8) * 255
    return result, final_mask


def weighted_blend(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    weight = mask.astype(np.float32) / 255.0
    base_f = base.astype(np.float32)
    overlay_f = overlay.astype(np.float32)
    result = base_f * (1.0 - weight[:, :, None]) + overlay_f * weight[:, :, None]
    return _ensure_uint8(result)


def feather_blend(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray, feather_radius: int = 15) -> np.ndarray:
    weight = _prepare_weight(mask, feather_radius)
    return weighted_blend(base, overlay, (weight * 255).astype(np.uint8))


def weighted_average_canvas(image: np.ndarray, mask: np.ndarray):
    return image, mask


def feather_blend_canvas(image: np.ndarray, mask: np.ndarray, feather_radius: int = 15):
    if mask is None:
        return image
    weight = _prepare_weight(mask, feather_radius)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result = image.astype(np.float32) * weight[:, :, None]
    return _ensure_uint8(result)
