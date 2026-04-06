"""DOM 色彩均衡与匀色处理。"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("图像为空")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _match_cdf(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.uint8)
    template = np.asarray(template, dtype=np.uint8)

    s_values, s_indices, s_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template.ravel(), return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[s_indices].reshape(source.shape).astype(np.uint8)


def _build_average_reference(images: List[dict]) -> np.ndarray:
    ref = _ensure_bgr(images[0]["image"])
    ref_h, ref_w = ref.shape[:2]
    acc = np.zeros_like(ref, dtype=np.float32)
    count = 0
    for item in images:
        img = _ensure_bgr(item["image"])
        if img.shape[:2] != (ref_h, ref_w):
            img = cv2.resize(img, (ref_w, ref_h), interpolation=cv2.INTER_AREA)
        acc += img.astype(np.float32)
        count += 1
    return np.clip(acc / max(count, 1), 0, 255).astype(np.uint8)


def align_mean_brightness(images: List[dict], reference_index: int | None = 0) -> List[np.ndarray]:
    """基于 LAB 空间 L 通道的平均亮度对齐。"""
    if not images:
        return []

    ref_img = _ensure_bgr(images[reference_index]["image"]) if reference_index is not None else _build_average_reference(images)
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)
    ref_l = ref_lab[:, :, 0].astype(np.float32)
    ref_mean = ref_l.mean()
    ref_std = ref_l.std() if ref_l.std() > 1e-6 else 1.0

    results = []
    for item in images:
        img = _ensure_bgr(item["image"])
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l = lab[:, :, 0].astype(np.float32)
        mean = l.mean()
        std = l.std() if l.std() > 1e-6 else 1.0
        adjusted = (l - mean) * (ref_std / std) + ref_mean
        lab[:, :, 0] = np.clip(adjusted, 0, 255).astype(np.uint8)
        results.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    return results


def match_histogram(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """逐通道直方图匹配。"""
    src = _ensure_bgr(image)
    ref = _ensure_bgr(reference)
    matched = np.empty_like(src)
    for channel in range(3):
        matched[:, :, channel] = _match_cdf(src[:, :, channel], ref[:, :, channel])
    return matched


def match_histogram_images(images: List[dict], reference_index: int | None = 0) -> List[np.ndarray]:
    if not images:
        return []
    ref_img = _ensure_bgr(images[reference_index]["image"]) if reference_index is not None else _build_average_reference(images)
    return [match_histogram(_ensure_bgr(item["image"]), ref_img) for item in images]
