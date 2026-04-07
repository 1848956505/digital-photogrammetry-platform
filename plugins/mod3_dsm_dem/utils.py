"""模块三工具函数。"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage


def ensure_output_dir(output_dir: Optional[str] = None) -> str:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    root_dir = Path(__file__).resolve().parents[2]
    target = root_dir / "output" / "mod3"
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def timestamped_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def read_image(path: str, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, flag)
    if image is None:
        raise ValueError(f"无法读取影像: {path}")
    return image


def save_preview_image(path: str, image: np.ndarray) -> str:
    image = np.asarray(image)
    ext = Path(path).suffix.lower() or ".png"
    suffix = ext if ext.startswith(".") else f".{ext}"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"无法保存图像: {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)
    return path


def save_grid_to_npy(path: str, grid: np.ndarray) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(grid, dtype=np.float32))
    return path


def save_point_cloud_xyz(path: str, points: np.ndarray) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(points, dtype=np.float32), fmt="%.6f")
    return path


def resize_to_max_side(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = 1.0
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return image, scale


def align_image_sizes(left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    left_resized = cv2.resize(left, (w, h), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA)
    return left_resized, right_resized


def clahe_gray(image: np.ndarray, enabled: bool) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if not enabled:
        return gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def normalize_to_u8(array: np.ndarray, mask: Optional[np.ndarray] = None, color_map: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if mask is None:
        valid = np.isfinite(arr)
    else:
        valid = np.asarray(mask, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if np.any(valid):
        valid_values = arr[valid]
        v_min = float(np.min(valid_values))
        v_max = float(np.max(valid_values))
        if v_max > v_min:
            out[valid] = np.clip((arr[valid] - v_min) / (v_max - v_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            out[valid] = 128
    if color_map is not None:
        return cv2.applyColorMap(out, color_map)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def hillshade(grid: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    z = np.asarray(grid, dtype=np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(z)
    filled = fill_invalid_nearest(z, valid_mask)
    grad_y, grad_x = np.gradient(filled)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(grad_x * grad_x + grad_y * grad_y))
    aspect = np.arctan2(-grad_x, grad_y)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(45.0)
    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    shaded = np.clip(shaded, 0, 1)
    shaded_u8 = (shaded * 255.0).astype(np.uint8)
    shaded_u8[~valid_mask] = 0
    return cv2.cvtColor(shaded_u8, cv2.COLOR_GRAY2BGR)


def fill_invalid_nearest(grid: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    data = np.asarray(grid, dtype=np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(data)
    if not np.any(valid_mask):
        raise ValueError("栅格中没有有效数值，无法进行插值。")
    filled = data.copy()
    invalid = ~valid_mask
    filled[invalid] = 0.0
    indices = ndimage.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    filled = filled[tuple(indices)]
    return filled


def build_point_cloud_from_grid(grid: np.ndarray, valid_mask: np.ndarray, max_points: int = 25000) -> Optional[np.ndarray]:
    if grid is None or valid_mask is None or not np.any(valid_mask):
        return None
    y_coords, x_coords = np.nonzero(valid_mask)
    z_values = grid[valid_mask]
    count = len(z_values)
    if count == 0:
        return None
    if count > max_points:
        step = max(1, count // max_points)
        y_coords = y_coords[::step]
        x_coords = x_coords[::step]
        z_values = z_values[::step]
    points = np.column_stack([x_coords.astype(np.float32), y_coords.astype(np.float32), z_values.astype(np.float32)])
    return points


def format_elapsed(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    return f"{seconds:.2f} s"
