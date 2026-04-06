"""DOM 导出工具。"""
from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None


def _normalize_for_save(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return image


def save_png(image: np.ndarray, path: str) -> str:
    normalized = _normalize_for_save(image)
    ok, encoded = cv2.imencode(".png", normalized)
    if ok:
        encoded.tofile(path)
        return path
    if Image is None:
        raise RuntimeError("PNG 编码失败")
    if normalized.ndim == 3 and normalized.shape[2] == 3:
        pil_array = normalized[:, :, ::-1]
    elif normalized.ndim == 3 and normalized.shape[2] == 4:
        pil_array = normalized[:, :, [2, 1, 0, 3]]
    else:
        pil_array = normalized
    pil = Image.fromarray(pil_array)
    pil.save(path)
    return path


def save_tiff(image: np.ndarray, path: str) -> str:
    normalized = _normalize_for_save(image)
    ok, encoded = cv2.imencode(".tif", normalized)
    if ok:
        encoded.tofile(path)
        return path
    if Image is None:
        raise RuntimeError("TIFF 编码失败")
    if normalized.ndim == 3 and normalized.shape[2] == 3:
        pil_array = normalized[:, :, ::-1]
    elif normalized.ndim == 3 and normalized.shape[2] == 4:
        pil_array = normalized[:, :, [2, 1, 0, 3]]
    else:
        pil_array = normalized
    pil = Image.fromarray(pil_array)
    pil.save(path, format="TIFF")
    return path


def save_geotiff_if_possible(image: np.ndarray, path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    if rasterio is None or not metadata:
        return False
    transform = metadata.get("transform")
    if transform is None:
        return False

    crs = metadata.get("crs")
    if crs is None:
        return False

    profile = {
        "driver": "GTiff",
        "height": image.shape[0],
        "width": image.shape[1],
        "count": image.shape[2] if image.ndim == 3 else 1,
        "dtype": image.dtype,
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        if image.ndim == 2:
            dst.write(image, 1)
        else:
            if image.shape[2] == 4:
                rgb = image[:, :, [2, 1, 0, 3]]
            else:
                rgb = image[:, :, ::-1]
            for i in range(rgb.shape[2]):
                dst.write(rgb[:, :, i], i + 1)
    return True
