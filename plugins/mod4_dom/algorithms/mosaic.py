"""DOM 镶嵌与几何拼接。"""
from __future__ import annotations

import os
import time
import warnings
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

try:
    import rasterio
    from rasterio.merge import merge as rio_merge
    try:
        from rasterio.errors import NotGeoreferencedWarning
    except Exception:  # pragma: no cover
        NotGeoreferencedWarning = None
except Exception:  # pragma: no cover
    rasterio = None
    rio_merge = None
    NotGeoreferencedWarning = None


def _read_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    raw = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取影像: {path}")
    return image


def load_images_from_workspace_entries(entries: List[Dict[str, str]]) -> List[Dict[str, object]]:
    loaded = []
    for entry in entries:
        loaded.append({
            "name": entry["name"],
            "path": entry["path"],
            "image": _read_image(entry["path"]),
        })
    return loaded


def _geo_available(path: str) -> bool:
    if rasterio is None:
        return False
    try:
        with warnings.catch_warnings():
            if NotGeoreferencedWarning is not None:
                warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(path) as ds:
                return ds.crs is not None and ds.transform is not None
    except Exception:
        return False


def has_geo_metadata(entries: List[Dict[str, str]]) -> bool:
    if not entries:
        return False
    return all(_geo_available(entry["path"]) for entry in entries)


def mosaic_with_georef(entries: List[Dict[str, object]], progress_cb: Callable[[int], None] | None = None) -> Dict[str, object]:
    if rasterio is None or rio_merge is None:
        raise RuntimeError("当前环境缺少 rasterio，无法执行地理镶嵌")

    start = time.time()
    datasets = []
    try:
        for entry in entries:
            with warnings.catch_warnings():
                if NotGeoreferencedWarning is not None:
                    warnings.simplefilter("ignore", NotGeoreferencedWarning)
                datasets.append(rasterio.open(entry["path"]))
        if any(ds.crs is None for ds in datasets):
            raise RuntimeError("输入影像缺少 CRS，无法执行地理镶嵌")

        merged, transform = rio_merge(datasets)
        if progress_cb:
            progress_cb(55)

        if merged.ndim == 2:
            image = np.clip(merged, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = np.moveaxis(merged, 0, -1)
            if image.shape[2] == 1:
                image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
            elif image.shape[2] > 3:
                image = image[:, :, :3]
            image = np.clip(image, 0, 255).astype(np.uint8)

        mask = (image.sum(axis=2) > 0).astype(np.uint8) * 255
        meta = datasets[0].meta.copy()
        meta.update({
            "height": image.shape[0],
            "width": image.shape[1],
            "count": image.shape[2] if image.ndim == 3 else 1,
            "transform": transform,
        })
        return {
            "mode": "georef",
            "image": image,
            "mask": mask,
            "layers": None,
            "metadata": meta,
            "elapsed_text": f"{time.time() - start:.3f}s",
        }
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass


def _detect_and_compute(image: np.ndarray, method: str = "ORB"):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    method = method.upper()
    if method == "SIFT" and hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create()
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=3000)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        norm = cv2.NORM_HAMMING
    return keypoints, descriptors, norm


def estimate_homography(src_image: np.ndarray, dst_image: np.ndarray, method: str = "ORB") -> Tuple[np.ndarray, Dict[str, object]]:
    kp1, des1, norm = _detect_and_compute(src_image, method)
    kp2, des2, _ = _detect_and_compute(dst_image, method)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        raise RuntimeError("特征点不足，无法估计单应矩阵")

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn_matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 4:
        raise RuntimeError("有效匹配点不足，无法估计单应矩阵")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    if H is None:
        raise RuntimeError("单应矩阵估计失败")

    inliers = int(mask.ravel().sum()) if mask is not None else 0
    return H, {
        "matches": len(good),
        "inliers": inliers,
        "method": method.upper(),
    }


def _warp_corners(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, H)


def mosaic_with_feature_matching(
    entries: List[Dict[str, object]],
    feature_method: str = "ORB",
    progress_cb: Callable[[int], None] | None = None,
) -> Dict[str, object]:
    start = time.time()
    images = [entry["image"] for entry in entries]
    if len(images) < 2:
        raise ValueError("至少需要两张影像")

    transforms = [np.eye(3, dtype=np.float64)]
    stats = []
    for idx in range(1, len(images)):
        if progress_cb:
            progress_cb(30 + int(20 * idx / max(1, len(images) - 1)))
        current = images[idx]
        prev = images[idx - 1]
        try:
            H_rel, stat = estimate_homography(current, prev, method=feature_method)
        except Exception:
            H_rel, stat = estimate_homography(current, images[0], method=feature_method)
            transforms.append(transforms[0] @ H_rel)
        else:
            transforms.append(transforms[idx - 1] @ H_rel)
        stats.append(stat)

    all_corners = []
    for image, H in zip(images, transforms):
        all_corners.append(_warp_corners(image, H))
    all_corners = np.concatenate(all_corners, axis=0)

    min_x = int(np.floor(all_corners[:, 0, 0].min()))
    min_y = int(np.floor(all_corners[:, 0, 1].min()))
    max_x = int(np.ceil(all_corners[:, 0, 0].max()))
    max_y = int(np.ceil(all_corners[:, 0, 1].max()))

    offset = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64)
    canvas_w = max(1, max_x - min_x)
    canvas_h = max(1, max_y - min_y)

    layers = []
    union_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for image, H in zip(images, transforms):
        warp_h = offset @ H
        warped = cv2.warpPerspective(image, warp_h, (canvas_w, canvas_h))
        base_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(base_mask, warp_h, (canvas_w, canvas_h))
        union_mask = cv2.bitwise_or(union_mask, warped_mask)
        layers.append({
            "image": warped,
            "mask": warped_mask,
        })

    return {
        "mode": "feature",
        "layers": layers,
        "mask": union_mask,
        "metadata": {
            "canvas_size": (canvas_w, canvas_h),
            "offset": offset,
            "transforms": transforms,
            "stats": stats,
        },
        "elapsed_text": f"{time.time() - start:.3f}s",
    }


def crop_valid_region(image: np.ndarray, mask: np.ndarray):
    if mask is None:
        return image, mask
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image, mask
    x, y, w, h = cv2.boundingRect(coords)
    return image[y : y + h, x : x + w], mask[y : y + h, x : x + w]
