"""模块三处理器。"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from core.log_manager import log_manager

from .models import DemResult, DsmResult
from .utils import (
    align_image_sizes,
    build_point_cloud_from_grid,
    clahe_gray,
    fill_invalid_nearest,
    format_elapsed,
    hillshade,
    normalize_to_u8,
    read_image,
    resize_to_max_side,
    timestamped_name,
)


class DsmDemProcessor:
    """模块三核心处理器。"""

    def generate_dsm(self, config: Dict[str, Any], progress_cb=None, stage_cb=None) -> DsmResult:
        started = time.perf_counter()
        left_path = config.get("left_image_path") or ""
        right_path = config.get("right_image_path") or ""
        left_name = config.get("left_image_name") or ""
        right_name = config.get("right_image_name") or ""
        self._update(stage_cb, progress_cb, "输入检查", 5)
        self._validate_inputs(left_path, right_path)

        log_manager.info(f"模块三：开始生成 DSM，左影像={left_name}，右影像={right_name}")
        left_image = read_image(left_path, grayscale=False)
        right_image = read_image(right_path, grayscale=False)
        max_side = int(config.get("max_processing_side", 1280))
        left_image, _ = resize_to_max_side(left_image, max_side)
        right_image, _ = resize_to_max_side(right_image, max_side)
        left_image, right_image = align_image_sizes(left_image, right_image)

        self._update(stage_cb, progress_cb, "影像预处理", 15)
        left_gray = clahe_gray(left_image, bool(config.get("use_clahe", True)))
        right_gray = clahe_gray(right_image, bool(config.get("use_clahe", True)))

        self._update(stage_cb, progress_cb, "近似配准", 30)
        rectified_right, match_info = self._approximate_rectify(left_gray, right_gray)
        if match_info["status"] != "ok":
            raise ValueError(match_info["message"])
        log_manager.info(
            f"模块三：ORB 配准完成，匹配点 {match_info['raw_matches']}，内点 {match_info['inliers']}，内点率 {match_info['inlier_ratio']:.2%}"
        )

        self._update(stage_cb, progress_cb, "密集匹配", 55)
        disparity = self._compute_disparity(left_gray, rectified_right, config)

        self._update(stage_cb, progress_cb, "视差后处理", 70)
        disparity_filtered, valid_mask, valid_ratio = self._post_process_disparity(disparity, config)
        if valid_ratio < 0.02:
            raise ValueError(f"视差有效率过低（{valid_ratio:.2%}），无法生成可靠 DSM。")

        self._update(stage_cb, progress_cb, "DSM 栅格生成", 85)
        dsm_grid, mode_text = self._disparity_to_relative_height(disparity_filtered, valid_mask, config)
        z_min, z_max = self._range_of_grid(dsm_grid, valid_mask)
        if not np.any(valid_mask):
            raise ValueError("DSM 网格没有有效值。")

        disparity_preview = normalize_to_u8(disparity_filtered, valid_mask, cv2.COLORMAP_JET)
        preview_hillshade = hillshade(dsm_grid, valid_mask)
        point_cloud = build_point_cloud_from_grid(dsm_grid, valid_mask)
        elapsed = format_elapsed(time.perf_counter() - started)
        self._update(stage_cb, progress_cb, "DSM 完成", 100)

        return DsmResult(
            success=True,
            message="DSM 生成完成",
            left_name=left_name,
            right_name=right_name,
            result_name=timestamped_name("DSM"),
            disparity=disparity_filtered,
            disparity_preview=disparity_preview,
            dsm_grid=dsm_grid,
            hillshade=preview_hillshade,
            point_cloud=point_cloud,
            valid_mask=valid_mask,
            valid_ratio=valid_ratio,
            z_min=z_min,
            z_max=z_max,
            elapsed_text=elapsed,
            mode_text=mode_text,
        )

    def generate_dem(self, dsm_result: DsmResult, config: Dict[str, Any], progress_cb=None, stage_cb=None) -> DemResult:
        started = time.perf_counter()
        if dsm_result is None or dsm_result.dsm_grid is None or dsm_result.valid_mask is None:
            raise ValueError("当前没有可用于生成 DEM 的 DSM 栅格。")

        self._update(stage_cb, progress_cb, "读取 DSM", 10)
        dsm_grid = np.asarray(dsm_result.dsm_grid, dtype=np.float32)
        valid_mask = np.asarray(dsm_result.valid_mask, dtype=bool)
        if not np.any(valid_mask):
            raise ValueError("DSM 网格缺少有效像元，无法生成 DEM。")

        method_text = str(config.get("dem_method", "形态学滤波（P0）"))
        if "坡度" in method_text:
            self._update(stage_cb, progress_cb, "坡度地面滤波", 45)
            dem_grid, ground_mask = self._slope_ground_filter(dsm_grid, valid_mask, config)
        else:
            self._update(stage_cb, progress_cb, "形态学地面滤波", 45)
            dem_grid, ground_mask = self._morphological_ground_filter(dsm_grid, valid_mask, config)

        self._update(stage_cb, progress_cb, "DEM 插值与平滑", 75)
        dem_grid = self._smooth_dem(dem_grid, valid_mask, config)
        z_min, z_max = self._range_of_grid(dem_grid, valid_mask)
        hillshade_image = hillshade(dem_grid, valid_mask)
        mask_preview = self._mask_preview(ground_mask)
        elapsed = format_elapsed(time.perf_counter() - started)
        ground_ratio = float(np.count_nonzero(ground_mask)) / float(np.count_nonzero(valid_mask))
        self._update(stage_cb, progress_cb, "DEM 完成", 100)

        return DemResult(
            success=True,
            message="DEM 生成完成",
            result_name=timestamped_name("DEM"),
            dem_grid=dem_grid,
            ground_mask=ground_mask,
            ground_mask_preview=mask_preview,
            hillshade=hillshade_image,
            valid_mask=valid_mask,
            ground_ratio=ground_ratio,
            z_min=z_min,
            z_max=z_max,
            elapsed_text=elapsed,
            mode_text="相对高程，不代表绝对地理高程",
        )

    def _validate_inputs(self, left_path: str, right_path: str):
        if not left_path or not right_path:
            raise ValueError("请先在右侧面板中选择左右影像。")
        if left_path == right_path:
            raise ValueError("左右影像不能是同一张图像。")

    def _approximate_rectify(self, left_gray: np.ndarray, right_gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        orb = cv2.ORB_create(2500)
        kp1, des1 = orb.detectAndCompute(left_gray, None)
        kp2, des2 = orb.detectAndCompute(right_gray, None)
        if des1 is None or des2 is None or len(kp1) < 12 or len(kp2) < 12:
            return right_gray, {"status": "error", "message": "影像纹理过弱或特征点不足，无法完成近似配准。"}

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        knn = matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        if len(good_matches) < 12:
            return right_gray, {"status": "error", "message": "特征匹配数量不足，无法估计稳定变换。"}

        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        _, inlier_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if inlier_mask is None:
            return right_gray, {"status": "error", "message": "RANSAC 配准失败，无法继续密集匹配。"}

        inlier_mask = inlier_mask.reshape(-1).astype(bool)
        inliers = int(np.count_nonzero(inlier_mask))
        inlier_ratio = inliers / max(1, len(good_matches))
        if inlier_ratio < 0.2 or inliers < 10:
            return right_gray, {"status": "error", "message": "配准内点比例过低，无法生成可靠视差。"}

        y_offsets = dst_pts[inlier_mask, 1] - src_pts[inlier_mask, 1]
        median_dy = float(np.median(y_offsets)) if y_offsets.size else 0.0
        # 只做近似极线对齐，不消除水平视差本身。
        transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, median_dy]], dtype=np.float32)
        warped = cv2.warpAffine(right_gray, transform, (left_gray.shape[1], left_gray.shape[0]))
        return warped, {
            "status": "ok",
            "raw_matches": len(good_matches),
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "median_vertical_shift": median_dy,
            "message": "ok",
        }

    def _compute_disparity(self, left_gray: np.ndarray, right_gray: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        block_size = int(config.get("block_size", 7))
        if block_size % 2 == 0:
            block_size += 1
        num_disparities = int(config.get("num_disparities", 96))
        num_disparities = max(16, (num_disparities // 16) * 16)
        min_disparity = int(config.get("min_disparity", 0))
        uniqueness_ratio = int(config.get("uniqueness_ratio", 10))
        speckle_window_size = int(config.get("speckle_window_size", 50))
        speckle_range = int(config.get("speckle_range", 2))
        p1 = 8 * block_size * block_size
        p2 = 32 * block_size * block_size
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            disp12MaxDiff=1,
            P1=p1,
            P2=p2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity

    def _post_process_disparity(self, disparity: np.ndarray, config: Dict[str, Any]):
        min_valid = max(0.1, float(config.get("min_valid_disparity", 0.5)))
        filtered = disparity.copy()
        valid_mask = np.isfinite(filtered) & (filtered > min_valid)
        filtered[~valid_mask] = 0.0
        filtered = cv2.medianBlur(filtered, 5)
        valid_mask = np.isfinite(filtered) & (filtered > min_valid)
        valid_ratio = float(np.count_nonzero(valid_mask)) / float(filtered.size)
        return filtered, valid_mask, valid_ratio

    def _disparity_to_relative_height(self, disparity: np.ndarray, valid_mask: np.ndarray, config: Dict[str, Any]):
        focal_mm = float(config.get("focal_length_mm", 0.0))
        baseline_m = float(config.get("baseline_m", 0.0))
        pixel_size_um = float(config.get("pixel_size_um", 0.0))

        height = np.full(disparity.shape, np.nan, dtype=np.float32)
        if focal_mm > 0 and baseline_m > 0 and pixel_size_um > 0:
            focal_px = focal_mm / (pixel_size_um / 1000.0)
            rel_depth = (focal_px * baseline_m) / np.maximum(disparity[valid_mask], 1e-3)
            rel_depth = rel_depth - np.min(rel_depth)
            height[valid_mask] = rel_depth.astype(np.float32)
            mode_text = "已使用焦距/基线/像元参数恢复比例意义更明确的相对高程，但不承诺绝对高程精度。"
        else:
            rel_height = disparity[valid_mask]
            rel_height = rel_height - np.min(rel_height)
            height[valid_mask] = rel_height.astype(np.float32)
            mode_text = "当前结果为相对 DSM，不代表绝对地理高程。"

        if np.count_nonzero(valid_mask) == 0:
            raise ValueError("视差结果中没有有效像元，无法恢复 DSM。")
        height = fill_invalid_nearest(height, valid_mask)
        return height, mode_text

    def _morphological_ground_filter(self, dsm_grid: np.ndarray, valid_mask: np.ndarray, config: Dict[str, Any]):
        filled = fill_invalid_nearest(dsm_grid, valid_mask)
        kernel_size = max(3, int(config.get("morph_kernel_size", 9)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        threshold = float(config.get("ground_threshold", 1.5))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
        opened = cv2.GaussianBlur(opened, (0, 0), sigmaX=1.0)
        diff = filled - opened
        ground_mask = valid_mask & (diff <= threshold)
        if np.count_nonzero(ground_mask) < 16:
            raise ValueError("地面候选区域过少，无法生成稳定 DEM。")
        dem = opened.copy()
        dem[~ground_mask] = np.nan
        dem = fill_invalid_nearest(dem, ground_mask)
        return dem.astype(np.float32), ground_mask

    def _smooth_dem(self, dem_grid: np.ndarray, valid_mask: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        sigma = float(config.get("smooth_sigma", 1.0))
        if sigma <= 0:
            return dem_grid
        dem = fill_invalid_nearest(dem_grid, valid_mask)
        dem = cv2.GaussianBlur(dem, (0, 0), sigmaX=sigma)
        return dem.astype(np.float32)

    def _slope_ground_filter(self, dsm_grid: np.ndarray, valid_mask: np.ndarray, config: Dict[str, Any]):
        filled = fill_invalid_nearest(dsm_grid, valid_mask)
        grad_y, grad_x = np.gradient(filled)
        slope = np.sqrt(grad_x * grad_x + grad_y * grad_y)
        threshold = max(0.05, float(config.get("ground_threshold", 1.5)))
        kernel_size = max(3, int(config.get("morph_kernel_size", 9)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        local_min = cv2.erode(filled, np.ones((kernel_size, kernel_size), dtype=np.uint8))
        height_delta = filled - local_min
        ground_mask = valid_mask & (slope <= threshold) & (height_delta <= threshold * 1.5)
        if np.count_nonzero(ground_mask) < 16:
            raise ValueError("坡度地面候选区域过少，无法生成稳定 DEM。")
        dem = filled.copy()
        dem[~ground_mask] = np.nan
        dem = fill_invalid_nearest(dem, ground_mask)
        return dem.astype(np.float32), ground_mask

    def _mask_preview(self, mask: np.ndarray) -> np.ndarray:
        data = np.zeros(mask.shape + (3,), dtype=np.uint8)
        data[mask] = (0, 180, 0)
        data[~mask] = (50, 50, 50)
        return data

    def _range_of_grid(self, grid: np.ndarray, valid_mask: Optional[np.ndarray]):
        values = np.asarray(grid, dtype=np.float32)
        if valid_mask is None:
            valid = np.isfinite(values)
        else:
            valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
        if not np.any(valid):
            return 0.0, 0.0
        return float(np.min(values[valid])), float(np.max(values[valid]))

    def _update(self, stage_cb, progress_cb, stage: str, progress: int):
        if stage_cb:
            stage_cb(stage)
        if progress_cb:
            progress_cb(progress)
        log_manager.info(f"模块三：{stage}")
