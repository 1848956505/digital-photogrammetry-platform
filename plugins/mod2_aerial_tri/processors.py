"""
模块二：空中三角测量处理器
包含相对定向、区域网平差、残差分析、异常点检测等功能
"""
import os
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import cv2


class RelativeOrientation:
    """相对定向处理器"""
    
    @staticmethod
    def compute_relative_orientation(
        img1_points: np.ndarray, 
        img2_points: np.ndarray,
        focal_length: float
    ) -> Dict[str, Any]:
        """
        计算相对定向参数
        
        Args:
            img1_points: 影像1控制点坐标 (N x 2)
            img2_points: 影像2控制点坐标 (N x 2)
            focal_length: 焦距 (mm)
            
        Returns:
            相对定向参数和旋转矩阵
        """
        n = len(img1_points)
        if n < 5:
            return {"success": False, "error": "需要至少5个控制点"}
        
        # 模拟相对定向计算（实际需要内外方位元素）
        # 这里使用简化的模拟实现
        omega = np.random.randn() * 0.01  # 航向倾角
        phi = np.random.randn() * 0.01    # 旁向倾角
        kappa = np.random.randn() * 0.01  # 旋角
        
        # 旋转矩阵
        R = RelativeOrientation._rotation_matrix(omega, phi, kappa)
        
        # 计算相对定向精度
        residual = np.random.randn(n, 2) * 0.5  # 模拟残差
        rmse = np.sqrt(np.mean(residual**2))
        
        return {
            "success": True,
            "omega": omega,
            "phi": phi,
            "kappa": kappa,
            "rotation_matrix": R,
            "rmse": rmse,
            "point_count": n
        }
    
    @staticmethod
    def _rotation_matrix(omega: float, phi: float, kappa: float) -> np.ndarray:
        """计算旋转矩阵"""
        # R = R_omega * R_phi * R_kappa
        cos_o, sin_o = np.cos(omega), np.sin(omega)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        cos_k, sin_k = np.cos(kappa), np.sin(kappa)
        
        R = np.array([
            [cos_p*cos_k, -cos_p*sin_k, sin_p],
            cos_o*sin_k + sin_o*sin_p*cos_k, cos_o*cos_k - sin_o*sin_p*sin_k, -sin_o*cos_p,
            sin_o*sin_k - cos_o*sin_p*cos_k, sin_o*cos_k + cos_o*sin_p*sin_k, cos_o*cos_p
        ])
        
        return R


class BundleAdjustment:
    """区域网平差处理器"""
    
    @staticmethod
    def indirect_adjustment(
        control_points: Dict[str, Tuple[float, float, float]],
        tie_points: List[Dict],
        interior_orientation: Dict
    ) -> Dict[str, Any]:
        """
        间接平差法
        
        Args:
            control_points: 控制点坐标 {name: (X, Y, Z)}
            tie_points: 连接点列表
            interior_orientation: 内方位元素 {x0, y0, f}
            
        Returns:
            平差结果
        """
        # 模拟平差计算
        n_obs = len(tie_points) * 2  # 每个点两个观测值
        n_unk = 6 + len(tie_points) * 3  # 6个外方位元素 + 每个点的3个坐标
        
        # 设计矩阵
        A = np.random.randn(n_obs, n_unk) * 0.1
        L = np.random.randn(n_obs) * 0.5  # 观测值
        P = np.eye(n_obs)  # 权矩阵
        
        # 法方程求解
        N = A.T @ P @ A
        Ls = A.T @ P @ L
        
        try:
            X = np.linalg.solve(N, Ls)  # 未知数
            V = A @ X - L  # 残差
            
            # 单位权中误差
            DOF = n_obs - n_unk
            sigma0 = np.sqrt((V.T @ P @ V) / DOF)
            
            # 点位精度
            point_rmse = np.sqrt(np.mean(V**2))
            
            return {
                "success": True,
                "method": "间接平差",
                "sigma0": sigma0,
                "point_rmse": point_rmse,
                "observations": n_obs,
                "unknowns": n_unk,
                "degrees_of_freedom": DOF,
                "max_residual": np.max(np.abs(V)),
                "mean_residual": np.mean(np.abs(V))
            }
        except np.linalg.LinAlgError:
            return {"success": False, "error": "法方程求解失败"}
    
    @staticmethod
    def bundle_adjustment(
        image_observations: List[Dict],
        control_points: Dict[str, Tuple[float, float, float]],
        interior_orientation: Dict
    ) -> Dict[str, Any]:
        """
        光束法平差
        
        Args:
            image_observations: 像点观测列表
            control_points: 控制点坐标
            interior_orientation: 内方位元素
            
        Returns:
            平差结果
        """
        n_images = len(image_observations)
        n_control = len(control_points)
        
        # 模拟光束法平差迭代
        iterations = 10
        initial_error = 10.0
        final_error = 0.5
        
        errors = []
        for i in range(iterations):
            errors.append(initial_error - (initial_error - final_error) * (i + 1) / iterations)
        
        return {
            "success": True,
            "method": "光束法平差",
            "iterations": iterations,
            "initial_rmse": initial_error,
            "final_rmse": final_error,
            "images": n_images,
            "control_points": n_control,
            "converged": True,
            "point_coverage": 0.95
        }


class ResidualAnalysis:
    """残差分析处理器"""
    
    @staticmethod
    def analyze_residuals(
        residuals: np.ndarray,
        observations: np.ndarray
    ) -> Dict[str, Any]:
        """
        残差分析
        
        Args:
            residuals: 残差向量
            observations: 观测值
            
        Returns:
            残差分析报告
        """
        n = len(residuals)
        
        # 基本统计
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        max_res = np.max(np.abs(residuals))
        min_res = np.min(residuals)
        
        # RMSE
        rmse = np.sqrt(np.mean(residuals**2))
        
        # 中误差
        sigma = np.sqrt(np.sum(residuals**2) / (n - 1))
        
        # 残差分布检验 (3σ 原则)
        outliers_3sigma = np.sum(np.abs(residuals) > 3 * sigma)
        outliers_percentage = outliers_3sigma / n * 100
        
        # 正态性检验 (简化)
        skewness = np.mean(((residuals - mean_res) / sigma) ** 3)
        kurtosis = np.mean(((residuals - mean_res) / sigma) ** 4) - 3
        
        return {
            "success": True,
            "count": n,
            "mean": mean_res,
            "std": std_res,
            "max": max_res,
            "min": min_res,
            "rmse": rmse,
            "sigma": sigma,
            "outliers_3sigma": outliers_3sigma,
            "outliers_percentage": outliers_percentage,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "status": "良好" if outliers_percentage < 5 else "存在异常"
        }
    
    @staticmethod
    def generate_residual_plot_data(residuals: np.ndarray) -> Dict[str, Any]:
        """生成残差分布图数据"""
        n = len(residuals)
        
        # 直方图数据
        hist, bin_edges = np.histogram(residuals, bins=20)
        
        # 累积分布
        sorted_res = np.sort(residuals)
        cumulative = np.arange(1, n + 1) / n
        
        return {
            "histogram": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "sorted_residuals": sorted_res.tolist(),
            "cumulative": cumulative.tolist()
        }


class OutlierDetection:
    """AI 异常点检测处理器"""
    
    @staticmethod
    def dbscan_clustering(
        points: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> Dict[str, Any]:
        """
        DBSCAN 聚类检测粗差
        
        Args:
            points: 点坐标 (N x 3)
            eps: 邻域半径
            min_samples: 最小样本数
            
        Returns:
            检测结果
        """
        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            return {"success": False, "error": "需要安装 scikit-learn"}
        
        # 标准化
        points_normalized = (points - points.mean(axis=0)) / points.std(axis=0)
        
        # DBSCAN 聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_normalized)
        labels = clustering.labels_
        
        # 统计结果
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        # 异常点（噪声点）
        outlier_indices = np.where(labels == -1)[0]
        
        return {
            "success": True,
            "method": "DBSCAN",
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "outlier_percentage": n_noise / len(points) * 100,
            "outlier_indices": outlier_indices.tolist()
        }
    
    @staticmethod
    def isolation_forest(
        points: np.ndarray,
        contamination: float = 0.1
    ) -> Dict[str, Any]:
        """
        孤立森林检测异常点
        
        Args:
            points: 点坐标 (N x 3)
            contamination: 污染比例估计
            
        Returns:
            检测结果
        """
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            return {"success": False, "error": "需要安装 scikit-learn"}
        
        # 标准化
        points_normalized = (points - points.mean(axis=0)) / points.std(axis=0)
        
        # 孤立森林
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(points_normalized)
        scores = clf.score_samples(points_normalized)
        
        # 异常点
        outlier_indices = np.where(predictions == -1)[0]
        
        return {
            "success": True,
            "method": "孤立森林",
            "n_outliers": len(outlier_indices),
            "outlier_percentage": len(outlier_indices) / len(points) * 100,
            "outlier_indices": outlier_indices.tolist(),
            "anomaly_scores": scores.tolist()
        }
    
    @staticmethod
    def combined_detection(
        points: np.ndarray,
        use_dbscan: bool = True,
        use_isolation: bool = True
    ) -> Dict[str, Any]:
        """
        综合异常点检测
        
        Args:
            points: 点坐标
            use_dbscan: 是否使用 DBSCAN
            use_isolation: 是否使用孤立森林
            
        Returns:
            综合检测结果
        """
        results = {}
        all_outliers = set()
        
        if use_dbscan:
            dbscan_result = OutlierDetection.dbscan_clustering(points)
            if dbscan_result.get("success"):
                results["dbscan"] = dbscan_result
                all_outliers.update(dbscan_result.get("outlier_indices", []))
        
        if use_isolation:
            iso_result = OutlierDetection.isolation_forest(points)
            if iso_result.get("success"):
                results["isolation_forest"] = iso_result
                all_outliers.update(iso_result.get("outlier_indices", []))
        
        return {
            "success": True,
            "methods_used": list(results.keys()),
            "combined_outliers": list(all_outliers),
            "n_outliers": len(all_outliers),
            "outlier_percentage": len(all_outliers) / len(points) * 100,
            "individual_results": results
        }


# ======================================================================
# Real, workspace-driven implementation used by the module-2 plugin.
# The classes above are kept for backward compatibility, while the code
# below is the version that is actually exercised by the new plugin UI.
# ======================================================================

from dataclasses import dataclass
from typing import Sequence
import math
import tempfile
try:
    from scipy.optimize import least_squares
except Exception:
    least_squares = None


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    return arr.copy()


def _load_bgr_image(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def _camera_matrix(image: np.ndarray, focal_scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    focal = max(float(w), float(h)) * float(focal_scale)
    return np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _project(points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_3d, dtype=np.float64).reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(pts, np.asarray(rvec, dtype=np.float64), np.asarray(tvec, dtype=np.float64), k, None)
    return projected.reshape(-1, 2)


def _triangulate(k: np.ndarray, rmat: np.ndarray, tvec: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    p1 = k @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = k @ np.hstack([rmat, np.asarray(tvec, dtype=np.float64).reshape(3, 1)])
    pts4 = cv2.triangulatePoints(p1, p2, pts1.T.astype(np.float64), pts2.T.astype(np.float64)).T
    denom = np.clip(pts4[:, 3:4], 1e-12, None)
    return pts4[:, :3] / denom


def _keypoints_from_points(points: np.ndarray) -> list:
    return [tuple(map(float, pt[:2])) for pt in np.asarray(points)]


def _build_match_records(
    pts1: np.ndarray,
    pts2: np.ndarray,
    matches: Sequence[Any],
    inlier_mask: Optional[np.ndarray] = None,
) -> list:
    records = []
    inlier_mask = np.asarray(inlier_mask).astype(bool) if inlier_mask is not None else None
    for idx, match in enumerate(matches):
        records.append({
            "left": tuple(map(float, pts1[match.queryIdx])),
            "right": tuple(map(float, pts2[match.trainIdx])),
            "inlier": bool(inlier_mask[idx]) if inlier_mask is not None and idx < len(inlier_mask) else True,
            "distance": float(getattr(match, "distance", 0.0)),
            "index": int(idx),
        })
    return records


def _draw_text_block(canvas: np.ndarray, lines: Sequence[str], origin=(28, 44), color=(32, 32, 32)) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(canvas, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, color, 2, cv2.LINE_AA)
        y += 30


def _histogram_canvas(values: np.ndarray, title: str, width: int = 900, height: int = 520) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (210, 210, 210), 1)
    cv2.putText(canvas, title, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (24, 24, 24), 2, cv2.LINE_AA)
    if values.size == 0:
        _draw_text_block(canvas, ["No residuals available."])
        return canvas

    arr = np.asarray(values, dtype=np.float64).ravel()
    bins = min(24, max(8, int(math.sqrt(arr.size))))
    hist, edges = np.histogram(arr, bins=bins)
    hist_max = max(int(hist.max()), 1)
    plot_left, plot_top = 70, 90
    plot_right, plot_bottom = width - 40, height - 80
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top
    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (220, 220, 220), 1)
    bar_w = max(1, int(plot_w / len(hist)))
    for i, count in enumerate(hist):
        x1 = plot_left + i * bar_w + 2
        x2 = min(plot_right, x1 + bar_w - 4)
        bar_h = int((count / hist_max) * (plot_h - 10))
        y1 = plot_bottom - bar_h
        cv2.rectangle(canvas, (x1, y1), (x2, plot_bottom), (70, 130, 180), -1)
    mean = float(arr.mean())
    std = float(arr.std())
    _draw_text_block(
        canvas,
        [
            f"count: {arr.size}",
            f"mean: {mean:.4f}",
            f"std: {std:.4f}",
            f"min: {float(arr.min()):.4f}",
            f"max: {float(arr.max()):.4f}",
        ],
        origin=(plot_left, height - 36),
    )
    return canvas


def _summary_canvas(title: str, lines: Sequence[str], width: int = 1100, height: int = 760) -> np.ndarray:
    canvas = np.full((height, width, 3), 247, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (210, 210, 210), 1)
    cv2.rectangle(canvas, (0, 0), (width - 1, 72), (234, 240, 246), -1)
    cv2.putText(canvas, title, (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    _draw_text_block(canvas, list(lines), origin=(34, 118))
    return canvas


def _point_cloud_colors(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    depth = pts[:, 2]
    mn, mx = float(depth.min()), float(depth.max())
    if abs(mx - mn) < 1e-12:
        norm = np.zeros_like(depth)
    else:
        norm = (depth - mn) / (mx - mn)
    colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
    colors[:, 0] = np.clip((255 * (1.0 - norm)), 0, 255).astype(np.uint8)
    colors[:, 1] = np.clip((255 * norm), 0, 255).astype(np.uint8)
    colors[:, 2] = np.clip(120 + 80 * (1.0 - np.abs(norm - 0.5) * 2.0), 0, 255).astype(np.uint8)
    return colors


@dataclass
class AerialTriangulationResult:
    success: bool
    message: str
    summary: Dict[str, Any]
    left_image: Optional[np.ndarray] = None
    right_image: Optional[np.ndarray] = None
    compare_matches: Optional[list] = None
    left_keypoints: Optional[list] = None
    right_keypoints: Optional[list] = None
    points_3d: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    overview_canvas: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None
    match_mask: Optional[np.ndarray] = None
    camera_matrix: Optional[np.ndarray] = None
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None
    refined_rotation: Optional[np.ndarray] = None
    refined_translation: Optional[np.ndarray] = None


class AerialTriangulationProcessor:
    def __init__(self, focal_scale: float = 1.2):
        self.focal_scale = float(focal_scale)

    def _detect_features(self, image: np.ndarray, method: str = "SIFT", n_features: int = 2500) -> Dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        method_name = (method or "SIFT").upper()
        detector_name = method_name
        if method_name == "SIFT" and hasattr(cv2, "SIFT_create"):
            detector = cv2.SIFT_create(nfeatures=int(n_features))
            norm = cv2.NORM_L2
        else:
            detector = cv2.ORB_create(nfeatures=int(n_features))
            norm = cv2.NORM_HAMMING
            detector_name = "ORB"
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        pts = np.array([kp.pt for kp in keypoints], dtype=np.float32) if keypoints else np.empty((0, 2), dtype=np.float32)
        return {
            "algorithm": detector_name,
            "keypoints": keypoints,
            "points": pts,
            "descriptors": descriptors,
            "norm": norm,
        }

    def _match_features(
        self,
        left_desc: Optional[np.ndarray],
        right_desc: Optional[np.ndarray],
        algorithm: str = "AUTO",
        ratio: float = 0.75,
    ) -> list:
        if left_desc is None or right_desc is None or len(left_desc) == 0 or len(right_desc) == 0:
            return []
        method = (algorithm or "AUTO").upper()
        if method == "FLANN" and left_desc.dtype != np.uint8:
            matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=48))
        else:
            norm = cv2.NORM_HAMMING if left_desc.dtype == np.uint8 else cv2.NORM_L2
            matcher = cv2.BFMatcher(norm, crossCheck=False)
        raw = matcher.knnMatch(left_desc, right_desc, k=2)
        matches = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if n.distance <= 1e-12:
                continue
            if m.distance < ratio * n.distance:
                matches.append(m)
        return matches

    def relative_orientation(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        feature_method: str = "SIFT",
        matcher_method: str = "AUTO",
        n_features: int = 2500,
        ratio: float = 0.75,
        ransac_threshold: float = 1.5,
    ) -> AerialTriangulationResult:
        left = _ensure_bgr(left_image)
        right = _ensure_bgr(right_image)
        left_feat = self._detect_features(left, feature_method, n_features)
        right_feat = self._detect_features(right, feature_method, n_features)
        matches = self._match_features(left_feat["descriptors"], right_feat["descriptors"], matcher_method, ratio)

        if len(matches) < 8:
            canvas = _summary_canvas(
                "Relative Orientation",
                [
                    "status: failed",
                    "reason: not enough matches",
                    f"left features: {len(left_feat['points'])}",
                    f"right features: {len(right_feat['points'])}",
                    f"matches: {len(matches)}",
                ],
            )
            return AerialTriangulationResult(
                success=False,
                message="匹配点不足，无法进行相对定向",
                summary={
                    "left_features": len(left_feat["points"]),
                    "right_features": len(right_feat["points"]),
                    "matches": len(matches),
                    "inliers": 0,
                    "reprojection_rmse": None,
                },
                left_image=left,
                right_image=right,
                left_keypoints=_keypoints_from_points(left_feat["points"]),
                right_keypoints=_keypoints_from_points(right_feat["points"]),
                overview_canvas=canvas,
            )

        pts1 = np.float32([left_feat["points"][m.queryIdx] for m in matches])
        pts2 = np.float32([right_feat["points"][m.trainIdx] for m in matches])
        k = _camera_matrix(left, self.focal_scale)

        E, raw_mask = cv2.findEssentialMat(
            pts1,
            pts2,
            cameraMatrix=k,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=float(ransac_threshold),
        )
        if E is None or raw_mask is None:
            canvas = _summary_canvas(
                "Relative Orientation",
                [
                    "status: failed",
                    "reason: essential matrix estimation failed",
                    f"left features: {len(left_feat['points'])}",
                    f"right features: {len(right_feat['points'])}",
                    f"matches: {len(matches)}",
                ],
            )
            return AerialTriangulationResult(
                success=False,
                message="基础矩阵/本质矩阵估计失败",
                summary={
                    "left_features": len(left_feat["points"]),
                    "right_features": len(right_feat["points"]),
                    "matches": len(matches),
                    "inliers": 0,
                    "reprojection_rmse": None,
                },
                left_image=left,
                right_image=right,
                left_keypoints=_keypoints_from_points(left_feat["points"]),
                right_keypoints=_keypoints_from_points(right_feat["points"]),
                overview_canvas=canvas,
            )

        pose_inliers, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, k, mask=raw_mask)
        inlier_mask = pose_mask.ravel().astype(bool) if pose_mask is not None else np.ones(len(matches), dtype=bool)
        inlier_pts1 = pts1[inlier_mask]
        inlier_pts2 = pts2[inlier_mask]
        if len(inlier_pts1) < 4:
            inlier_pts1 = pts1
            inlier_pts2 = pts2
            inlier_mask = np.ones(len(matches), dtype=bool)

        points_3d = _triangulate(k, R, t, inlier_pts1, inlier_pts2)
        proj1 = _project(points_3d, np.zeros(3), np.zeros(3), k)
        proj2 = _project(points_3d, cv2.Rodrigues(R)[0], t.reshape(3), k)
        residual1 = proj1 - inlier_pts1
        residual2 = proj2 - inlier_pts2
        residual_norm = np.sqrt((residual1[:, 0] ** 2 + residual1[:, 1] ** 2 + residual2[:, 0] ** 2 + residual2[:, 1] ** 2) / 2.0)
        rmse = float(np.sqrt(np.mean(residual_norm ** 2))) if residual_norm.size else None
        colors = _point_cloud_colors(points_3d)
        overview = _summary_canvas(
            "Relative Orientation",
            [
                "status: success",
                f"detector: {left_feat['algorithm']}",
                f"matches: {len(matches)}",
                f"inliers: {int(inlier_mask.sum())}",
                f"inlier rate: {inlier_mask.mean():.2%}",
                f"baseline norm: {float(np.linalg.norm(t)):.6f}",
                f"reprojection rmse: {rmse:.4f} px" if rmse is not None else "reprojection rmse: n/a",
            ],
        )
        return AerialTriangulationResult(
            success=True,
            message="相对定向完成",
            summary={
                "left_features": len(left_feat["points"]),
                "right_features": len(right_feat["points"]),
                "matches": len(matches),
                "inliers": int(inlier_mask.sum()),
                "inlier_rate": float(inlier_mask.mean()),
                "reprojection_rmse": rmse,
                "baseline": float(np.linalg.norm(t)),
                "detector": left_feat["algorithm"],
                "matcher": matcher_method.upper(),
            },
            left_image=left,
            right_image=right,
            compare_matches=_build_match_records(left_feat["points"], right_feat["points"], matches, raw_mask.ravel().astype(bool)),
            left_keypoints=_keypoints_from_points(left_feat["points"]),
            right_keypoints=_keypoints_from_points(right_feat["points"]),
            points_3d=points_3d,
            colors=colors,
            overview_canvas=overview,
            residuals=np.column_stack([residual1, residual2]),
            inlier_mask=inlier_mask,
            match_mask=raw_mask.ravel().astype(bool),
            camera_matrix=k,
            rotation=R,
            translation=t.reshape(3),
        )

    def bundle_adjustment(
        self,
        orientation: AerialTriangulationResult,
        max_points: int = 120,
    ) -> AerialTriangulationResult:
        if not orientation.success or orientation.points_3d is None or orientation.camera_matrix is None:
            return AerialTriangulationResult(
                success=False,
                message="请先执行相对定向",
                summary={"status": "missing orientation"},
            )

        all_pts1 = np.asarray([m["left"] for m in orientation.compare_matches or []], dtype=np.float64)
        all_pts2 = np.asarray([m["right"] for m in orientation.compare_matches or []], dtype=np.float64)
        if len(all_pts1) == 0 or len(all_pts2) == 0:
            return AerialTriangulationResult(
                success=False,
                message="缺少可用于平差的匹配点",
                summary={"status": "no matches"},
            )
        if orientation.match_mask is not None and len(orientation.match_mask) == len(all_pts1):
            inlier_sel = np.asarray(orientation.match_mask, dtype=bool)
            pts1 = all_pts1[inlier_sel]
            pts2 = all_pts2[inlier_sel]
        else:
            pts1 = all_pts1
            pts2 = all_pts2
        if pts1 is None or pts2 is None or len(pts1) == 0:
            # fall back to the original inlier points
            if orientation.compare_matches:
                pts1 = np.array([m["left"] for m in orientation.compare_matches if m.get("inlier", True)], dtype=np.float64)
                pts2 = np.array([m["right"] for m in orientation.compare_matches if m.get("inlier", True)], dtype=np.float64)
            else:
                return AerialTriangulationResult(success=False, message="缺少可用于平差的点", summary={})

        pts1 = pts1[:max_points]
        pts2 = pts2[:max_points]
        pts3d = orientation.points_3d[: len(pts1)].copy()
        k = orientation.camera_matrix
        rvec0, _ = cv2.Rodrigues(orientation.rotation)
        tvec0 = orientation.translation.reshape(3, 1).astype(np.float64)
        before1 = _project(pts3d, np.zeros(3), np.zeros(3), k)
        before2 = _project(pts3d, rvec0, tvec0, k)
        before_rmse = float(np.sqrt(np.mean(np.concatenate([(before1 - pts1), (before2 - pts2)]) ** 2)))

        refined_rvec = rvec0.copy()
        refined_tvec = tvec0.copy()
        refined_points = pts3d.copy()
        iterations = 1
        method_used = "solvePnP+retriangulation"

        try:
            if "least_squares" in globals() and least_squares is not None:
                x0 = np.hstack([refined_rvec.ravel(), refined_tvec.ravel(), refined_points.ravel()])

                def residual_func(params: np.ndarray) -> np.ndarray:
                    rvec = params[:3]
                    tvec = params[3:6]
                    pts = params[6:].reshape(-1, 3)
                    p1 = _project(pts, np.zeros(3), np.zeros(3), k)
                    p2 = _project(pts, rvec, tvec, k)
                    return np.concatenate([(p1 - pts1).ravel(), (p2 - pts2).ravel()])

                result = least_squares(
                    residual_func,
                    x0,
                    method="trf",
                    loss="soft_l1",
                    f_scale=1.0,
                    max_nfev=80,
                )
                if result.success:
                    iterations = int(getattr(result, "nfev", 1))
                    refined_rvec = result.x[:3].reshape(3, 1)
                    refined_tvec = result.x[3:6].reshape(3, 1)
                    refined_points = result.x[6:].reshape(-1, 3)
                    method_used = "least_squares"
            else:
                ok, refined_rvec, refined_tvec = cv2.solvePnP(
                    refined_points,
                    pts2,
                    k,
                    None,
                    refined_rvec,
                    refined_tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok:
                    method_used = "solvePnP"
                    R_refined, _ = cv2.Rodrigues(refined_rvec)
                    refined_points = _triangulate(k, R_refined, refined_tvec, pts1, pts2)
        except Exception:
            method_used = "fallback"

        after1 = _project(refined_points, np.zeros(3), np.zeros(3), k)
        after2 = _project(refined_points, refined_rvec, refined_tvec, k)
        after_rmse = float(np.sqrt(np.mean(np.concatenate([(after1 - pts1), (after2 - pts2)]) ** 2)))
        residual_gain = float(before_rmse - after_rmse)
        rmat_refined, _ = cv2.Rodrigues(refined_rvec)
        colors = _point_cloud_colors(refined_points)
        overview = _summary_canvas(
            "Bundle Adjustment",
            [
                "status: success",
                f"method: {method_used}",
                f"points: {len(refined_points)}",
                f"iterations: {iterations}",
                f"before rmse: {before_rmse:.4f} px",
                f"after rmse: {after_rmse:.4f} px",
                f"improvement: {residual_gain:.4f} px",
            ],
        )

        residual_vectors = np.column_stack([(after1 - pts1), (after2 - pts2)])
        return AerialTriangulationResult(
            success=True,
            message="区域网平差完成",
            summary={
                "method": method_used,
                "iterations": iterations,
                "point_count": int(len(refined_points)),
                "before_rmse": before_rmse,
                "after_rmse": after_rmse,
                "improvement": residual_gain,
            },
            points_3d=refined_points,
            colors=colors,
            overview_canvas=overview,
            residuals=residual_vectors,
            camera_matrix=k,
            rotation=rmat_refined,
            translation=refined_tvec.reshape(3),
            refined_rotation=rmat_refined,
            refined_translation=refined_tvec.reshape(3),
        )

    def analyze_residuals(self, residuals: np.ndarray) -> Dict[str, Any]:
        arr = np.asarray(residuals, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size == 0:
            return {"success": False, "message": "no residuals"}
        norms = np.linalg.norm(arr, axis=1)
        mean = float(norms.mean())
        std = float(norms.std())
        rmse = float(np.sqrt(np.mean(norms ** 2)))
        sigma = float(np.sqrt(np.sum(norms ** 2) / max(len(norms) - 1, 1)))
        outliers_3sigma = int(np.sum(norms > 3 * sigma))
        outliers_percentage = float(outliers_3sigma / len(norms) * 100.0)
        skewness = float(np.mean(((norms - mean) / (std if std > 1e-12 else 1.0)) ** 3))
        kurtosis = float(np.mean(((norms - mean) / (std if std > 1e-12 else 1.0)) ** 4) - 3.0)
        canvas = _histogram_canvas(norms, "Residual Distribution")
        return {
            "success": True,
            "count": int(len(norms)),
            "mean": mean,
            "std": std,
            "max": float(norms.max()),
            "min": float(norms.min()),
            "rmse": rmse,
            "sigma": sigma,
            "outliers_3sigma": outliers_3sigma,
            "outliers_percentage": outliers_percentage,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "status": "良好" if outliers_percentage < 5.0 else "存在异常",
            "canvas": canvas,
            "norms": norms,
        }

    def detect_outliers(
        self,
        residuals: np.ndarray,
        points_3d: Optional[np.ndarray] = None,
        use_dbscan: bool = True,
        use_isolation: bool = True,
        eps: float = 0.8,
        min_samples: int = 5,
        contamination: float = 0.12,
    ) -> Dict[str, Any]:
        arr = np.asarray(residuals, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.size == 0:
            return {"success": False, "message": "no residuals"}
        norms = np.linalg.norm(arr, axis=1)
        features = arr.copy()
        if points_3d is not None:
            pts = np.asarray(points_3d, dtype=np.float64)
            if len(pts) >= len(features):
                pts = pts[: len(features)]
            features = np.column_stack([features[: len(pts)], pts[:, :3]])
            norms = norms[: len(pts)]
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-12)
        individual = {}
        union = set()

        if use_dbscan:
            try:
                from sklearn.cluster import DBSCAN
                labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(features)
                idx = np.where(labels == -1)[0]
                individual["dbscan"] = {
                    "n_noise": int(len(idx)),
                    "outlier_indices": idx.tolist(),
                }
                union.update(idx.tolist())
            except Exception as exc:
                individual["dbscan"] = {"error": str(exc)}

        if use_isolation:
            try:
                from sklearn.ensemble import IsolationForest
                clf = IsolationForest(contamination=float(contamination), random_state=42)
                pred = clf.fit_predict(features)
                idx = np.where(pred == -1)[0]
                individual["isolation_forest"] = {
                    "n_outliers": int(len(idx)),
                    "outlier_indices": idx.tolist(),
                }
                union.update(idx.tolist())
            except Exception as exc:
                individual["isolation_forest"] = {"error": str(exc)}

        if not individual:
            threshold = float(np.median(norms) + 2.5 * np.median(np.abs(norms - np.median(norms))))
            idx = np.where(norms > threshold)[0]
            individual["robust_threshold"] = {
                "threshold": threshold,
                "outlier_indices": idx.tolist(),
            }
            union.update(idx.tolist())

        outlier_mask = np.zeros(len(norms), dtype=bool)
        if union:
            outlier_mask[list(sorted(union))] = True
        canvas = _histogram_canvas(norms, "Outlier Detection")
        return {
            "success": True,
            "methods_used": list(individual.keys()),
            "individual_results": individual,
            "outlier_mask": outlier_mask,
            "outlier_indices": sorted(union),
            "n_outliers": int(len(union)),
            "outlier_percentage": float(len(union) / len(norms) * 100.0),
            "canvas": canvas,
        }

    def save_point_cloud(self, points: np.ndarray, prefix: str = "aerial_point_cloud") -> str:
        path = os.path.join(tempfile.gettempdir(), f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npy")
        np.save(path, np.asarray(points, dtype=np.float32))
        return path
