"""模块三数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class DsmResult:
    success: bool
    message: str
    left_name: str = ""
    right_name: str = ""
    result_name: str = ""
    disparity: Optional[np.ndarray] = None
    disparity_preview: Optional[np.ndarray] = None
    dsm_grid: Optional[np.ndarray] = None
    hillshade: Optional[np.ndarray] = None
    point_cloud: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    valid_ratio: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    elapsed_text: str = "-"
    mode_text: str = "相对高程，不代表绝对地理高程"
    grid_path: Optional[str] = None
    disparity_path: Optional[str] = None
    hillshade_path: Optional[str] = None
    point_cloud_path: Optional[str] = None
    workspace_disparity_path: Optional[str] = None
    workspace_hillshade_path: Optional[str] = None

    def summary_text(self) -> str:
        return (
            f"{self.mode_text} · 视差有效率 {self.valid_ratio:.2%} · "
            f"高程范围 {self.z_min:.3f} ~ {self.z_max:.3f} · 耗时 {self.elapsed_text}"
        )

    def size_text(self) -> str:
        if self.dsm_grid is None:
            return "-"
        h, w = self.dsm_grid.shape[:2]
        return f"{w} x {h}"


@dataclass
class DemResult:
    success: bool
    message: str
    result_name: str = ""
    dem_grid: Optional[np.ndarray] = None
    ground_mask: Optional[np.ndarray] = None
    ground_mask_preview: Optional[np.ndarray] = None
    hillshade: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    ground_ratio: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    elapsed_text: str = "-"
    mode_text: str = "相对高程，不代表绝对地理高程"
    grid_path: Optional[str] = None
    hillshade_path: Optional[str] = None
    mask_path: Optional[str] = None
    workspace_hillshade_path: Optional[str] = None
    workspace_mask_path: Optional[str] = None

    def summary_text(self) -> str:
        return (
            f"{self.mode_text} · 地面比例 {self.ground_ratio:.2%} · "
            f"高程范围 {self.z_min:.3f} ~ {self.z_max:.3f} · 耗时 {self.elapsed_text}"
        )

    def size_text(self) -> str:
        if self.dem_grid is None:
            return "-"
        h, w = self.dem_grid.shape[:2]
        return f"{w} x {h}"


@dataclass
class DsmDemSession:
    dsm_result: Optional[DsmResult] = None
    dem_result: Optional[DemResult] = None

    def summary_lines(self) -> List[str]:
        lines: List[str] = ["模块三当前会话"]
        if self.dsm_result and self.dsm_result.success:
            lines.extend(
                [
                    f"DSM: {self.dsm_result.result_name}",
                    f"左/右影像: {self.dsm_result.left_name} / {self.dsm_result.right_name}",
                    f"视差有效率: {self.dsm_result.valid_ratio:.2%}",
                    f"DSM 尺寸: {self.dsm_result.size_text()}",
                    f"DSM 高程范围: {self.dsm_result.z_min:.3f} ~ {self.dsm_result.z_max:.3f}",
                    f"DSM 耗时: {self.dsm_result.elapsed_text}",
                    f"说明: {self.dsm_result.mode_text}",
                ]
            )
        else:
            lines.append("DSM: 暂无结果")

        if self.dem_result and self.dem_result.success:
            lines.extend(
                [
                    f"DEM: {self.dem_result.result_name}",
                    f"DEM 尺寸: {self.dem_result.size_text()}",
                    f"DEM 高程范围: {self.dem_result.z_min:.3f} ~ {self.dem_result.z_max:.3f}",
                    f"地面比例: {self.dem_result.ground_ratio:.2%}",
                    f"DEM 耗时: {self.dem_result.elapsed_text}",
                    f"说明: {self.dem_result.mode_text}",
                ]
            )
        else:
            lines.append("DEM: 暂无结果")
        return lines
