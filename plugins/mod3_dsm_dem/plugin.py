"""模块三：DSM / DEM 生产插件。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from core.base_interface import IPlugin
from core.event_bus import EventTopics, get_event_bus
from core.log_manager import log_manager

from .models import DsmDemSession, DsmResult, DemResult
from .processors import DsmDemProcessor
from .ui import DsmDemControlPanel
from .utils import ensure_output_dir, save_grid_to_npy, save_point_cloud_xyz, save_preview_image


class Mod3ProcessingThread(QThread):
    """模块三后台处理线程。"""

    progress = Signal(int)
    stage = Signal(str)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, plugin: "DsmDemPlugin", mode: str, config: Dict[str, Any]):
        super().__init__()
        self.plugin = plugin
        self.mode = mode
        self.config = config

    def run(self):
        try:
            result = self.plugin._execute_pipeline(self.mode, self.config, self.progress.emit, self.stage.emit)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class DsmDemPlugin(IPlugin):
    """DSM / DEM 生产模块。"""

    def __init__(self, workspace):
        super().__init__(workspace)
        self.panel: Optional[DsmDemControlPanel] = None
        self._event_bus = get_event_bus()
        self._worker: Optional[Mod3ProcessingThread] = None
        self._processor = DsmDemProcessor()
        self._session = DsmDemSession()
        self.workspace.data_added.connect(self._on_workspace_data_changed)
        self.workspace.data_updated.connect(self._on_workspace_data_changed)

    def plugin_info(self) -> Dict[str, Any]:
        return {
            "name": "DSM/DEM生产",
            "group": "模块",
            "version": "2.0.0",
            "description": "基于双像密集匹配生成相对 DSM，并通过传统地面滤波生成相对 DEM。",
        }

    def get_ui_panel(self) -> QWidget:
        if self.panel is None:
            self.panel = DsmDemControlPanel()
            self.panel.refresh_requested.connect(self.refresh_workspace_images)
            self.panel.browse_output_requested.connect(self._browse_output_dir)
            self.panel.run_dsm_requested.connect(self._run_dsm)
            self.panel.run_dem_requested.connect(self._run_dem)
            self.panel.run_pipeline_requested.connect(self._run_pipeline)
            self.panel.show_dsm_requested.connect(self._show_latest_dsm)
            self.panel.show_dem_requested.connect(self._show_latest_dem)
            self.panel.show_ground_mask_requested.connect(self._show_ground_mask)
            self.panel.show_hillshade_compare_requested.connect(self._show_hillshade_compare)
            self.panel.export_requested.connect(self._export_latest_results)
            self.refresh_workspace_images()
            self._sync_ai_model_state()
            self._update_session_summary()
        return self.panel

    def execute(self, *args, **kwargs):
        mode = kwargs.get("mode", "pipeline")
        config = kwargs.get("config") or self._collect_config()
        try:
            result = self._execute_pipeline(mode, config, None, None)
            return {
                "success": True,
                "message": result.get("message", "完成"),
                "result_name": result.get("result_name"),
                "summary": result.get("summary"),
            }
        except Exception as exc:
            log_manager.error(f"模块三执行失败: {exc}")
            return {"success": False, "message": str(exc)}

    def refresh_workspace_images(self):
        if self.panel is None:
            return
        images = self.workspace.get_all_images()
        self.panel.set_workspace_images(list(images.keys()), images)
        self._sync_ai_model_state()

    def on_activate(self):
        self.refresh_workspace_images()
        self._update_session_summary()

    def _collect_config(self) -> Dict[str, Any]:
        if self.panel is None:
            return {}
        return self.panel.get_config()

    def _run_dsm(self):
        self._start_worker("dsm")

    def _run_dem(self):
        self._start_worker("dem")

    def _run_pipeline(self):
        self._start_worker("pipeline")

    def _start_worker(self, mode: str):
        if self.panel is None:
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(self.panel, "提示", "当前已有任务在运行，请等待完成。")
            return

        config = self._collect_config()
        self.panel.set_busy(True)
        self.panel.set_progress(0)
        self.panel.set_status("准备执行")
        self.panel.set_stage("输入检查")

        self._worker = Mod3ProcessingThread(self, mode, config)
        self._worker.progress.connect(self.panel.set_progress)
        self._worker.stage.connect(self.panel.set_stage)
        self._worker.completed.connect(self._on_worker_completed)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

    def _on_worker_completed(self, result: Dict[str, Any]):
        if self.panel is not None:
            self.panel.set_busy(False)
            self.panel.set_progress(100)
            self.panel.set_status(result.get("message", "完成"))
        self._update_session_summary()

    def _on_worker_failed(self, error: str):
        if self.panel is not None:
            self.panel.set_busy(False)
            self.panel.set_status(f"执行失败: {error}")
            QMessageBox.warning(self.panel, "模块三执行失败", error)
        log_manager.error(f"模块三执行失败: {error}")

    def _execute_pipeline(self, mode: str, config: Dict[str, Any], progress_cb=None, stage_cb=None) -> Dict[str, Any]:
        mode = (mode or "pipeline").lower()
        if mode not in {"dsm", "dem", "pipeline"}:
            mode = "pipeline"

        if mode in {"dsm", "pipeline"}:
            dsm_result = self._processor.generate_dsm(config, progress_cb, stage_cb)
            self._session.dsm_result = dsm_result
            self._publish_dsm_result(dsm_result, config)
            if mode == "dsm":
                self._update_session_summary()
                return {
                    "success": True,
                    "message": dsm_result.message,
                    "result_name": dsm_result.result_name,
                    "summary": self._session.summary_lines(),
                }

        if self._session.dsm_result is None or not self._session.dsm_result.success:
            raise ValueError("没有可用于生成 DEM 的 DSM 结果。")

        dem_result = self._processor.generate_dem(self._session.dsm_result, config, progress_cb, stage_cb)
        self._session.dem_result = dem_result
        self._publish_dem_result(dem_result, config)
        self._update_session_summary()
        return {
            "success": True,
            "message": dem_result.message if mode != "dsm" else self._session.dsm_result.message,
            "result_name": dem_result.result_name,
            "summary": self._session.summary_lines(),
        }

    def _publish_dsm_result(self, result: DsmResult, config: Dict[str, Any]):
        output_dir = ensure_output_dir(config.get("output_dir"))
        disparity_name = f"{result.result_name}_视差图"
        hillshade_name = f"{result.result_name}_hillshade"

        disparity_path = save_preview_image(os.path.join(output_dir, f"{disparity_name}.png"), result.disparity_preview)
        hillshade_path = save_preview_image(os.path.join(output_dir, f"{hillshade_name}.png"), result.hillshade)
        grid_path = save_grid_to_npy(os.path.join(output_dir, f"{result.result_name}.npy"), result.dsm_grid)
        point_cloud_path = None
        if result.point_cloud is not None and len(result.point_cloud) > 0:
            point_cloud_path = save_point_cloud_xyz(os.path.join(output_dir, f"{result.result_name}_稠密点云.xyz"), result.point_cloud)
            self.workspace.add_pointcloud(f"{result.result_name}_稠密点云", point_cloud_path)

        self.workspace.add_processed_image(disparity_name, result.disparity_preview)
        self.workspace.add_processed_image(hillshade_name, result.hillshade)

        result.disparity_path = disparity_path
        result.hillshade_path = hillshade_path
        result.grid_path = grid_path
        result.point_cloud_path = point_cloud_path
        result.workspace_disparity_path = self.workspace.get_processed_image(disparity_name)
        result.workspace_hillshade_path = self.workspace.get_processed_image(hillshade_name)

        self._event_bus.publish(
            EventTopics.TOPIC_IMAGE_UPDATED,
            {
                "name": hillshade_name,
                "path": result.workspace_hillshade_path or hillshade_path,
                "title": f"DSM 结果: {result.result_name}",
                "summary": result.summary_text(),
                "size_text": result.size_text(),
                "kind": "dsm_preview",
            },
        )
        self._publish_surface_request(result.result_name, result.dsm_grid, result.valid_mask, result.z_min, result.z_max, "DSM 表面")
        if point_cloud_path:
            log_manager.info(f"模块三：DSM 稠密点云已输出 {point_cloud_path}")
        log_manager.info(
            f"模块三：DSM 完成，视差有效率 {result.valid_ratio:.2%}，高程范围 {result.z_min:.3f} ~ {result.z_max:.3f}"
        )

    def _publish_dem_result(self, result: DemResult, config: Dict[str, Any]):
        output_dir = ensure_output_dir(config.get("output_dir"))
        hillshade_name = f"{result.result_name}_hillshade"
        mask_name = f"{result.result_name}_地面掩膜"

        hillshade_path = save_preview_image(os.path.join(output_dir, f"{hillshade_name}.png"), result.hillshade)
        mask_path = save_preview_image(os.path.join(output_dir, f"{mask_name}.png"), result.ground_mask_preview)
        grid_path = save_grid_to_npy(os.path.join(output_dir, f"{result.result_name}.npy"), result.dem_grid)

        self.workspace.add_processed_image(hillshade_name, result.hillshade)
        self.workspace.add_processed_image(mask_name, result.ground_mask_preview)
        self.workspace.set_dem(result.result_name, grid_path)

        result.hillshade_path = hillshade_path
        result.mask_path = mask_path
        result.grid_path = grid_path
        result.workspace_hillshade_path = self.workspace.get_processed_image(hillshade_name)
        result.workspace_mask_path = self.workspace.get_processed_image(mask_name)

        self._event_bus.publish(
            EventTopics.TOPIC_IMAGE_UPDATED,
            {
                "name": hillshade_name,
                "path": result.workspace_hillshade_path or hillshade_path,
                "title": f"DEM 结果: {result.result_name}",
                "summary": result.summary_text(),
                "size_text": result.size_text(),
                "kind": "dem",
            },
        )
        self._publish_surface_request(result.result_name, result.dem_grid, result.valid_mask, result.z_min, result.z_max, "DEM 表面")
        log_manager.info(
            f"模块三：DEM 完成，地面比例 {result.ground_ratio:.2%}，高程范围 {result.z_min:.3f} ~ {result.z_max:.3f}"
        )

    def _publish_surface_request(self, name: str, z_grid, mask, z_min: float, z_max: float, title: str):
        payload = {
            "type": "surface_grid",
            "name": name,
            "z_grid": z_grid,
            "mask": mask,
            "meta": {
                "mode": "relative_height",
                "z_min": float(z_min),
                "z_max": float(z_max),
            },
        }
        self._event_bus.publish(
            EventTopics.TOPIC_VIEW_3D_REQUEST,
            {
                "name": name,
                "data_type": "surface_grid",
                "payload": payload,
                "kind": "surface",
                "grid": z_grid,
                "title": title,
            },
        )

    def _show_latest_dsm(self):
        if self._session.dsm_result is None or not self._session.dsm_result.success:
            if self.panel is not None:
                QMessageBox.information(self.panel, "提示", "当前还没有可显示的 DSM 结果。")
            return
        result = self._session.dsm_result
        self._publish_surface_request(result.result_name, result.dsm_grid, result.valid_mask, result.z_min, result.z_max, "DSM 表面")

    def _show_latest_dem(self):
        if self._session.dem_result is None or not self._session.dem_result.success:
            if self.panel is not None:
                QMessageBox.information(self.panel, "提示", "当前还没有可显示的 DEM 结果。")
            return
        result = self._session.dem_result
        self._publish_surface_request(result.result_name, result.dem_grid, result.valid_mask, result.z_min, result.z_max, "DEM 表面")

    def _show_ground_mask(self):
        if self._session.dem_result is None or not self._session.dem_result.success:
            if self.panel is not None:
                QMessageBox.information(self.panel, "提示", "当前还没有地面掩膜结果。")
            return
        if self._session.dsm_result is None:
            return
        left_path = self.workspace.get_image(self._session.dsm_result.left_name)
        self._event_bus.publish(
            EventTopics.TOPIC_VIEW_COMPARE_REQUEST,
            {
                "left": {"image": left_path, "name": self._session.dsm_result.left_name},
                "right": {"image": self._session.dem_result.ground_mask_preview, "name": "地面掩膜"},
                "title": "原始左图 vs 地面掩膜",
                "sync": True,
            },
        )

    def _show_hillshade_compare(self):
        if self._session.dsm_result is None or not self._session.dsm_result.success:
            if self.panel is not None:
                QMessageBox.information(self.panel, "提示", "当前还没有可对比的 DSM 结果。")
            return
        if self._session.dem_result is None or not self._session.dem_result.success:
            if self.panel is not None:
                QMessageBox.information(self.panel, "提示", "当前还没有可对比的 DEM 结果。")
            return
        self._event_bus.publish(
            EventTopics.TOPIC_VIEW_COMPARE_REQUEST,
            {
                "left": {
                    "image": self._session.dsm_result.hillshade,
                    "name": f"{self._session.dsm_result.result_name}_hillshade",
                },
                "right": {
                    "image": self._session.dem_result.hillshade,
                    "name": f"{self._session.dem_result.result_name}_hillshade",
                },
                "title": "DSM hillshade vs DEM hillshade",
                "sync": True,
            },
        )

    def _export_latest_results(self):
        if self.panel is None:
            return
        lines = []
        if self._session.dsm_result and self._session.dsm_result.success:
            lines.append(f"DSM 网格: {self._session.dsm_result.grid_path or '未生成'}")
            lines.append(f"DSM 视差图: {self._session.dsm_result.disparity_path or '未生成'}")
            lines.append(f"DSM hillshade: {self._session.dsm_result.hillshade_path or '未生成'}")
            lines.append(f"DSM 点云: {self._session.dsm_result.point_cloud_path or '未生成'}")
        if self._session.dem_result and self._session.dem_result.success:
            lines.append(f"DEM 网格: {self._session.dem_result.grid_path or '未生成'}")
            lines.append(f"DEM hillshade: {self._session.dem_result.hillshade_path or '未生成'}")
            lines.append(f"地面掩膜: {self._session.dem_result.mask_path or '未生成'}")
        if not lines:
            QMessageBox.information(self.panel, "提示", "当前还没有可以导出的模块三结果。")
            return
        self.panel.set_result_info("\n".join(lines))
        self.panel.set_status("已整理当前导出路径")

    def _browse_output_dir(self):
        if self.panel is None:
            return
        current = self.panel.output_dir_text()
        directory = QFileDialog.getExistingDirectory(self.panel, "选择模块三输出目录", current or str(Path.cwd()))
        if directory:
            self.panel.set_output_dir(directory)

    def _sync_ai_model_state(self):
        if self.panel is None:
            return
        model_path = Path(__file__).with_name("models").joinpath("pointnet_ground.onnx")
        self.panel.set_ai_model_available(model_path.exists(), str(model_path))

    def _update_session_summary(self):
        if self.panel is None:
            return
        self.panel.set_result_info("\n".join(self._session.summary_lines()))

    def _on_workspace_data_changed(self, data_type, name, data):
        if data_type == "images":
            self.refresh_workspace_images()
