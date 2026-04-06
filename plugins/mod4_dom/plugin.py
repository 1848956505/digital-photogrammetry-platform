"""模块四 DOM 生产插件。"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from core.base_interface import IPlugin
from core.event_bus import EventTopics, get_event_bus
from core.log_manager import log_manager

from plugins.mod4_dom.algorithms.color_balance import align_mean_brightness, match_histogram_images
from plugins.mod4_dom.algorithms.export import save_geotiff_if_possible, save_png, save_tiff
from plugins.mod4_dom.algorithms.mosaic import (
    crop_valid_region,
    has_geo_metadata,
    load_images_from_workspace_entries,
    mosaic_with_feature_matching,
    mosaic_with_georef,
)
from plugins.mod4_dom.algorithms.seam import compose_layers
from plugins.mod4_dom.ui import DomControlPanel


class DomProcessingThread(QThread):
    """后台 DOM 处理线程。"""

    progress = Signal(int)
    stage = Signal(str)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, plugin: "DomPlugin", config: Dict[str, Any]):
        super().__init__()
        self.plugin = plugin
        self.config = config

    def run(self):
        try:
            result = self.plugin._execute_pipeline(self.config, self.progress.emit, self.stage.emit)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class DomPlugin(IPlugin):
    """DOM 生产插件。"""

    def __init__(self, workspace):
        super().__init__(workspace)
        self.panel: Optional[DomControlPanel] = None
        self._worker: Optional[DomProcessingThread] = None
        self._event_bus = get_event_bus()
        self.workspace.data_added.connect(self._on_workspace_data_changed)
        self.workspace.data_updated.connect(self._on_workspace_data_changed)

    def plugin_info(self) -> Dict[str, Any]:
        return {
            "name": "DOM生产",
            "group": "模块",
            "version": "2.0.0",
            "description": "多图镶嵌、匀色、融合与导出，生成 DOM 结果",
        }

    def get_ui_panel(self) -> QWidget:
        if self.panel is None:
            self.panel = DomControlPanel()
            self.panel.refresh_requested.connect(self.refresh_workspace_images)
            self.panel.generate_requested.connect(self._generate_dom)
            self.panel.browse_export_requested.connect(self._browse_export_path)
            self.refresh_workspace_images()
        return self.panel

    def execute(self, *args, **kwargs):
        """同步执行入口，返回结构化状态对象。"""
        config = kwargs.get("config") or {}
        try:
            result = self._execute_pipeline(config, None)
            return {
                "success": True,
                "message": result.get("message", "完成"),
                "result_name": result.get("result_name"),
                "export_path": result.get("export_path"),
            }
        except Exception as exc:
            log_manager.error(f"DOM 执行失败: {exc}")
            return {"success": False, "message": str(exc)}

    def refresh_workspace_images(self):
        if self.panel is None:
            return
        images = self.workspace.get("images", {})
        names = list(images.keys())
        self.panel.set_images(names)
        self.panel.set_status(f"已加载 {len(names)} 张原始影像")
        self.panel.set_stage("就绪")

    def _on_workspace_data_changed(self, data_type=None, data_id=None, data=None):
        if data_type in {"images", "processed_images", "dom", "dem"}:
            self.refresh_workspace_images()

    def _browse_export_path(self):
        if self.panel is None:
            return
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_name = os.path.join(root_dir, "output", "dom_result.png")
        path, _ = QFileDialog.getSaveFileName(
            self.panel,
            "选择 DOM 导出路径",
            default_name,
            "PNG (*.png);;TIFF (*.tif *.tiff);;GeoTIFF (*.tif *.tiff)",
        )
        if path:
            self.panel.set_export_path(path)

    def _generate_dom(self):
        if self.panel is None:
            return
        config = self.panel.get_config()
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self.panel, "提示", "DOM 生成正在进行中，请稍候。")
            return

        log_manager.info(
            "DOM 生产开始: "
            f"输入={'全部影像' if config.get('auto_use_all', True) else '手动选择'}，"
            f"镶嵌模式={config.get('mosaic_mode', 'auto')}，"
            f"特征算法={config.get('feature_method', 'ORB')}，"
            f"匀色={'启用' if config.get('enable_color_balance', True) else '关闭'}，"
            f"融合={config.get('blend_method', 'feather')}，"
            f"导出={'启用' if config.get('export_enabled', False) else '关闭'}"
        )
        log_manager.debug("DOM 配置摘要:\n" + self._format_config_summary(config))
        self.panel.set_busy(True)
        self.panel.set_status("正在启动 DOM 生产流程...")
        self.panel.set_stage("准备中")
        self._worker = DomProcessingThread(self, config)
        self._worker.progress.connect(self.panel.set_progress)
        self._worker.stage.connect(self.panel.set_stage)
        self._worker.completed.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_finished(self, result: Dict[str, Any]):
        if self.panel is not None:
            self.panel.set_busy(False)
            self.panel.set_progress(100)
            self.panel.set_stage("完成")
            self.panel.set_status(result.get("message", "DOM 生成完成"))
            result_text = self._format_result_summary(result)
            self.panel.set_result_info(result_text)
        log_manager.info(
            "DOM 生产完成: "
            f"{result.get('result_name', '-')} | "
            f"尺寸 {result.get('size_text', '-')} | "
            f"耗时 {result.get('elapsed_text', '-')} | "
            f"导出 {result.get('export_path') or '未导出'}"
        )

    def _on_worker_error(self, message: str):
        if self.panel is not None:
            self.panel.set_busy(False)
            self.panel.set_stage("失败")
            self.panel.set_status(f"处理失败: {message}")
            QMessageBox.critical(self.panel, "DOM 生成失败", message)
        log_manager.error(f"DOM 生成失败: {message}")

    def _collect_input_images(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        images = self.workspace.get("images", {})
        if not images:
            raise ValueError("Workspace 中没有可用的原始影像")

        auto_all = config.get("auto_use_all", True)
        selected_names = config.get("selected_names", [])

        if auto_all or not selected_names:
            names = list(images.keys())
        else:
            names = [name for name in selected_names if name in images]

        if len(names) < 2:
            raise ValueError("DOM 生产至少需要两张影像")

        entries = []
        for name in names:
            item = images.get(name)
            if not item or "path" not in item:
                raise ValueError(f"影像 {name} 缺少有效路径")
            entries.append({"name": name, "path": item["path"]})
        return entries

    def _validate_inputs(self, entries: List[Dict[str, Any]]):
        for entry in entries:
            if not entry.get("path"):
                raise ValueError(f"影像 {entry.get('name')} 路径为空")
            if not os.path.exists(entry["path"]):
                raise ValueError(f"影像不存在: {entry['path']}")
        return True

    def _detect_mosaic_mode(self, entries: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
        mode = config.get("mosaic_mode", "auto")
        if mode == "geo_first":
            return "georef" if has_geo_metadata(entries) else "feature"
        if mode == "feature_first":
            return "feature"
        return "georef" if has_geo_metadata(entries) else "feature"

    def _format_config_summary(self, config: Dict[str, Any]) -> str:
        input_mode = "Workspace 全部影像" if config.get("auto_use_all", True) else "手动选择"
        mosaic_mode = config.get("mosaic_mode", "auto")
        feature_method = config.get("feature_method", "ORB")
        color_balance = "启用" if config.get("enable_color_balance", True) else "关闭"
        color_method = config.get("color_method", "mean")
        blend_method = config.get("blend_method", "feather")
        export_enabled = "启用" if config.get("export_enabled", False) else "关闭"
        export_format = config.get("export_format", "PNG")
        export_path = config.get("export_path") or "自动路径"
        return (
            f"输入来源: {input_mode}\n"
            f"镶嵌模式: {mosaic_mode}\n"
            f"特征算法: {feature_method}\n"
            f"匀色: {color_balance} / {color_method}\n"
            f"融合: {blend_method}\n"
            f"导出: {export_enabled} / {export_format}\n"
            f"导出路径: {export_path}"
        )

    def _format_result_summary(self, result: Dict[str, Any]) -> str:
        lines = [
            f"结果: {result.get('result_name', '-')}",
            f"尺寸: {result.get('size_text', '-')}",
            f"耗时: {result.get('elapsed_text', '-')}",
            f"导出: {result.get('export_path') or '未导出'}",
        ]
        processed_path = result.get("processed_path")
        if processed_path:
            lines.append(f"工作区路径: {processed_path}")
        return "\n".join(lines)

    def _preprocess_images(self, images: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        loaded = load_images_from_workspace_entries(images)
        if len(loaded) < 2:
            raise ValueError("有效输入影像不足")

        if not config.get("enable_color_balance", True):
            return loaded

        method = config.get("color_method", "mean")
        reference_index = 0 if config.get("use_reference", True) else None

        if method == "histogram":
            balanced = match_histogram_images(loaded, reference_index=reference_index)
        else:
            balanced = align_mean_brightness(loaded, reference_index=reference_index)

        for idx, item in enumerate(loaded):
            item["image"] = balanced[idx]
        return loaded

    def _mosaic_images(self, images: List[Dict[str, Any]], config: Dict[str, Any], progress_cb=None):
        mode = self._detect_mosaic_mode(images, config)
        if mode == "georef":
            try:
                return mosaic_with_georef(images, progress_cb=progress_cb)
            except Exception as exc:
                log_manager.warning(f"地理镶嵌不可用，回退到特征匹配路径: {exc}")
        return mosaic_with_feature_matching(
            images,
            feature_method=config.get("feature_method", "ORB"),
            progress_cb=progress_cb,
        )

    def _blend_images(self, mosaic_result: Dict[str, Any], config: Dict[str, Any]):
        if mosaic_result.get("image") is not None and mosaic_result.get("mask") is None:
            return mosaic_result["image"], None

        layers = mosaic_result.get("layers")
        if not layers:
            return mosaic_result["image"], mosaic_result.get("mask")

        blend_method = config.get("blend_method", "feather")
        final_image, final_mask = compose_layers(
            layers,
            method=blend_method,
            feather_radius=int(config.get("feather_radius", 15)),
        )
        return final_image, final_mask

    def _update_workspace_image(self, image, process_type: str = "DOM"):
        timestamp = datetime.now().strftime("%H%M%S_%f")
        result_name = f"{process_type}_{timestamp}"
        self.workspace.add_processed_image(result_name, image)
        processed_path = self.workspace.get_processed_image(result_name)
        self.workspace.set_dom(result_name, processed_path)
        log_manager.info(f"DOM 结果已写入 workspace: {result_name}")
        return result_name, processed_path

    def _ensure_export_path(self, config: Dict[str, Any], export_format: str) -> str:
        export_path = config.get("export_path") or ""
        if export_path:
            return export_path

        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(root_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        ext = ".png" if export_format == "PNG" else ".tif"
        if export_format == "TIFF":
            ext = ".tif"
        elif export_format == "GEOTIFF":
            ext = ".tif"
        return os.path.join(output_dir, f"dom_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")

    def _export_if_needed(self, image, mosaic_result: Dict[str, Any], config: Dict[str, Any]):
        if not config.get("export_enabled", False):
            return None

        export_format = (config.get("export_format") or "PNG").upper()
        export_path = self._ensure_export_path(config, export_format)
        export_dir = os.path.dirname(export_path)
        if export_dir and not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        metadata = mosaic_result.get("metadata", {})
        if export_format == "TIFF":
            save_tiff(image, export_path)
        elif export_format == "GEOTIFF":
            if not save_geotiff_if_possible(image, export_path, metadata=metadata):
                save_tiff(image, export_path)
        else:
            save_png(image, export_path)
        return export_path

    def _emit_stage(self, stage_cb, text: str):
        log_manager.info(f"DOM 阶段: {text}")
        if stage_cb:
            stage_cb(text)

    def _execute_pipeline(self, config: Dict[str, Any], progress_cb=None, stage_cb=None) -> Dict[str, Any]:
        self._emit_stage(stage_cb, "读取输入影像")
        entries = self._collect_input_images(config)
        self._validate_inputs(entries)

        if progress_cb:
            progress_cb(5)

        self._emit_stage(stage_cb, "执行匀色预处理")
        loaded = self._preprocess_images(entries, config)
        if progress_cb:
            progress_cb(25)

        self._emit_stage(stage_cb, "执行镶嵌计算")
        mosaic_result = self._mosaic_images(loaded, config, progress_cb=progress_cb)
        if progress_cb:
            progress_cb(70)

        self._emit_stage(stage_cb, "执行重叠区融合")
        blended_image, blended_mask = self._blend_images(mosaic_result, config)
        if blended_mask is not None:
            cropped_image, cropped_mask = crop_valid_region(blended_image, blended_mask)
        else:
            cropped_image, cropped_mask = blended_image, None

        if progress_cb:
            progress_cb(85)

        self._emit_stage(stage_cb, "写回 Workspace")
        result_name, processed_path = self._update_workspace_image(cropped_image, "DOM")

        self._emit_stage(stage_cb, "导出文件")
        export_path = self._export_if_needed(cropped_image, mosaic_result, config)
        if export_path and os.path.exists(export_path):
            self.workspace.set_dom(result_name, export_path)

        if progress_cb:
            progress_cb(100)

        h, w = cropped_image.shape[:2]
        message = "DOM 生产完成"
        if export_path:
            message = f"DOM 生产完成并导出到 {export_path}"
        elapsed_text = mosaic_result.get("elapsed_text", "-")
        display_path = export_path if export_path and os.path.exists(export_path) else processed_path
        display_title = f"DOM 结果: {result_name}"
        display_summary = f"{w} x {h} · 耗时 {elapsed_text}"
        if export_path:
            display_summary += f" · 导出 {export_path}"
        self._event_bus.publish(
            EventTopics.TOPIC_IMAGE_UPDATED,
            {
                "name": result_name,
                "path": display_path,
                "processed_path": processed_path,
                "export_path": export_path,
                "kind": "dom",
                "title": display_title,
                "summary": display_summary,
                "size_text": f"{w} x {h}",
                "elapsed_text": elapsed_text,
            },
        )
        log_manager.info(
            "DOM 生产结果: "
            f"{result_name} / {w}x{h} / "
            f"{elapsed_text} / "
            f"{export_path or '未导出'}"
        )
        return {
            "success": True,
            "message": message,
            "result_name": result_name,
            "processed_path": processed_path,
            "export_path": export_path,
            "size_text": f"{w} x {h}",
            "elapsed_text": elapsed_text,
        }
