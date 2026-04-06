"""模块四 DOM 控制面板。"""
from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class DomControlPanel(QWidget):
    """DOM 生产控制面板。"""

    refresh_requested = Signal()
    generate_requested = Signal()
    browse_export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("DOM 生产")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976d2;")
        layout.addWidget(title)

        intro = QLabel(
            "当前版本支持多图镶嵌、匀色、融合与导出。"
            "建议先保持“自动使用全部影像”和“自动判断模式”，"
            "再按数据质量逐步调整匀色与融合参数。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #555; padding: 4px 6px; background: #f6f8fb; border-radius: 4px;")
        layout.addWidget(intro)

        input_group = QGroupBox("输入影像")
        input_layout = QVBoxLayout(input_group)
        self.workspace_all_check = QCheckBox("自动使用 Workspace 中全部影像")
        self.workspace_all_check.setChecked(True)
        self.workspace_all_check.setToolTip("勾选后将直接使用 Workspace.images 中的全部原始影像。")
        input_layout.addWidget(self.workspace_all_check)

        self.image_count_label = QLabel("当前已载入 0 张影像")
        self.image_count_label.setStyleSheet("color: #666;")
        input_layout.addWidget(self.image_count_label)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.image_list.setMinimumHeight(150)
        self.image_list.setToolTip("取消上方自动选择后，可在此手动挑选用于 DOM 生产的影像。")
        input_layout.addWidget(self.image_list)

        input_btns = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新列表")
        self.select_all_btn = QPushButton("全选")
        self.select_none_btn = QPushButton("全不选")
        input_btns.addWidget(self.refresh_btn)
        input_btns.addWidget(self.select_all_btn)
        input_btns.addWidget(self.select_none_btn)
        input_layout.addLayout(input_btns)
        layout.addWidget(input_group)

        mosaic_group = QGroupBox("镶嵌设置")
        mosaic_layout = QFormLayout(mosaic_group)
        self.mosaic_mode = QComboBox()
        self.mosaic_mode.addItems(["自动判断", "地理参考优先", "特征匹配优先"])
        self.mosaic_mode.setToolTip("自动判断会优先识别地理参考，失败后回退到特征匹配。")
        mosaic_layout.addRow("模式", self.mosaic_mode)

        self.feature_method = QComboBox()
        self.feature_method.addItems(["ORB", "SIFT"])
        self.feature_method.setToolTip("无地理参考时用于图像配准的特征提取方法。")
        mosaic_layout.addRow("特征算法", self.feature_method)
        layout.addWidget(mosaic_group)

        color_group = QGroupBox("色彩调整")
        color_layout = QFormLayout(color_group)
        self.enable_color_balance = QCheckBox("启用匀色")
        self.enable_color_balance.setChecked(True)
        self.enable_color_balance.setToolTip("先对单张影像做亮度或直方图均衡，再进入镶嵌流程。")
        color_layout.addRow("", self.enable_color_balance)

        self.color_method = QComboBox()
        self.color_method.addItems(["平均亮度对齐", "直方图匹配"])
        self.color_method.setToolTip("平均亮度更稳，直方图匹配更适合色差较大的相邻影像。")
        color_layout.addRow("匀色方法", self.color_method)

        self.use_reference = QCheckBox("使用参考影像")
        self.use_reference.setChecked(True)
        self.use_reference.setToolTip("勾选后会以首张影像或平均参考作为匀色目标。")
        color_layout.addRow("", self.use_reference)
        layout.addWidget(color_group)

        blend_group = QGroupBox("融合设置")
        blend_layout = QFormLayout(blend_group)
        self.blend_method = QComboBox()
        self.blend_method.addItems(["羽化", "加权平均"])
        self.blend_method.setToolTip("羽化更平滑，加权平均更直接。")
        blend_layout.addRow("融合方式", self.blend_method)
        self.feather_radius = QSpinBox()
        self.feather_radius.setRange(1, 80)
        self.feather_radius.setValue(15)
        self.feather_radius.setToolTip("仅在羽化模式下生效，数值越大，过渡越平缓。")
        blend_layout.addRow("羽化半径", self.feather_radius)
        layout.addWidget(blend_group)

        export_group = QGroupBox("导出设置")
        export_layout = QFormLayout(export_group)
        self.export_enabled = QCheckBox("导出到文件")
        self.export_enabled.setChecked(True)
        self.export_enabled.setToolTip("关闭后只写回 Workspace，不额外生成磁盘文件。")
        export_layout.addRow("", self.export_enabled)

        self.export_format = QComboBox()
        self.export_format.addItems(["PNG", "TIFF", "GeoTIFF"])
        self.export_format.setToolTip("GeoTIFF 仅在输入具备地理参考信息时才能完整写出。")
        export_layout.addRow("导出格式", self.export_format)

        export_path_row = QHBoxLayout()
        self.export_path_edit = QLineEdit()
        self.export_path_edit.setPlaceholderText("可选，留空则自动保存到 output/ 目录")
        self.export_path_edit.setToolTip("留空时会自动写入项目根目录的 output 文件夹。")
        self.browse_export_btn = QPushButton("浏览...")
        self.browse_export_btn.setToolTip("选择一个具体文件路径，便于固定输出位置。")
        export_path_row.addWidget(self.export_path_edit)
        export_path_row.addWidget(self.browse_export_btn)
        export_path_widget = QWidget()
        export_path_widget.setLayout(export_path_row)
        export_layout.addRow("导出路径", export_path_widget)
        layout.addWidget(export_group)

        action_row = QHBoxLayout()
        self.generate_btn = QPushButton("生成 DOM")
        self.generate_btn.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; font-weight: bold; padding: 10px; }"
        )
        self.generate_btn.setToolTip("开始执行 DOM 生成流程。")
        action_row.addWidget(self.generate_btn)
        layout.addLayout(action_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.stage_label = QLabel("当前阶段: 就绪")
        self.stage_label.setStyleSheet("color: #1f4e79; padding: 4px 0; font-weight: 600;")
        layout.addWidget(self.stage_label)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #555; padding: 4px;")
        layout.addWidget(self.status_label)

        summary_header = QLabel("当前参数预览")
        summary_header.setStyleSheet("font-weight: bold; color: #444;")
        layout.addWidget(summary_header)

        self.config_summary = QTextEdit()
        self.config_summary.setReadOnly(True)
        self.config_summary.setMinimumHeight(110)
        self.config_summary.setPlaceholderText("调整上方参数时，这里会显示当前 DOM 生产配置摘要。")
        layout.addWidget(self.config_summary)

        result_header = QLabel("运行结果")
        result_header.setStyleSheet("font-weight: bold; color: #444;")
        layout.addWidget(result_header)

        self.result_info = QTextEdit()
        self.result_info.setReadOnly(True)
        self.result_info.setMinimumHeight(110)
        self.result_info.setFont(QFont("Consolas"))
        self.result_info.setPlaceholderText("执行完成后，这里会显示结果摘要、尺寸、耗时和导出路径。")
        layout.addWidget(self.result_info)

        layout.addStretch()

        self.refresh_btn.clicked.connect(self.refresh_requested.emit)
        self.generate_btn.clicked.connect(self.generate_requested.emit)
        self.browse_export_btn.clicked.connect(self.browse_export_requested.emit)
        self.select_all_btn.clicked.connect(self.select_all_images)
        self.select_none_btn.clicked.connect(self.select_none_images)
        self.workspace_all_check.toggled.connect(self._sync_selection_state)
        self.workspace_all_check.toggled.connect(self._update_summary)
        self.enable_color_balance.toggled.connect(self._sync_dependent_controls)
        self.enable_color_balance.toggled.connect(self._update_summary)
        self.mosaic_mode.currentIndexChanged.connect(self._update_summary)
        self.feature_method.currentIndexChanged.connect(self._update_summary)
        self.color_method.currentIndexChanged.connect(self._update_summary)
        self.use_reference.toggled.connect(self._update_summary)
        self.blend_method.currentIndexChanged.connect(self._sync_dependent_controls)
        self.blend_method.currentIndexChanged.connect(self._update_summary)
        self.feather_radius.valueChanged.connect(self._update_summary)
        self.export_enabled.toggled.connect(self._sync_dependent_controls)
        self.export_enabled.toggled.connect(self._update_summary)
        self.export_format.currentIndexChanged.connect(self._update_summary)
        self.export_path_edit.textChanged.connect(self._update_summary)
        self._sync_selection_state(self.workspace_all_check.isChecked())
        self._sync_dependent_controls()
        self._update_summary()

    def _sync_selection_state(self, checked: bool):
        self.image_list.setEnabled(not checked)
        self.select_all_btn.setEnabled(not checked)
        self.select_none_btn.setEnabled(not checked)

    def _sync_dependent_controls(self):
        enable_color = self.enable_color_balance.isChecked()
        self.color_method.setEnabled(enable_color)
        self.use_reference.setEnabled(enable_color)

        use_feather = self.blend_method.currentText() == "羽化"
        self.feather_radius.setEnabled(use_feather)

        export_enabled = self.export_enabled.isChecked()
        self.export_format.setEnabled(export_enabled)
        self.export_path_edit.setEnabled(export_enabled)
        self.browse_export_btn.setEnabled(export_enabled)
        self._update_summary()

    def set_images(self, names: List[str]):
        current_selection = set(self.selected_image_names())
        self.image_list.clear()
        for name in names:
            item = QListWidgetItem(name)
            self.image_list.addItem(item)
            if name in current_selection:
                item.setSelected(True)
        self.image_count_label.setText(f"当前已载入 {len(names)} 张影像")

    def select_all_images(self):
        for i in range(self.image_list.count()):
            self.image_list.item(i).setSelected(True)

    def select_none_images(self):
        self.image_list.clearSelection()

    def selected_image_names(self) -> List[str]:
        return [item.text() for item in self.image_list.selectedItems()]

    def _selected_image_display_text(self) -> str:
        names = self.selected_image_names()
        if not names:
            return "未手动选择"
        if len(names) <= 5:
            return "、".join(names)
        return "、".join(names[:5]) + f" 等 {len(names)} 张"

    def get_config(self) -> Dict[str, object]:
        mosaic_mode_map = {
            "自动判断": "auto",
            "地理参考优先": "geo_first",
            "特征匹配优先": "feature_first",
        }
        color_method_map = {
            "平均亮度对齐": "mean",
            "直方图匹配": "histogram",
        }
        blend_method_map = {
            "羽化": "feather",
            "加权平均": "weighted",
        }
        export_format_map = {
            "PNG": "PNG",
            "TIFF": "TIFF",
            "GeoTIFF": "GEOTIFF",
        }
        return {
            "auto_use_all": self.workspace_all_check.isChecked(),
            "selected_names": self.selected_image_names(),
            "mosaic_mode": mosaic_mode_map.get(self.mosaic_mode.currentText(), "auto"),
            "feature_method": self.feature_method.currentText(),
            "enable_color_balance": self.enable_color_balance.isChecked(),
            "color_method": color_method_map.get(self.color_method.currentText(), "mean"),
            "use_reference": self.use_reference.isChecked(),
            "blend_method": blend_method_map.get(self.blend_method.currentText(), "feather"),
            "feather_radius": self.feather_radius.value(),
            "export_enabled": self.export_enabled.isChecked(),
            "export_format": export_format_map.get(self.export_format.currentText(), "PNG"),
            "export_path": self.export_path_edit.text().strip(),
        }

    def set_export_path(self, path: str):
        self.export_path_edit.setText(path)

    def set_status(self, text: str):
        self.status_label.setText(text)

    def set_stage(self, text: str):
        self.stage_label.setText(f"当前阶段: {text}")

    def set_result_info(self, text: str):
        self.result_info.setPlainText(text)

    def set_progress(self, value: int):
        self.progress_bar.setValue(value)

    def set_busy(self, busy: bool):
        self.generate_btn.setEnabled(not busy)
        self.refresh_btn.setEnabled(not busy)
        self.workspace_all_check.setEnabled(not busy)
        self.image_list.setEnabled(not busy and not self.workspace_all_check.isChecked())
        self.select_all_btn.setEnabled(not busy and not self.workspace_all_check.isChecked())
        self.select_none_btn.setEnabled(not busy and not self.workspace_all_check.isChecked())
        if not busy:
            self._sync_selection_state(self.workspace_all_check.isChecked())
            self._sync_dependent_controls()
            self.set_stage("就绪")

    def _update_summary(self):
        summary = [
            f"输入来源: {'Workspace 全部影像' if self.workspace_all_check.isChecked() else self._selected_image_display_text()}",
            f"镶嵌模式: {self.mosaic_mode.currentText()} / {self.feature_method.currentText()}",
            f"匀色: {'启用' if self.enable_color_balance.isChecked() else '关闭'}"
            + (
                f" ({self.color_method.currentText()}, "
                f"{'参考影像' if self.use_reference.isChecked() else '自动平均参考'})"
                if self.enable_color_balance.isChecked()
                else ""
            ),
            f"融合: {self.blend_method.currentText()}"
            + (f" / 羽化半径 {self.feather_radius.value()}" if self.blend_method.currentText() == "羽化" else ""),
            f"导出: {'启用' if self.export_enabled.isChecked() else '关闭'}"
            + (
                f" ({self.export_format.currentText()}"
                + (f", {self.export_path_edit.text().strip()}" if self.export_path_edit.text().strip() else ", 自动路径")
                + ")"
                if self.export_enabled.isChecked()
                else ""
            ),
        ]
        self.config_summary.setPlainText("\n".join(summary))
