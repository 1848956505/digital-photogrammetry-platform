"""模块三控制面板。"""

from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class DsmDemControlPanel(QWidget):
    """DSM / DEM 生产控制面板。"""

    refresh_requested = Signal()
    browse_output_requested = Signal()
    run_dsm_requested = Signal()
    run_dem_requested = Signal()
    run_pipeline_requested = Signal()
    show_dsm_requested = Signal()
    show_dem_requested = Signal()
    show_ground_mask_requested = Signal()
    show_hillshade_compare_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workspace_images: List[str] = []
        self._workspace_paths: Dict[str, str] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("DSM / DEM 生产")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976d2;")
        layout.addWidget(title)

        intro = QLabel(
            "当前版本采用近似立体像对的相对 DSM MVP 流程，并基于传统形态学地面滤波生成相对 DEM。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #555; padding: 4px 6px; background: #f6f8fb; border-radius: 4px;")
        layout.addWidget(intro)

        input_group = QGroupBox("数据输入")
        input_layout = QFormLayout(input_group)
        self.left_combo = QComboBox()
        self.right_combo = QComboBox()
        self.refresh_btn = QPushButton("刷新影像列表")
        self.refresh_btn.clicked.connect(self.refresh_requested.emit)
        input_layout.addRow("左影像", self.left_combo)
        input_layout.addRow("右影像", self.right_combo)
        input_layout.addRow("", self.refresh_btn)
        layout.addWidget(input_group)

        dsm_group = QGroupBox("DSM 参数")
        dsm_layout = QFormLayout(dsm_group)
        self.algorithm_label = QLabel("StereoSGBM")
        dsm_layout.addRow("匹配算法", self.algorithm_label)
        self.max_side_spin = QSpinBox()
        self.max_side_spin.setRange(512, 4096)
        self.max_side_spin.setValue(1280)
        dsm_layout.addRow("最大处理边长", self.max_side_spin)
        self.use_clahe_check = QCheckBox("启用 CLAHE")
        self.use_clahe_check.setChecked(True)
        dsm_layout.addRow("", self.use_clahe_check)
        self.min_disparity_spin = QSpinBox()
        self.min_disparity_spin.setRange(0, 256)
        self.min_disparity_spin.setValue(0)
        dsm_layout.addRow("最小视差", self.min_disparity_spin)
        self.num_disparities_spin = QSpinBox()
        self.num_disparities_spin.setRange(16, 512)
        self.num_disparities_spin.setSingleStep(16)
        self.num_disparities_spin.setValue(96)
        dsm_layout.addRow("视差范围", self.num_disparities_spin)
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(3, 21)
        self.block_size_spin.setSingleStep(2)
        self.block_size_spin.setValue(7)
        dsm_layout.addRow("Block Size", self.block_size_spin)
        self.uniqueness_spin = QSpinBox()
        self.uniqueness_spin.setRange(1, 30)
        self.uniqueness_spin.setValue(10)
        dsm_layout.addRow("Uniqueness", self.uniqueness_spin)
        self.speckle_window_spin = QSpinBox()
        self.speckle_window_spin.setRange(0, 500)
        self.speckle_window_spin.setValue(50)
        dsm_layout.addRow("Speckle Window", self.speckle_window_spin)
        self.speckle_range_spin = QSpinBox()
        self.speckle_range_spin.setRange(0, 64)
        self.speckle_range_spin.setValue(2)
        dsm_layout.addRow("Speckle Range", self.speckle_range_spin)
        self.focal_length_spin = QDoubleSpinBox()
        self.focal_length_spin.setRange(0.0, 500.0)
        self.focal_length_spin.setDecimals(3)
        self.focal_length_spin.setSuffix(" mm")
        dsm_layout.addRow("焦距", self.focal_length_spin)
        self.baseline_spin = QDoubleSpinBox()
        self.baseline_spin.setRange(0.0, 10.0)
        self.baseline_spin.setDecimals(4)
        self.baseline_spin.setSuffix(" m")
        dsm_layout.addRow("基线", self.baseline_spin)
        self.pixel_size_spin = QDoubleSpinBox()
        self.pixel_size_spin.setRange(0.0, 50.0)
        self.pixel_size_spin.setDecimals(3)
        self.pixel_size_spin.setSuffix(" um")
        dsm_layout.addRow("像元尺寸", self.pixel_size_spin)
        self.mode_hint_label = QLabel("当前结果默认为相对高程，不代表绝对地理高程。")
        self.mode_hint_label.setWordWrap(True)
        self.mode_hint_label.setStyleSheet("color: #9c5a00;")
        dsm_layout.addRow("高程说明", self.mode_hint_label)
        layout.addWidget(dsm_group)

        dem_group = QGroupBox("DEM 参数")
        dem_layout = QFormLayout(dem_group)
        self.dem_method_combo = QComboBox()
        self.dem_method_combo.addItems(["形态学滤波（P0）", "坡度滤波（P1）"])
        dem_layout.addRow("滤波方法", self.dem_method_combo)
        self.kernel_size_spin = QSpinBox()
        self.kernel_size_spin.setRange(3, 51)
        self.kernel_size_spin.setSingleStep(2)
        self.kernel_size_spin.setValue(9)
        dem_layout.addRow("结构元素大小", self.kernel_size_spin)
        self.ground_threshold_spin = QDoubleSpinBox()
        self.ground_threshold_spin.setRange(0.1, 50.0)
        self.ground_threshold_spin.setDecimals(3)
        self.ground_threshold_spin.setValue(1.5)
        dem_layout.addRow("地面阈值", self.ground_threshold_spin)
        self.smooth_sigma_spin = QDoubleSpinBox()
        self.smooth_sigma_spin.setRange(0.0, 10.0)
        self.smooth_sigma_spin.setDecimals(2)
        self.smooth_sigma_spin.setValue(1.0)
        dem_layout.addRow("平滑强度", self.smooth_sigma_spin)
        layout.addWidget(dem_group)

        ai_group = QGroupBox("AI 扩展位")
        ai_layout = QFormLayout(ai_group)
        self.ai_enable_check = QCheckBox("AI 增强地面分类")
        self.ai_enable_check.setEnabled(False)
        ai_layout.addRow("", self.ai_enable_check)
        self.ai_status_label = QLabel("未安装模型，当前使用传统滤波")
        self.ai_status_label.setWordWrap(True)
        self.ai_status_label.setStyleSheet("color: #666;")
        ai_layout.addRow("模型状态", self.ai_status_label)
        layout.addWidget(ai_group)

        output_group = QGroupBox("输出设置")
        output_layout = QFormLayout(output_group)
        output_dir_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("留空则自动写入 output/mod3/")
        self.browse_output_btn = QPushButton("浏览...")
        self.browse_output_btn.clicked.connect(self.browse_output_requested.emit)
        output_dir_row.addWidget(self.output_dir_edit)
        output_dir_row.addWidget(self.browse_output_btn)
        output_dir_widget = QWidget()
        output_dir_widget.setLayout(output_dir_row)
        output_layout.addRow("输出目录", output_dir_widget)
        layout.addWidget(output_group)

        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        run_row = QHBoxLayout()
        self.run_dsm_btn = QPushButton("生成 DSM")
        self.run_dem_btn = QPushButton("生成 DEM")
        self.run_pipeline_btn = QPushButton("一键完整流程")
        run_row.addWidget(self.run_dsm_btn)
        run_row.addWidget(self.run_dem_btn)
        run_row.addWidget(self.run_pipeline_btn)
        action_layout.addLayout(run_row)
        show_row = QHBoxLayout()
        self.show_dsm_btn = QPushButton("显示 DSM 3D")
        self.show_dem_btn = QPushButton("显示 DEM 3D")
        self.show_mask_btn = QPushButton("显示地面掩膜")
        show_row.addWidget(self.show_dsm_btn)
        show_row.addWidget(self.show_dem_btn)
        show_row.addWidget(self.show_mask_btn)
        action_layout.addLayout(show_row)
        compare_row = QHBoxLayout()
        self.show_compare_btn = QPushButton("DSM/DEM 对比")
        compare_row.addWidget(self.show_compare_btn)
        compare_row.addStretch(1)
        action_layout.addLayout(compare_row)
        self.export_btn = QPushButton("导出结果路径")
        action_layout.addWidget(self.export_btn)
        layout.addWidget(action_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.stage_label = QLabel("当前阶段: 就绪")
        self.stage_label.setStyleSheet("color: #1f4e79; font-weight: 600;")
        layout.addWidget(self.stage_label)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #555;")
        layout.addWidget(self.status_label)

        summary_header = QLabel("结果摘要")
        summary_header.setStyleSheet("font-weight: bold; color: #444;")
        layout.addWidget(summary_header)
        self.result_info = QTextEdit()
        self.result_info.setReadOnly(True)
        self.result_info.setMinimumHeight(180)
        layout.addWidget(self.result_info)

        layout.addStretch()

        self.run_dsm_btn.clicked.connect(self.run_dsm_requested.emit)
        self.run_dem_btn.clicked.connect(self.run_dem_requested.emit)
        self.run_pipeline_btn.clicked.connect(self.run_pipeline_requested.emit)
        self.show_dsm_btn.clicked.connect(self.show_dsm_requested.emit)
        self.show_dem_btn.clicked.connect(self.show_dem_requested.emit)
        self.show_mask_btn.clicked.connect(self.show_ground_mask_requested.emit)
        self.show_compare_btn.clicked.connect(self.show_hillshade_compare_requested.emit)
        self.export_btn.clicked.connect(self.export_requested.emit)

    def set_workspace_images(self, names: List[str], name_to_path: Dict[str, str]):
        self._workspace_images = list(names)
        self._workspace_paths = dict(name_to_path)
        current_left = self.left_combo.currentText()
        current_right = self.right_combo.currentText()
        self.left_combo.blockSignals(True)
        self.right_combo.blockSignals(True)
        self.left_combo.clear()
        self.right_combo.clear()
        self.left_combo.addItems(names)
        self.right_combo.addItems(names)
        if current_left in names:
            self.left_combo.setCurrentText(current_left)
        elif names:
            self.left_combo.setCurrentIndex(0)
        if current_right in names:
            self.right_combo.setCurrentText(current_right)
        elif len(names) > 1:
            self.right_combo.setCurrentIndex(1)
        elif names:
            self.right_combo.setCurrentIndex(0)
        self.left_combo.blockSignals(False)
        self.right_combo.blockSignals(False)

    def get_config(self) -> Dict[str, object]:
        left_name = self.left_combo.currentText().strip()
        right_name = self.right_combo.currentText().strip()
        return {
            "left_image_name": left_name,
            "right_image_name": right_name,
            "left_image_path": self._workspace_paths.get(left_name, ""),
            "right_image_path": self._workspace_paths.get(right_name, ""),
            "max_processing_side": self.max_side_spin.value(),
            "use_clahe": self.use_clahe_check.isChecked(),
            "min_disparity": self.min_disparity_spin.value(),
            "num_disparities": self.num_disparities_spin.value(),
            "block_size": self.block_size_spin.value(),
            "uniqueness_ratio": self.uniqueness_spin.value(),
            "speckle_window_size": self.speckle_window_spin.value(),
            "speckle_range": self.speckle_range_spin.value(),
            "focal_length_mm": self.focal_length_spin.value(),
            "baseline_m": self.baseline_spin.value(),
            "pixel_size_um": self.pixel_size_spin.value(),
            "dem_method": self.dem_method_combo.currentText(),
            "morph_kernel_size": self.kernel_size_spin.value(),
            "ground_threshold": self.ground_threshold_spin.value(),
            "smooth_sigma": self.smooth_sigma_spin.value(),
            "output_dir": self.output_dir_edit.text().strip(),
        }

    def set_output_dir(self, path: str):
        self.output_dir_edit.setText(path)

    def output_dir_text(self) -> str:
        return self.output_dir_edit.text().strip()

    def set_status(self, text: str):
        self.status_label.setText(text)

    def set_stage(self, text: str):
        self.stage_label.setText(f"当前阶段: {text}")

    def set_progress(self, value: int):
        self.progress_bar.setValue(value)

    def set_result_info(self, text: str):
        self.result_info.setPlainText(text)

    def set_busy(self, busy: bool):
        for widget in [
            self.left_combo,
            self.right_combo,
            self.refresh_btn,
            self.run_dsm_btn,
            self.run_dem_btn,
            self.run_pipeline_btn,
            self.show_dsm_btn,
            self.show_dem_btn,
            self.show_mask_btn,
            self.show_compare_btn,
            self.export_btn,
            self.browse_output_btn,
        ]:
            widget.setEnabled(not busy)
        for widget in [
            self.max_side_spin,
            self.use_clahe_check,
            self.min_disparity_spin,
            self.num_disparities_spin,
            self.block_size_spin,
            self.uniqueness_spin,
            self.speckle_window_spin,
            self.speckle_range_spin,
            self.focal_length_spin,
            self.baseline_spin,
            self.pixel_size_spin,
            self.dem_method_combo,
            self.kernel_size_spin,
            self.ground_threshold_spin,
            self.smooth_sigma_spin,
            self.output_dir_edit,
        ]:
            widget.setEnabled(not busy)

    def set_ai_model_available(self, available: bool, model_path: str):
        self.ai_enable_check.setEnabled(available)
        self.ai_enable_check.setChecked(False)
        if available:
            self.ai_status_label.setText(f"已检测到真实模型文件：{model_path}")
        else:
            self.ai_status_label.setText("未安装模型，当前使用传统滤波")
