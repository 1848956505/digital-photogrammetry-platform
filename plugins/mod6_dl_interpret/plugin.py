"""
模块六：基于深度学习的遥感影像解译插件
"""
import os
import sys
import numpy as np
import cv2
from typing import Dict, Any, Optional
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                              QGroupBox, QFormLayout, QComboBox, QSpinBox, 
                              QCheckBox, QFileDialog, QMessageBox, QProgressBar,
                              QHBoxLayout, QSlider, QTableWidget, QTableWidgetItem,
                              QHeaderView, QSplitter, QTextEdit)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont

from core.base_interface import IPlugin
from core.log_manager import log_manager

# 添加项目路径，确保能找到 LoveDA
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
loveda_path = os.path.join(project_root, 'LoveDA-master', 'Semantic_Segmentation')
if os.path.exists(loveda_path):
    sys.path.insert(0, loveda_path)

# 导入算法模块
try:
    from plugins.mod6_dl_interpret.algorithms.hrnet_model import HRNetSegmentor
    from plugins.mod6_dl_interpret.algorithms.preprocessor import ImagePreprocessor
    from plugins.mod6_dl_interpret.algorithms.postprocessor import ResultPostprocessor
    from plugins.mod6_dl_interpret.algorithms.metrics import SegmentationMetrics
    ALGORITHMS_AVAILABLE = True
except ImportError as e:
    ALGORITHMS_AVAILABLE = False
    log_manager.warning(f"算法模块导入失败: {e}")


class SegmentationThread(QThread):
    """语义分割后台线程"""
    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)
    log = Signal(str)
    
    def __init__(self, segmentor, image_path, output_dir=None):
        super().__init__()
        self.segmentor = segmentor
        self.image_path = image_path
        self.output_dir = output_dir
    
    def run(self):
        try:
            self.log.emit("开始读取图像...")
            self.progress.emit(10)
            
            # 读取图像
            image = ImagePreprocessor.read_image(self.image_path)
            if image is None:
                raise ValueError("无法读取图像")
            
            self.log.emit("图像读取完成，开始预处理...")
            self.progress.emit(30)
            
            # 预测
            self.log.emit("开始语义分割中...")
            prediction = self.segmentor.predict(image)
            
            self.progress.emit(80)
            self.log.emit("分割完成，后处理中...")
            
            # 保存结果
            result = {
                'image': image,
                'prediction': prediction,
                'image_path': self.image_path
            }
            
            self.progress.emit(100)
            self.log.emit("处理完成！")
            
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))
            import traceback
            traceback.print_exc()


class DLInterpretPlugin(IPlugin):
    """基于深度学习的遥感影像解译模块"""
    
    def __init__(self, workspace):
        super().__init__(workspace)
        self.current_image = None
        self.current_prediction = None
        self.segmentor = None
        self.model_loaded = False
        
        # 尝试初始化模型
        self._init_model()
    
    def _init_model(self):
        """初始化模型"""
        try:
            # 查找模型路径
            model_path = os.path.join(
                project_root, 
                'LoveDA-master', 
                'Semantic_Segmentation', 
                'hrnetw32.pth'
            )
            
            if os.path.exists(model_path):
                log_manager.info(f"找到模型文件: {model_path}")
                self.segmentor = HRNetSegmentor(model_path)
                self.model_loaded = self.segmentor.is_available()
                
                if self.model_loaded:
                    log_manager.info("模型加载成功")
                else:
                    log_manager.warning("模型加载失败，将使用演示模式")
            else:
                log_manager.warning(f"模型文件不存在: {model_path}")
                # 创建一个空的 segmentor（演示模式）
                self.segmentor = HRNetSegmentor(None)
                
        except Exception as e:
            log_manager.error(f"初始化模型失败: {e}")
            self.segmentor = None
    
    def plugin_info(self) -> Dict[str, Any]:
        return {
            'name': '深度学习解译',
            'group': '模块',
            'version': '1.0.0',
            'description': '基于 HRNet 的遥感影像语义分割，支持7类地物识别'
        }
    
    def get_ui_panel(self) -> QWidget:
        """返回参数设置面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 标题
        title = QLabel("深度学习遥感影像解译")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976d2; padding: 10px;")
        layout.addWidget(title)
        
        # ==================== 1. 模型状态 ====================
        group_model = QGroupBox("模型状态")
        model_layout = QVBoxLayout()
        
        self.model_status_label = QLabel()
        self._update_model_status()
        model_layout.addWidget(self.model_status_label)
        
        self.btn_reload_model = QPushButton("重新加载模型")
        self.btn_reload_model.clicked.connect(self._reload_model)
        model_layout.addWidget(self.btn_reload_model)
        
        group_model.setLayout(model_layout)
        layout.addWidget(group_model)
        
        # ==================== 2. 图像选择 ====================
        group_input = QGroupBox("输入图像")
        input_layout = QVBoxLayout()
        
        # 按钮布局
        btn_layout = QHBoxLayout()
        
        self.btn_select_image = QPushButton("选择图像文件")
        self.btn_select_image.clicked.connect(self._select_image)
        btn_layout.addWidget(self.btn_select_image)
        
        self.btn_use_workspace = QPushButton("使用 Workspace 图像")
        self.btn_use_workspace.clicked.connect(self._use_workspace_image)
        btn_layout.addWidget(self.btn_use_workspace)
        
        input_layout.addLayout(btn_layout)
        
        self.image_path_label = QLabel("未选择图像")
        self.image_path_label.setStyleSheet("color: #666; font-size: 11px; word-wrap: true;")
        self.image_path_label.setWordWrap(True)
        input_layout.addWidget(self.image_path_label)
        
        group_input.setLayout(input_layout)
        layout.addWidget(group_input)
        
        # ==================== 3. 分割参数 ====================
        group_params = QGroupBox("分割参数")
        params_layout = QFormLayout()
        
        self.patch_size = QSpinBox()
        self.patch_size.setRange(256, 1024)
        self.patch_size.setValue(512)
        self.patch_size.setSingleStep(128)
        params_layout.addRow("分块大小:", self.patch_size)
        
        self.overlap_size = QSpinBox()
        self.overlap_size.setRange(0, 256)
        self.overlap_size.setValue(64)
        self.overlap_size.setSingleStep(32)
        params_layout.addRow("重叠大小:", self.overlap_size)
        
        self.overlay_alpha = QSlider(Qt.Horizontal)
        self.overlay_alpha.setRange(0, 100)
        self.overlay_alpha.setValue(50)
        self.overlay_alpha.setTickPosition(QSlider.TicksBelow)
        self.overlay_alpha.setTickInterval(10)
        params_layout.addRow("叠加透明度:", self.overlay_alpha)
        
        group_params.setLayout(params_layout)
        layout.addWidget(group_params)
        
        # ==================== 4. 显示模式 ====================
        group_display = QGroupBox("显示模式")
        display_layout = QVBoxLayout()
        
        self.display_mode = QComboBox()
        self.display_mode.addItems([
            "单独显示结果",
            "左右对比显示（左原图，右结果）",
            "叠加显示"
        ])
        self.display_mode.setCurrentIndex(1)  # 默认左右对比
        display_layout.addWidget(self.display_mode)
        
        group_display.setLayout(display_layout)
        layout.addWidget(group_display)
        
        # ==================== 4. 执行分割 ====================
        self.btn_segment = QPushButton("🚀 执行语义分割")
        self.btn_segment.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 12px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.btn_segment.clicked.connect(self._run_segmentation)
        layout.addWidget(self.btn_segment)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # 日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)
        
        # ==================== 5. 结果分析 ====================
        group_results = QGroupBox("结果分析")
        results_layout = QVBoxLayout()
        
        # 类别分布表格
        self.distribution_table = QTableWidget()
        self.distribution_table.setRowCount(7)
        self.distribution_table.setColumnCount(2)
        self.distribution_table.setHorizontalHeaderLabels(["类别", "占比"])
        self.distribution_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._init_distribution_table()
        results_layout.addWidget(self.distribution_table)
        
        self.btn_save_result = QPushButton("保存结果")
        self.btn_save_result.clicked.connect(self._save_result)
        self.btn_save_result.setEnabled(False)
        results_layout.addWidget(self.btn_save_result)
        
        group_results.setLayout(results_layout)
        layout.addWidget(group_results)
        
        layout.addStretch()
        
        # 状态
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.status_label)
        
        return widget
    
    def _init_distribution_table(self):
        """初始化类别分布表格"""
        i = 0
        for class_id in sorted(ResultPostprocessor.CLASSES.keys()):
            class_name = ResultPostprocessor.CLASSES[class_id]
            self.distribution_table.setItem(i, 0, QTableWidgetItem(class_name))
            self.distribution_table.setItem(i, 1, QTableWidgetItem("-"))
            i += 1
    
    def _update_model_status(self):
        """更新模型状态显示"""
        if self.segmentor is None:
            status_text = "❌ 模型未初始化"
            color = "red"
        elif not self.segmentor.is_available():
            info = self.segmentor.get_info()
            if info.get('torch_available'):
                status_text = "⚠️ 演示模式（PyTorch可用，模型未完全加载）"
            else:
                status_text = "⚠️ 演示模式（请安装 PyTorch）"
            color = "orange"
        else:
            info = self.segmentor.get_info()
            status_text = f"✅ 模型已加载\n设备: {info.get('device', 'unknown')}"
            color = "green"
        
        self.model_status_label.setText(status_text)
        self.model_status_label.setStyleSheet(f"color: {color}; padding: 10px;")
    
    def _reload_model(self):
        """重新加载模型"""
        self.log_text.append("正在重新加载模型...")
        self._init_model()
        self._update_model_status()
        self.log_text.append("模型重新加载完成")
    
    def _select_image(self):
        """选择图像文件"""
        path, _ = QFileDialog.getOpenFileName(
            None, "选择遥感影像", "",
            "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        if path:
            self.current_image_path = path
            self.image_path_label.setText(f"已选择: {os.path.basename(path)}")
            self.log_text.append(f"已选择图像: {path}")
    
    def _use_workspace_image(self):
        """使用 Workspace 中当前选中的图像"""
        # 从 workspace 获取图像
        images = self.workspace.get('images', {})
        if not images:
            QMessageBox.information(None, "提示", "Workspace 中没有图像，请先添加图像")
            return
        
        # 尝试获取当前选中的图像（通过 EventBus 或 workspace 状态）
        # 如果没有选中状态，就用第一个图像
        first_name = next(iter(images.keys()))
        first_item = images[first_name]
        
        if isinstance(first_item, dict) and 'path' in first_item:
            path = first_item['path']
            if os.path.exists(path):
                self.current_image_path = path
                self.image_path_label.setText(f"Workspace: {first_name}")
                self.log_text.append(f"使用 Workspace 图像: {first_name} ({path})")
                return
        
        QMessageBox.warning(None, "警告", "无法获取 Workspace 图像路径")
    
    def _run_segmentation(self):
        """执行语义分割"""
        if not hasattr(self, 'current_image_path') or not self.current_image_path:
            QMessageBox.information(None, "提示", "请先选择一张图像")
            return
        
        if self.segmentor is None:
            QMessageBox.warning(None, "警告", "模型未初始化")
            return
        
        # 禁用按钮
        self.btn_segment.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        
        # 创建后台线程
        self.seg_thread = SegmentationThread(
            self.segmentor, 
            self.current_image_path
        )
        self.seg_thread.progress.connect(self.progress_bar.setValue)
        self.seg_thread.log.connect(lambda msg: self.log_text.append(msg))
        self.seg_thread.finished.connect(self._on_segmentation_finished)
        self.seg_thread.error.connect(self._on_segmentation_error)
        
        self.seg_thread.start()
    
    def _on_segmentation_finished(self, result):
        """分割完成"""
        self.current_image = result['image']
        self.current_prediction = result['prediction']
        
        # 计算类别分布
        distribution = ResultPostprocessor.calculate_class_distribution(self.current_prediction)
        
        # 更新表格
        i = 0
        for class_name, percentage in distribution.items():
            self.distribution_table.setItem(i, 1, QTableWidgetItem(f"{percentage:.2f}%"))
            i += 1
        
        # 更新工作空间
        self._update_workspace_results()
        
        # 启用按钮
        self.btn_segment.setEnabled(True)
        self.btn_save_result.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("分割完成")
        self.log_text.append("✅ 分割成功完成！")
        
        QMessageBox.information(None, "完成", "语义分割完成！\n结果已添加到工作空间")
    
    def _on_segmentation_error(self, error_msg):
        """分割出错"""
        self.btn_segment.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.log_text.append(f"❌ 错误: {error_msg}")
        QMessageBox.critical(None, "错误", f"分割失败: {error_msg}")
    
    def _update_workspace_results(self):
        """更新工作空间中的结果"""
        if self.current_prediction is None:
            return
        
        try:
            from datetime import datetime
            import cv2
            
            timestamp = datetime.now().strftime("%H%M%S")
            display_mode = self.display_mode.currentText()
            
            # 1. 保存原始预测结果（灰度）
            pred_name = f"分割结果_{timestamp}"
            pred_path = os.path.join(project_root, 'output', f"{pred_name}.png")
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            ResultPostprocessor.save_result(self.current_prediction, pred_path, colormap=False)
            
            # 2. 保存彩色结果
            color_name = f"分割结果彩色_{timestamp}"
            color_path = os.path.join(project_root, 'output', f"{color_name}.png")
            ResultPostprocessor.save_result(self.current_prediction, color_path, colormap=True)
            
            # 3. 根据显示模式处理
            if self.current_image is not None:
                if "左右对比" in display_mode:
                    # 左右对比显示
                    self._publish_compare_view(timestamp)
                elif "叠加显示" in display_mode:
                    # 叠加显示
                    self._publish_overlay_view(timestamp)
                else:
                    # 单独显示结果
                    self._publish_single_view(timestamp, color_path, color_name)
            
            log_manager.info(f"分割结果已保存，显示模式: {display_mode}")
            
        except Exception as e:
            log_manager.error(f"保存结果失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _publish_single_view(self, timestamp: str, color_path: str, color_name: str):
        """单独显示结果"""
        try:
            from core.event_bus import EventTopics, get_event_bus
            event_bus = get_event_bus()
            
            # 添加到 workspace
            if hasattr(self.workspace, 'add_processed_image'):
                import cv2
                img_array = np.fromfile(color_path, dtype=np.uint8)
                result_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                self.workspace.add_processed_image(color_name, result_image)
            
            # 发布事件
            event_bus.publish(EventTopics.TOPIC_IMAGE_UPDATED, {
                "path": color_path, 
                "name": color_name
            })
            
            self.log_text.append(f"已显示结果: {color_name}")
            
        except Exception as e:
            log_manager.error(f"显示结果失败: {e}")
    
    def _publish_overlay_view(self, timestamp: str):
        """叠加显示"""
        try:
            import cv2
            from core.event_bus import EventTopics, get_event_bus
            event_bus = get_event_bus()
            
            # 生成叠加图像
            overlay_name = f"分割结果叠加_{timestamp}"
            overlay_path = os.path.join(project_root, 'output', f"{overlay_name}.png")
            
            overlay = ResultPostprocessor.overlay(
                self.current_image, 
                self.current_prediction,
                alpha=self.overlay_alpha.value() / 100.0
            )
            cv2.imwrite(overlay_path, overlay)
            
            # 添加到 workspace
            if hasattr(self.workspace, 'add_processed_image'):
                self.workspace.add_processed_image(overlay_name, overlay)
            
            # 发布事件
            event_bus.publish(EventTopics.TOPIC_IMAGE_UPDATED, {
                "path": overlay_path, 
                "name": overlay_name
            })
            
            self.log_text.append(f"已显示叠加结果: {overlay_name}")
            
        except Exception as e:
            log_manager.error(f"显示叠加结果失败: {e}")
    
    def _publish_compare_view(self, timestamp: str):
        """左右对比显示"""
        try:
            import cv2
            from core.event_bus import EventTopics, get_event_bus
            event_bus = get_event_bus()
            
            # 生成左右对比图像
            compare_name = f"分割对比_{timestamp}"
            compare_path = os.path.join(project_root, 'output', f"{compare_name}.png")
            
            # 获取彩色分割结果
            color_pred = ResultPostprocessor.colorize(self.current_prediction)
            
            # 确保原图是 BGR 格式
            if self.current_image.shape[2] == 3:
                orig_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            else:
                orig_bgr = self.current_image
            
            # 左右拼接
            h, w = orig_bgr.shape[:2]
            compare = np.zeros((h, w * 2, 3), dtype=np.uint8)
            compare[:, :w] = orig_bgr
            compare[:, w:] = color_pred
            
            # 添加标签
            cv2.putText(compare, "原图", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(compare, "分割结果", (w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 保存对比图
            cv2.imwrite(compare_path, compare)
            
            # 同时保存原图和结果到 workspace，方便单独查看
            # 1. 原图
            orig_name = f"原图_{timestamp}"
            orig_path = os.path.join(project_root, 'output', f"{orig_name}.png")
            cv2.imwrite(orig_path, orig_bgr)
            
            # 2. 分割结果彩色
            color_name = f"分割结果彩色_{timestamp}"
            color_path = os.path.join(project_root, 'output', f"{color_name}.png")
            ResultPostprocessor.save_result(self.current_prediction, color_path, colormap=True)
            
            # 添加到 workspace 并发布对比图
            if hasattr(self.workspace, 'add_processed_image'):
                self.workspace.add_processed_image(compare_name, compare)
            
            # 发布事件（显示对比图）
            event_bus.publish(EventTopics.TOPIC_IMAGE_UPDATED, {
                "path": compare_path, 
                "name": compare_name
            })
            
            self.log_text.append(f"已显示对比结果: {compare_name}")
            self.log_text.append(f"同时保存了: {orig_name}, {color_name}")
            
        except Exception as e:
            log_manager.error(f"显示对比结果失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_result(self):
        """保存结果"""
        if self.current_prediction is None:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            None, "保存分割结果", "",
            "PNG图像 (*.png);;TIFF图像 (*.tif)"
        )
        
        if path:
            try:
                # 判断是否保存彩色
                reply = QMessageBox.question(
                    None, "保存选项", 
                    "是否保存为彩色图像？",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                
                if reply == QMessageBox.Cancel:
                    return
                
                colormap = (reply == QMessageBox.Yes)
                ResultPostprocessor.save_result(self.current_prediction, path, colormap=colormap)
                
                QMessageBox.information(None, "成功", f"结果已保存到:\n{path}")
                log_manager.info(f"结果已保存: {path}")
                
            except Exception as e:
                QMessageBox.critical(None, "错误", f"保存失败: {str(e)}")
    
    def execute(self, *args, **kwargs):
        """执行处理（被外部调用）"""
        pass
