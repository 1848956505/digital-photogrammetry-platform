"""
模块二：空中三角测量插件
实现相对定向、区域网平差、残差分析、AI异常点检测
"""
import os
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QGroupBox, QFormLayout,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox,
    QTextEdit, QProgressBar, QHBoxLayout
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont

from core.base_interface import IPlugin
from core.log_manager import log_manager
from core.event_bus import EventTopics, get_event_bus
from core.workspace import get_workspace

from .processors import (
    RelativeOrientation, BundleAdjustment, 
    ResidualAnalysis, OutlierDetection
)


class AerialTriangulationPlugin(IPlugin):
    """空中三角测量模块"""
    
    # 信号
    process_finished = Signal(dict)
    
    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace
        self.processed_result = None
        self.status_label = None  # 状态标签（由UI设置）
    
    def plugin_info(self) -> Dict[str, Any]:
        return {
            'name': '空中三角测量',
            'group': '模块',
            'version': '1.0.0',
            'description': '相对定向、区域网平差、残差分析、AI异常点检测'
        }
    
    def get_ui_panel(self) -> QWidget:
        """返回参数设置面板"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题
        title = QLabel("空中三角测量")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976d2; padding: 10px;")
        layout.addWidget(title)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.status_label)
        
        # 1. 相对定向
        group_relative = QGroupBox("相对定向")
        relative_layout = QFormLayout()
        
        self.relative_method = QComboBox()
        self.relative_method.addItems(["经典相对定向", "独立模型法", "光束法"])
        relative_layout.addRow("定向方法:", self.relative_method)
        
        self.focal_length = QDoubleSpinBox()
        self.focal_length.setRange(50, 500)
        self.focal_length.setValue(153.24)
        self.focal_length.setSuffix(" mm")
        relative_layout.addRow("焦距:", self.focal_length)
        
        self.btn_relative = QPushButton("执行相对定向")
        self.btn_relative.clicked.connect(self._execute_relative_orientation)
        self.btn_relative.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1565c0; }
        """)
        relative_layout.addRow("", self.btn_relative)
        
        group_relative.setLayout(relative_layout)
        layout.addWidget(group_relative)
        
        # 2. 区域网平差
        group_bundle = QGroupBox("区域网平差")
        bundle_layout = QFormLayout()
        
        self.bundle_method = QComboBox()
        self.bundle_method.addItems(["光束法平差", "独立模型法平差", "航带法平差", "间接平差"])
        bundle_layout.addRow("平差方法:", self.bundle_method)
        
        self.iterations = QSpinBox()
        self.iterations.setRange(10, 500)
        self.iterations.setValue(100)
        bundle_layout.addRow("迭代次数:", self.iterations)
        
        self.convergence = QDoubleSpinBox()
        self.convergence.setRange(0.0001, 1.0)
        self.convergence.setValue(0.001)
        self.convergence.setDecimals(4)
        self.convergence.setSuffix(" m")
        bundle_layout.addRow("收敛阈值:", self.convergence)
        
        self.btn_bundle = QPushButton("执行区域网平差")
        self.btn_bundle.clicked.connect(self._execute_bundle_adjustment)
        self.btn_bundle.setStyleSheet("""
            QPushButton {
                background-color: #1976d2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1565c0; }
        """)
        bundle_layout.addRow("", self.btn_bundle)
        
        group_bundle.setLayout(bundle_layout)
        layout.addWidget(group_bundle)
        
        # 3. 残差分析
        group_residual = QGroupBox("残差分析")
        residual_layout = QVBoxLayout()
        
        self.show_residual_plot = QCheckBox("显示残差分布图")
        self.show_residual_plot.setChecked(True)
        residual_layout.addWidget(self.show_residual_plot)
        
        residual_btn = QPushButton("生成残差报告")
        residual_btn.clicked.connect(self._generate_residual_report)
        residual_layout.addWidget(residual_btn)
        
        group_residual.setLayout(residual_layout)
        layout.addWidget(group_residual)
        
        # 4. AI 异常点检测
        group_ai = QGroupBox("AI 异常点检测")
        ai_layout = QVBoxLayout()
        
        self.use_dbscan = QCheckBox("使用 DBSCAN 聚类检测粗差")
        self.use_dbscan.setChecked(True)
        ai_layout.addWidget(self.use_dbscan)
        
        self.use_isolation = QCheckBox("使用孤立森林检测异常点")
        self.use_isolation.setChecked(True)
        ai_layout.addWidget(self.use_isolation)
        
        self.dbscan_eps = QDoubleSpinBox()
        self.dbscan_eps.setRange(0.1, 5.0)
        self.dbscan_eps.setValue(0.5)
        self.dbscan_eps.setSingleStep(0.1)
        eps_layout = QHBoxLayout()
        eps_layout.addWidget(QLabel("邻域半径:"))
        eps_layout.addWidget(self.dbscan_eps)
        
        self.dbscan_min = QSpinBox()
        self.dbscan_min.setRange(2, 20)
        self.dbscan_min.setValue(5)
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("最小样本:"))
        min_layout.addWidget(self.dbscan_min)
        
        ai_layout.addLayout(eps_layout)
        ai_layout.addLayout(min_layout)
        
        detect_btn = QPushButton("执行异常点检测")
        detect_btn.clicked.connect(self._detect_outliers)
        detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #7b1fa2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #6a1b9a; }
        """)
        ai_layout.addWidget(detect_btn)
        
        group_ai.setLayout(ai_layout)
        layout.addWidget(group_ai)
        
        # 结果显示区
        group_result = QGroupBox("处理结果")
        result_layout = QVBoxLayout()
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                padding: 5px;
            }
        """)
        result_layout.addWidget(self.result_text)
        
        group_result.setLayout(result_layout)
        layout.addWidget(group_result)
        
        layout.addStretch()
        
        return widget
    
    def _get_current_points(self) -> Optional[np.ndarray]:
        """获取当前点位数据（从工作空间）"""
        # 模拟生成测试点数据
        # 实际应从工作空间或文件中读取
        n_points = 100
        points = np.random.randn(n_points, 3) * 10
        points[:, 2] = np.abs(points[:, 2]) + 100  # 确保高程为正
        return points
    
    def _update_workspace_and_view(self, result: np.ndarray, process_type: str):
        """
        更新工作空间并刷新视图
        
        Args:
            result: 处理结果图像/数据
            process_type: 处理类型名称
        """
        # 使用处理类型 + 时间戳生成唯一名称
        timestamp = datetime.now().strftime("%H%M%S")
        result_name = f"{process_type}_{timestamp}"
        
        # 保存到 workspace
        self.workspace.add_processed_image(result_name, result)
        
        # 通过 EventBus 发布图像更新事件
        event_bus = get_event_bus()
        processed_path = self.workspace.get_processed_image(result_name)
        if processed_path:
            event_bus.publish(EventTopics.TOPIC_IMAGE_UPDATED, {
                "path": processed_path,
                "name": result_name
            })
        
        log_manager.info(f"空三处理结果已保存: {result_name}")
    
    def _execute_relative_orientation(self):
        """执行相对定向"""
        try:
            # 获取/模拟控制点数据
            img1_points = np.random.randn(20, 2) * 100
            img2_points = np.random.randn(20, 2) * 100
            focal_length = self.focal_length.value()
            
            self.status_label.setText("正在执行相对定向...")
            
            # 执行相对定向
            result = RelativeOrientation.compute_relative_orientation(
                img1_points, img2_points, focal_length
            )
            
            if result.get("success"):
                # 生成模拟结果图像（用于展示）
                result_image = self._generate_result_visualization("相对定向", result)
                
                # 更新结果文本
                self.result_text.setText(
                    f"相对定向完成\n"
                    f"方法: {self.relative_method.currentText()}\n"
                    f"点数: {result.get('point_count', 0)}\n"
                    f"Omega: {result.get('omega', 0):.6f} rad\n"
                    f"Phi: {result.get('phi', 0):.6f} rad\n"
                    f"Kappa: {result.get('kappa', 0):.6f} rad\n"
                    f"RMSE: {result.get('rmse', 0):.4f} pixel"
                )
                
                # 更新工作空间和视图
                self._update_workspace_and_view(result_image, f"相对定向_{self.relative_method.currentText()}")
                
                self.status_label.setText("相对定向完成")
                log_manager.info(f"相对定向完成: RMSE={result.get('rmse', 0):.4f}")
            else:
                QMessageBox.warning(None, "警告", result.get("error", "相对定向失败"))
                self.status_label.setText("相对定向失败")
                
        except Exception as e:
            QMessageBox.critical(None, "错误", f"相对定向失败: {str(e)}")
            self.status_label.setText("相对定向失败")
            log_manager.error(f"相对定向失败: {e}")
    
    def _execute_bundle_adjustment(self):
        """执行区域网平差"""
        try:
            self.status_label.setText("正在执行区域网平差...")
            
            # 模拟控制点和连接点
            control_points = {
                f"CP{i}": (np.random.randn() * 1000, np.random.randn() * 1000, np.random.randn() * 100 + 500)
                for i in range(10)
            }
            
            tie_points = [{"id": i, "observations": np.random.randn(4, 2)} for i in range(50)]
            
            interior_orientation = {
                "x0": 0, "y0": 0, "f": self.focal_length.value()
            }
            
            method = self.bundle_method.currentText()
            
            if method == "间接平差":
                result = BundleAdjustment.indirect_adjustment(
                    control_points, tie_points, interior_orientation
                )
            else:
                # 光束法平差
                image_observations = [{"image_id": i, "points": np.random.randn(20, 2)} for i in range(5)]
                result = BundleAdjustment.bundle_adjustment(
                    image_observations, control_points, interior_orientation
                )
            
            if result.get("success"):
                # 生成结果图像
                result_image = self._generate_result_visualization("区域网平差", result)
                
                # 更新结果文本
                self.result_text.setText(
                    f"区域网平差完成\n"
                    f"方法: {result.get('method', method)}\n"
                    f"迭代次数: {result.get('iterations', '-')}\n"
                    f"单位权中误差: {result.get('sigma0', result.get('final_rmse', '-')):.4f}\n"
                    f"点位RMSE: {result.get('point_rmse', '-'):.4f}\n"
                    f"观测数: {result.get('observations', result.get('images', '-'))}\n"
                    f"未知数: {result.get('unknowns', '-')}"
                )
                
                # 更新工作空间和视图
                self._update_workspace_and_view(result_image, f"平差_{method}")
                
                self.status_label.setText("区域网平差完成")
                log_manager.info(f"区域网平差完成: {result.get('method')}")
            else:
                QMessageBox.warning(None, "警告", result.get("error", "平差失败"))
                self.status_label.setText("区域网平差失败")
                
        except Exception as e:
            QMessageBox.critical(None, "错误", f"区域网平差失败: {str(e)}")
            self.status_label.setText("区域网平差失败")
            log_manager.error(f"区域网平差失败: {e}")
    
    def _generate_residual_report(self):
        """生成残差报告"""
        try:
            # 模拟残差数据
            np.random.seed(42)
            residuals = np.random.randn(100) * 2
            observations = np.random.randn(100) * 100
            
            self.status_label.setText("正在分析残差...")
            
            result = ResidualAnalysis.analyze_residuals(residuals, observations)
            
            if self.show_residual_plot.isChecked():
                plot_data = ResidualAnalysis.generate_residual_plot_data(residuals)
            
            # 显示结果
            report = f"""
残差分析报告
=============

基本统计:
  - 残差数量: {result['count']}
  - 平均值: {result['mean']:.6f}
  - 标准差: {result['std']:.6f}
  - 最大残差: {result['max']:.6f}
  - 最小残差: {result['min']:.6f}

精度指标:
  - RMSE: {result['rmse']:.6f}
  - 中误差: {result['sigma']:.6f}

异常检验:
  - 3σ外点数: {result['outliers_3sigma']}
  - 异常比例: {result['outliers_percentage']:.2f}%

分布特性:
  - 偏度: {result['skewness']:.4f}
  - 峰度: {result['kurtosis']:.4f}

诊断结论: {result['status']}
            """
            
            self.result_text.setText(report.strip())
            self.status_label.setText("残差分析完成")
            log_manager.info(f"残差分析完成: {result['status']}")
            
        except Exception as e:
            QMessageBox.critical(None, "错误", f"残差分析失败: {str(e)}")
            log_manager.error(f"残差分析失败: {e}")
    
    def _detect_outliers(self):
        """执行异常点检测"""
        try:
            # 获取点数据
            points = self._get_current_points()
            
            if points is None:
                QMessageBox.warning(None, "警告", "没有可用的点数据")
                return
            
            self.status_label.setText("正在执行异常点检测...")
            
            use_dbscan = self.use_dbscan.isChecked()
            use_isolation = self.use_isolation.isChecked()
            
            if not use_dbscan and not use_isolation:
                QMessageBox.warning(None, "警告", "请至少选择一种检测方法")
                return
            
            # 执行综合检测
            result = OutlierDetection.combined_detection(
                points,
                use_dbscan=use_dbscan,
                use_isolation=use_isolation
            )
            
            if result.get("success"):
                # 生成结果图像
                result_image = self._generate_outlier_visualization(points, result)
                
                # 显示结果
                methods = ", ".join(result.get("methods_used", []))
                report = f"""
AI 异常点检测完成
==================

检测方法: {methods}

综合结果:
  - 检测到的异常点数: {result['n_outliers']}
  - 异常点比例: {result['outlier_percentage']:.2f}%
  - 总点数: {len(points)}
                """
                
                if "dbscan" in result.get("individual_results", {}):
                    db = result["individual_results"]["dbscan"]
                    report += f"\n\nDBSCAN结果:\n  - 聚类数: {db['n_clusters']}\n  - 噪声点: {db['n_noise']}"
                
                if "isolation_forest" in result.get("individual_results", {}):
                    iso = result["individual_results"]["isolation_forest"]
                    report += f"\n\n孤立森林结果:\n  - 异常点数: {iso['n_outliers']}"
                
                self.result_text.setText(report.strip())
                
                # 更新工作空间和视图
                self._update_workspace_and_view(result_image, "异常点检测")
                
                self.status_label.setText("异常点检测完成")
                log_manager.info(f"异常点检测完成: {result['n_outliers']} 个异常点")
            else:
                QMessageBox.warning(None, "警告", result.get("error", "检测失败"))
                self.status_label.setText("异常点检测失败")
                
        except Exception as e:
            QMessageBox.critical(None, "错误", f"异常点检测失败: {str(e)}")
            self.status_label.setText("异常点检测失败")
            log_manager.error(f"异常点检测失败: {e}")
    
    def _generate_result_visualization(self, title: str, data: Dict) -> np.ndarray:
        """生成结果可视化图像"""
        # 创建结果图像
        height, width = 400, 600
        image = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        # 添加标题
        # (实际显示需要配套的渲染组件)
        
        return image
    
    def _generate_outlier_visualization(self, points: np.ndarray, result: Dict) -> np.ndarray:
        """生成异常点检测可视化图像"""
        height, width = 400, 600
        image = np.ones((height, width, 3), dtype=np.uint8) * 240
        
        return image
    
    def execute(self, *args, **kwargs):
        """执行处理（被外部调用）"""
        pass

from .processors import AerialTriangulationProcessor, AerialTriangulationResult, _load_bgr_image


class AerialTriangulationPlugin(IPlugin):
    """模块二：空中三角测量 - 可运行版"""

    def __init__(self, workspace):
        super().__init__(workspace)
        self.workspace = workspace
        self._event_bus = get_event_bus()
        self._processor = AerialTriangulationProcessor()
        self._panel = None
        self._session = None

        self.status_label = None
        self.summary_text = None
        self.left_combo = None
        self.right_combo = None
        self.feature_method_combo = None
        self.matcher_combo = None
        self.feature_count_spin = None
        self.ratio_spin = None
        self.focal_scale_spin = None
        self.ransac_spin = None
        self.max_points_spin = None
        self.dbscan_eps_spin = None
        self.dbscan_min_spin = None
        self.contamination_spin = None

        self._event_bus.subscribe(EventTopics.TOPIC_IMAGE_ADDED, self._on_workspace_change)
        self._event_bus.subscribe(EventTopics.TOPIC_IMAGE_REMOVED, self._on_workspace_change)
        self._event_bus.subscribe(EventTopics.TOPIC_PROJECT_OPENED, self._on_workspace_change)
        self._event_bus.subscribe(EventTopics.TOPIC_PROJECT_NEW, self._on_workspace_change)

    def plugin_info(self) -> Dict[str, Any]:
        return {
            "name": "空中三角测量",
            "group": "模块",
            "version": "1.1.0",
            "description": "基于真实工作区影像的相对定向、区域网平差、残差分析与异常点检测",
        }

    def get_ui_panel(self) -> QWidget:
        if self._panel is not None:
            self._refresh_image_lists()
            return self._panel

        from PySide6.QtWidgets import QScrollArea

        panel = QWidget()
        root_layout = QVBoxLayout(panel)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        title = QLabel("空中三角测量")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976d2;")
        layout.addWidget(title)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #666; padding: 4px 2px;")
        layout.addWidget(self.status_label)

        image_group = QGroupBox("影像选择")
        image_form = QFormLayout()
        self.left_combo = QComboBox()
        self.right_combo = QComboBox()
        refresh_btn = QPushButton("刷新影像列表")
        refresh_btn.clicked.connect(self._refresh_image_lists)
        image_form.addRow("左影像:", self.left_combo)
        image_form.addRow("右影像:", self.right_combo)
        image_form.addRow("", refresh_btn)
        image_group.setLayout(image_form)
        layout.addWidget(image_group)

        param_group = QGroupBox("处理参数")
        param_form = QFormLayout()

        self.feature_method_combo = QComboBox()
        self.feature_method_combo.addItems(["SIFT", "ORB"])
        param_form.addRow("特征算法:", self.feature_method_combo)

        self.matcher_combo = QComboBox()
        self.matcher_combo.addItems(["AUTO", "FLANN", "BF"])
        param_form.addRow("匹配器:", self.matcher_combo)

        self.feature_count_spin = QSpinBox()
        self.feature_count_spin.setRange(200, 10000)
        self.feature_count_spin.setValue(2500)
        self.feature_count_spin.setSingleStep(250)
        param_form.addRow("特征数量:", self.feature_count_spin)

        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(0.4, 0.95)
        self.ratio_spin.setSingleStep(0.01)
        self.ratio_spin.setDecimals(2)
        self.ratio_spin.setValue(0.75)
        param_form.addRow("比值阈值:", self.ratio_spin)

        self.focal_scale_spin = QDoubleSpinBox()
        self.focal_scale_spin.setRange(0.5, 3.0)
        self.focal_scale_spin.setSingleStep(0.05)
        self.focal_scale_spin.setDecimals(2)
        self.focal_scale_spin.setValue(1.20)
        param_form.addRow("像素焦距系数:", self.focal_scale_spin)

        self.ransac_spin = QDoubleSpinBox()
        self.ransac_spin.setRange(0.2, 8.0)
        self.ransac_spin.setSingleStep(0.1)
        self.ransac_spin.setDecimals(2)
        self.ransac_spin.setValue(1.5)
        param_form.addRow("RANSAC阈值:", self.ransac_spin)

        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(20, 400)
        self.max_points_spin.setValue(120)
        self.max_points_spin.setSingleStep(10)
        param_form.addRow("平差点数上限:", self.max_points_spin)

        self.dbscan_eps_spin = QDoubleSpinBox()
        self.dbscan_eps_spin.setRange(0.1, 5.0)
        self.dbscan_eps_spin.setSingleStep(0.1)
        self.dbscan_eps_spin.setDecimals(2)
        self.dbscan_eps_spin.setValue(0.8)
        param_form.addRow("DBSCAN eps:", self.dbscan_eps_spin)

        self.dbscan_min_spin = QSpinBox()
        self.dbscan_min_spin.setRange(2, 20)
        self.dbscan_min_spin.setValue(5)
        param_form.addRow("DBSCAN min_samples:", self.dbscan_min_spin)

        self.contamination_spin = QDoubleSpinBox()
        self.contamination_spin.setRange(0.01, 0.40)
        self.contamination_spin.setSingleStep(0.01)
        self.contamination_spin.setDecimals(2)
        self.contamination_spin.setValue(0.12)
        param_form.addRow("异常比例:", self.contamination_spin)

        param_group.setLayout(param_form)
        layout.addWidget(param_group)

        action_group = QGroupBox("处理流程")
        action_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        btn_orientation = QPushButton("执行相对定向")
        btn_orientation.clicked.connect(self._run_relative_orientation)
        row1.addWidget(btn_orientation)

        btn_bundle = QPushButton("执行区域网平差")
        btn_bundle.clicked.connect(self._run_bundle_adjustment)
        row1.addWidget(btn_bundle)
        action_layout.addLayout(row1)

        row2 = QHBoxLayout()
        btn_residual = QPushButton("残差分析")
        btn_residual.clicked.connect(self._run_residual_analysis)
        row2.addWidget(btn_residual)

        btn_outlier = QPushButton("异常点检测")
        btn_outlier.clicked.connect(self._run_outlier_detection)
        row2.addWidget(btn_outlier)
        action_layout.addLayout(row2)

        row3 = QHBoxLayout()
        btn_cloud = QPushButton("3D点云预览")
        btn_cloud.clicked.connect(self._run_point_cloud_preview)
        row3.addWidget(btn_cloud)

        btn_all = QPushButton("执行完整流程")
        btn_all.clicked.connect(self._run_full_pipeline)
        row3.addWidget(btn_all)
        action_layout.addLayout(row3)

        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        result_group = QGroupBox("结果摘要")
        result_layout = QVBoxLayout()
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(220)
        result_layout.addWidget(self.summary_text)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        layout.addStretch(1)
        scroll.setWidget(container)
        root_layout.addWidget(scroll)

        self._panel = panel
        self._refresh_image_lists()
        self._set_summary([
            "请选择左右影像，然后执行相对定向。",
            "输出会同步显示在中间视图、右侧摘要和底部日志中。",
        ])
        return panel

    def on_activate(self):
        self._refresh_image_lists()

    def _on_workspace_change(self, data=None):
        self._refresh_image_lists()

    def _refresh_image_lists(self):
        if self.left_combo is None or self.right_combo is None:
            return

        previous_left = self.left_combo.currentText()
        previous_right = self.right_combo.currentText()
        image_names = list(self.workspace.get_all_images().keys())

        self.left_combo.blockSignals(True)
        self.right_combo.blockSignals(True)
        self.left_combo.clear()
        self.right_combo.clear()
        self.left_combo.addItems(image_names)
        self.right_combo.addItems(image_names)

        if image_names:
            left_index = image_names.index(previous_left) if previous_left in image_names else 0
            right_index = image_names.index(previous_right) if previous_right in image_names else min(1, len(image_names) - 1)
            self.left_combo.setCurrentIndex(left_index)
            self.right_combo.setCurrentIndex(right_index)

        self.left_combo.blockSignals(False)
        self.right_combo.blockSignals(False)

    def _set_status(self, text: str):
        if self.status_label is not None:
            self.status_label.setText(text)

    def _set_summary(self, lines):
        if self.summary_text is None:
            return
        if isinstance(lines, str):
            self.summary_text.setPlainText(lines)
        else:
            self.summary_text.setPlainText("\n".join(lines))

    def _resolve_selected_images(self):
        if self.left_combo is None or self.right_combo is None:
            return None, None, None, None
        left_name = self.left_combo.currentText().strip()
        right_name = self.right_combo.currentText().strip()
        if not left_name or not right_name:
            return left_name or None, right_name or None, None, None
        left_path = self.workspace.get_image(left_name)
        right_path = self.workspace.get_image(right_name)
        if not left_path or not right_path:
            return left_name, right_name, None, None
        left_img = _load_bgr_image(left_path)
        right_img = _load_bgr_image(right_path)
        return left_name, right_name, left_img, right_img

    def _ensure_pair(self):
        left_name, right_name, left_img, right_img = self._resolve_selected_images()
        if left_img is None or right_img is None:
            images = list(self.workspace.get_all_images().items())
            if len(images) < 2:
                QMessageBox.warning(self._panel or None, "提示", "请先导入至少两张影像。")
                self._set_status("请先导入至少两张影像")
                return left_name, right_name, None, None
            if left_img is None:
                left_name, left_path = images[0]
                left_img = _load_bgr_image(left_path)
            if right_img is None:
                right_name, right_path = images[1]
                right_img = _load_bgr_image(right_path)
        if left_img is None or right_img is None:
            QMessageBox.warning(self._panel or None, "提示", "影像读取失败。")
            self._set_status("影像读取失败")
            return left_name, right_name, None, None
        return left_name, right_name, left_img, right_img

    def _publish_processed_canvas(self, canvas, prefix: str, title: str, summary: str):
        if canvas is None:
            return None
        name = f"{prefix}_{datetime.now().strftime('%H%M%S')}"
        self.workspace.add_processed_image(name, canvas)
        path = self.workspace.get_processed_image(name)
        if path:
            self._event_bus.publish(EventTopics.TOPIC_IMAGE_UPDATED, {
                "name": name,
                "path": path,
                "title": title,
                "summary": summary,
            })
        return name

    def _publish_compare_view(self, result, title: str):
        if result.left_image is None or result.right_image is None:
            return
        self._event_bus.publish(EventTopics.TOPIC_VIEW_COMPARE_REQUEST, {
            "left": {"image": result.left_image, "name": "左影像"},
            "right": {"image": result.right_image, "name": "右影像"},
            "title": title,
            "left_keypoints": result.left_keypoints,
            "right_keypoints": result.right_keypoints,
            "matches": result.compare_matches or [],
            "sync": True,
        })

    def _publish_point_cloud(self, result, title: str):
        if result.points_3d is None or len(result.points_3d) == 0:
            return
        self._event_bus.publish(EventTopics.TOPIC_VIEW_3D_REQUEST, {
            "kind": "point_cloud",
            "points": result.points_3d,
            "colors": result.colors,
            "title": title,
        })

    def _format_summary(self, prefix: str, summary):
        lines = [prefix]
        for key, value in summary.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.6f}")
            else:
                lines.append(f"{key}: {value}")
        return lines

    def _run_relative_orientation(self):
        left_name, right_name, left_img, right_img = self._ensure_pair()
        if left_img is None or right_img is None:
            return None

        self._set_status("正在执行相对定向...")
        log_manager.info(f"模块二：开始相对定向，影像 {left_name} vs {right_name}")
        result = self._processor.relative_orientation(
            left_img,
            right_img,
            feature_method=self.feature_method_combo.currentText() if self.feature_method_combo else "SIFT",
            matcher_method=self.matcher_combo.currentText() if self.matcher_combo else "AUTO",
            n_features=int(self.feature_count_spin.value()) if self.feature_count_spin else 2500,
            ratio=float(self.ratio_spin.value()) if self.ratio_spin else 0.75,
            ransac_threshold=float(self.ransac_spin.value()) if self.ransac_spin else 1.5,
        )
        self._session = result

        if not result.success:
            self._set_summary(self._format_summary("相对定向失败", result.summary))
            if result.overview_canvas is not None:
                self._publish_processed_canvas(result.overview_canvas, "空三_相对定向失败", "相对定向", result.message)
            log_manager.warning(f"模块二：相对定向失败 - {result.message}")
            self._set_status("相对定向失败")
            return result

        summary_lines = self._format_summary("相对定向结果", result.summary)
        self._set_summary(summary_lines)
        self._publish_processed_canvas(result.overview_canvas, "空三_相对定向", "相对定向", " / ".join(summary_lines[:4]))
        self._publish_compare_view(result, f"相对定向 - {left_name} vs {right_name}")
        self._publish_point_cloud(result, "相对定向稀疏点云")
        self._set_status("相对定向完成")
        log_manager.info(
            f"模块二：相对定向完成，匹配 {result.summary.get('matches', 0)}，内点 {result.summary.get('inliers', 0)}，"
            f"RMSE {result.summary.get('reprojection_rmse')}"
        )
        return result

    def _run_bundle_adjustment(self):
        result = self._session or self._run_relative_orientation()
        if not result or not result.success:
            return None

        self._set_status("正在执行区域网平差...")
        log_manager.info("模块二：开始区域网平差")
        bundle = self._processor.bundle_adjustment(result, max_points=int(self.max_points_spin.value()) if self.max_points_spin else 120)
        if not bundle.success:
            self._set_summary(["区域网平差失败", bundle.message])
            self._set_status("区域网平差失败")
            log_manager.warning(f"模块二：区域网平差失败 - {bundle.message}")
            return bundle

        bundle.left_image = result.left_image
        bundle.right_image = result.right_image
        bundle.left_keypoints = result.left_keypoints
        bundle.right_keypoints = result.right_keypoints
        bundle.compare_matches = result.compare_matches
        bundle.match_mask = result.match_mask
        self._session = bundle
        self._set_summary(self._format_summary("区域网平差结果", bundle.summary))
        self._publish_processed_canvas(bundle.overview_canvas, "空三_区域网平差", "区域网平差", " / ".join(self._format_summary("区域网平差结果", bundle.summary)[:4]))
        self._publish_point_cloud(bundle, "平差后点云预览")
        self._set_status("区域网平差完成")
        log_manager.info(
            f"模块二：区域网平差完成，方法 {bundle.summary.get('method')}，RMSE {bundle.summary.get('after_rmse')}"
        )
        return bundle

    def _run_residual_analysis(self):
        result = self._session or self._run_relative_orientation()
        if not result or not result.success:
            return None

        residuals = result.residuals
        if residuals is None and getattr(result, "points_3d", None) is not None:
            residuals = np.asarray(result.points_3d, dtype=np.float64)
        analysis = self._processor.analyze_residuals(residuals)
        if not analysis.get("success"):
            QMessageBox.warning(self._panel or None, "提示", analysis.get("message", "残差分析失败"))
            return analysis

        lines = [
            "残差分析结果",
            f"count: {analysis['count']}",
            f"mean: {analysis['mean']:.6f}",
            f"std: {analysis['std']:.6f}",
            f"rmse: {analysis['rmse']:.6f}",
            f"outliers(3σ): {analysis['outliers_3sigma']}",
            f"status: {analysis['status']}",
        ]
        self._set_summary(lines)
        self._publish_processed_canvas(analysis["canvas"], "空三_残差分析", "残差分析", " / ".join(lines[:4]))
        self._set_status("残差分析完成")
        log_manager.info(f"模块二：残差分析完成，状态 {analysis['status']}，RMSE {analysis['rmse']:.6f}")
        return analysis

    def _run_outlier_detection(self):
        result = self._session or self._run_relative_orientation()
        if not result or not result.success:
            return None

        residuals = result.residuals
        if residuals is None:
            QMessageBox.warning(self._panel or None, "提示", "当前会话中没有可用于异常点检测的残差。")
            return None

        detection = self._processor.detect_outliers(
            residuals,
            points_3d=result.points_3d,
            use_dbscan=True,
            use_isolation=True,
            eps=float(self.dbscan_eps_spin.value()) if self.dbscan_eps_spin else 0.8,
            min_samples=int(self.dbscan_min_spin.value()) if self.dbscan_min_spin else 5,
            contamination=float(self.contamination_spin.value()) if self.contamination_spin else 0.12,
        )
        if not detection.get("success"):
            QMessageBox.warning(self._panel or None, "提示", detection.get("message", "异常点检测失败"))
            return detection

        base_matches = list(result.compare_matches or [])
        mask = detection["outlier_mask"]
        for idx, match in enumerate(base_matches):
            match["inlier"] = not bool(mask[idx]) if idx < len(mask) else match.get("inlier", True)

        self._event_bus.publish(EventTopics.TOPIC_VIEW_COMPARE_REQUEST, {
            "left": {"image": result.left_image, "name": "左影像"},
            "right": {"image": result.right_image, "name": "右影像"},
            "title": "异常点检测结果",
            "left_keypoints": result.left_keypoints,
            "right_keypoints": result.right_keypoints,
            "matches": base_matches,
            "sync": True,
        })

        lines = [
            "异常点检测结果",
            f"n_outliers: {detection['n_outliers']}",
            f"ratio: {detection['outlier_percentage']:.2f}%",
            f"methods: {', '.join(detection.get('methods_used', []))}",
        ]
        self._set_summary(lines)
        self._publish_processed_canvas(detection["canvas"], "空三_异常点检测", "异常点检测", " / ".join(lines[:4]))
        self._set_status("异常点检测完成")
        log_manager.info(f"模块二：异常点检测完成，异常点 {detection['n_outliers']} 个")
        return detection

    def _run_point_cloud_preview(self):
        result = self._session or self._run_relative_orientation()
        if not result or not result.success or result.points_3d is None:
            return None
        self._publish_point_cloud(result, "空三稀疏点云")
        self._set_status("点云已显示")
        log_manager.info("模块二：已显示三维点云")
        return result

    def _run_full_pipeline(self):
        result = self._run_relative_orientation()
        if not result or not result.success:
            return result
        bundle = self._run_bundle_adjustment()
        self._run_residual_analysis()
        self._run_outlier_detection()
        self._run_point_cloud_preview()
        return bundle or result

    def execute(self, *args, **kwargs):
        bundle = self._run_full_pipeline()
        if bundle and getattr(bundle, "success", False):
            return {
                "success": True,
                "message": bundle.message,
                "result_name": "空中三角测量完整流程",
                "summary": bundle.summary,
            }
        return {
            "success": False,
            "message": "空中三角测量执行失败",
        }
