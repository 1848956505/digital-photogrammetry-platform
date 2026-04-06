"""
高级2D影像渲染引擎 (ImageViewer)
基于QGraphicsView，支持无限漫游和专业缩放平移
"""
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                               QGraphicsItem, QScrollArea, QLabel, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QSize, QPoint
from PySide6.QtGui import QImage, QPixmap, QWheelEvent, QMouseEvent, QPainter, QPen, QColor

from core.workspace import get_workspace
from core.event_bus import EventTopics, get_event_bus
from core.log_manager import log_manager


class ImageGraphicsItem(QGraphicsPixmapItem):
    """自定义图像项，支持鼠标追踪"""
    
    def __init__(self, pixmap=None, parent=None):
        super().__init__(pixmap)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        self.setFlag(QGraphicsItem.ItemIsMovable, False)
        self._image_size = (0, 0)
    
    def set_image_size(self, size):
        """设置原始图像尺寸"""
        self._image_size = size
    
    def image_size(self):
        return self._image_size


class ImageViewer(QGraphicsView):
    """
    高级2D渲染引擎
    
    功能特性：
    - 滚轮以鼠标为中心缩放
    - 按住鼠标中键/左键平移
    - 多图层支持
    - 坐标追踪
    """
    
    # 信号
    mouse_moved = Signal(int, int, int, int, int)  # (col, row, r, g, b)
    image_loaded = Signal(str)
    zoom_changed = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 初始化场景
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        
        # 图像项
        self._image_item = None
        self._current_image_path = None
        self._image_array = None
        
        # 缩放相关
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        self._zoom_step = 1.15
        
        # 平移相关
        self._is_panning = False
        self._pan_start = QPoint()
        
        # 坐标转换
        self._image_size = (0, 0)
        
        # 交互设置
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)  # 平移时设为NoAnchor
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        
        # 背景
        self.setStyleSheet("background-color: #f5f5f5;")
        
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # 居中显示
        self.setAlignment(Qt.AlignCenter)
        
        # 十字丝样式
        self._show_crosshair = False
        
        # 初始化工作空间
        self._workspace = get_workspace()
        
        # 事件总线
        self._event_bus = get_event_bus()
        
        # 显示初始提示
        self._show_welcome()

        # 关键设置
        self.setDragMode(QGraphicsView.DragMode.NoDrag)  # 不使用内置拖拽
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # 强制滚动条策略
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # 启用鼠标追踪
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        
        # 订阅 EventBus 事件
        self._event_bus.subscribe(EventTopics.TOPIC_IMAGE_SELECTED, self._on_image_selected_event)
        self._event_bus.subscribe(EventTopics.TOPIC_IMAGE_UPDATED, self._on_image_updated_event)
    
    def _show_welcome(self):
        """显示欢迎信息"""
        self._scene.clear()
        self._image_item = None
        
        # 欢迎文字
        text_item = self._scene.addText("请导入影像\n\n支持格式: PNG, JPG, TIFF, BMP\n\n操作:\n• 滚轮缩放\n• 中键/左键平移\n• 拖放文件导入")
        text_item.setDefaultTextColor(QColor("#999999"))
        text_item.setFont(self.scene().font())
        
        # 居中
        rect = text_item.boundingRect()
        text_item.setPos(-rect.width()/2, -rect.height()/2)
    
    def load_image(self, path: str) -> bool:
        """
        加载图像
        
        Args:
            path: 图像文件路径
            
        Returns:
            bool: 是否加载成功
        """
        if not os.path.exists(path):
            return False
        
        try:
            # 解决 OpenCV 中文路径问题
            # 方法：使用 np.fromfile 读取二进制，再解码
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return False
            
            # BGR转RGB
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            self._image_array = img
            self._image_size = (img.shape[1], img.shape[0])
            self._current_image_path = path
            
            # 转换并显示
            self._display_image()
            
            # 发送信号
            self.image_loaded.emit(path)
            
            # 注册到工作空间
            name = os.path.splitext(os.path.basename(path))[0]
            self._workspace.add_image(name, path)
            
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def _display_image(self):
        """在场景中显示图像"""
        self._scene.clear()
        
        if self._image_array is None:
            self._show_welcome()
            return
        
        h, w = self._image_array.shape[:2]
        bytes_per_line = w * 3
        
        # 创建QImage
        qimage = QImage(
            self._image_array.data,
            w, h,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        # 创建QPixmap
        pixmap = QPixmap.fromImage(qimage)
        
        # 创建图形项
        self._image_item = ImageGraphicsItem(pixmap)
        self._image_item.set_image_size(self._image_size)
        self._scene.addItem(self._image_item)

        # 【关键修复】：设置一个巨大的场景范围，允许无限平移
        # 比如图像四周扩展 100,000 像素
        self._scene.setSceneRect(-50000, -50000, 100000 + self._image_size[0], 100000 + self._image_size[1])
        
        # 重置视图
        self._zoom_factor = 1.0
        self._pan_offset = QPointF(0, 0)
        
        # 适应窗口
        self.fit_to_window()

    def fit_to_window(self):
        """重置视图：将图像完整居中显示在窗口内"""
        if self._image_item is None:
            return

        # 1. 获取图像的边界矩形 (通常是 0, 0, width, height)
        rect = self._image_item.boundingRect()
        if rect.isEmpty():
            return

        # 2. 核心：使用 fitInView 自动计算缩放并居中
        # Qt.AspectRatioMode.KeepAspectRatio 确保图像不因拉伸变形
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

        # 3. 关键：同步更新内部的 _zoom_factor
        # 否则下次滚动鼠标滚轮时，缩放比例会从旧的值突跳
        # transform().m11() 获取当前的水平缩放比例
        self._zoom_factor = self.transform().m11()

        # 4. 发射缩放改变信号，让状态栏等 UI 更新
        self.zoom_changed.emit(self._zoom_factor)

    def _apply_zoom(self):
        """应用缩放变换"""
        # 保存当前视图中心
        center = self.mapToScene(self.viewport().rect().center())

        # 重置并应用缩放
        self.resetTransform()
        self.scale(self._zoom_factor, self._zoom_factor)

        # 保持中心点不变
        self.centerOn(center)

        self.zoom_changed.emit(self._zoom_factor)

    def wheelEvent(self, event: QWheelEvent):
        # 无论是否有图像，都处理滚轮事件
        # 如果没有图像，返回让父类处理（可以滚动欢迎文字）
        if self._image_item is None:
            super().wheelEvent(event)
            return

        # 获取缩放前的鼠标场景位置
        old_scene_pos = self.mapToScene(event.position().toPoint())

        # 计算缩放
        zoom_in = event.angleDelta().y() > 0
        factor = self._zoom_step if zoom_in else 1 / self._zoom_step

        new_zoom = self._zoom_factor * factor
        if self._min_zoom <= new_zoom <= self._max_zoom:
            self._zoom_factor = new_zoom

            # 应用缩放
            self.resetTransform()
            self.scale(self._zoom_factor, self._zoom_factor)

            # 获取缩放后的鼠标场景位置并计算偏移，实现以鼠标为中心缩放
            new_scene_pos = self.mapToScene(event.position().toPoint())
            delta = new_scene_pos - old_scene_pos

            # 调整视口位置
            self.translate(delta.x(), delta.y())

            self.zoom_changed.emit(self._zoom_factor)

        event.accept()
    
    def _zoom_at(self, target_zoom: float, center_point: QPointF):
        """以指定点为中心缩放"""
        old_zoom = self._zoom_factor
        self._zoom_factor = target_zoom
        
        # 计算新位置
        viewport_center = self.viewport().rect().center()
        
        # 让指定点移动到视口中心
        self.centerOn(center_point)
        
        # 应用缩放变换
        self._apply_zoom()
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件 - 开始平移"""
        if event.button() == Qt.MiddleButton or \
           (event.button() == Qt.LeftButton and event.modifiers() == Qt.NoModifier):
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.OpenHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件 - 平移和坐标追踪"""
        # 1. 处理平移逻辑
        if self._is_panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()

            # 修改滚动条数值实现平移
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            # 注意：这里不要直接 return，让后面的坐标追踪也能跑

        # 2. 处理坐标追踪逻辑 (始终执行，即使在平移)
        if self._image_item is not None:
            scene_pos = self.mapToScene(event.pos())

            col = int(scene_pos.x())
            row = int(scene_pos.y())

            # 检查是否在图像范围内
            if 0 <= col < self._image_size[0] and 0 <= row < self._image_size[1]:
                # 获取像素值
                if self._image_array is not None:
                    # 注意处理 OpenCV 坐标 (row, col)
                    pixel = self._image_array[row, col]
                    if isinstance(pixel, np.ndarray):
                        r, g, b = pixel[:3]
                    else:
                        # 处理单通道灰度图
                        r = g = b = pixel
                    self.mouse_moved.emit(col, row, r, g, b)
            else:
                # 如果移出图像区域，发送特殊信号或清除状态
                self.mouse_moved.emit(col, row, -1, -1, -1)

        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件 - 结束平移"""
        if event.button() == Qt.MiddleButton or event.button() == Qt.LeftButton:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def zoom_in(self):
        """放大"""
        if self._image_item is not None:
            new_zoom = min(self._max_zoom, self._zoom_factor * self._zoom_step)
            if new_zoom != self._zoom_factor:
                center = self.viewport().rect().center()
                self._zoom_at(new_zoom, self.mapToScene(center))
    
    def zoom_out(self):
        """缩小"""
        if self._image_item is not None:
            new_zoom = max(self._min_zoom, self._zoom_factor / self._zoom_step)
            if new_zoom != self._zoom_factor:
                center = self.viewport().rect().center()
                self._zoom_at(new_zoom, self.mapToScene(center))
    

    

    
    def clear(self):
        """清空"""
        self._image_array = None
        self._current_image_path = None
        self._image_size = (0, 0)
        self._show_welcome()
    
    @property
    def has_image(self) -> bool:
        return self._image_array is not None
    
    @property
    def image_size(self) -> tuple:
        return self._image_size
    
    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor
    
    # ==================== EventBus 事件处理 ====================
    
    def _on_image_selected_event(self, data):
        """处理影像选中事件"""
        if isinstance(data, dict):
            path = data.get("path")
        else:
            path = data
        
        if path and self.load_image(path):
            name = os.path.splitext(os.path.basename(path))[0]
            log_manager.info(f"事件响应: 加载选中影像 {name}")
    
    def _on_image_updated_event(self, data):
        """处理图像更新事件（模块处理后刷新视图）"""
        if isinstance(data, dict):
            path = data.get("path")
        else:
            path = data
        
        if path and self.load_image(path):
            name = data.get("name", "处理结果") if isinstance(data, dict) else "处理结果"
            log_manager.info(f"事件响应: 刷新处理结果 {name}")


# 兼容性别名
GraphicsImageViewer = ImageViewer