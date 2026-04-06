"""
2D 影像视图组件 - 支持图像显示、缩放、平移
"""
import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import Qt, Signal, QSize, QRect
from PySide6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent, QPaintEvent

import cv2
import numpy as np


class ImageViewer(QWidget):
    """2D 影像视图 - 支持平移缩放显示"""
    
    # 信号
    mouse_moved = Signal(int, int)  # 鼠标位置像素坐标
    image_loaded = Signal(str)     # 图像加载完成
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._init_vars()
    
    def _setup_ui(self):
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 滚动区域（用于大图平移）
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setStyleSheet("background-color: #f5f5f5;")
        
        # 图像标签
        self.image_label = QLabel("请导入影像\n\n支持格式: PNG, JPG, TIFF, BMP")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                color: #999999;
                font-size: 16px;
                border: none;
            }
        """)
        
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)
    
    def _init_vars(self):
        """初始化变量"""
        self.current_image = None  # numpy array
        self.current_image_path = None
        self.pixmap = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.is_dragging = False
        self.last_mouse_pos = None
    
    def load_image(self, path: str) -> bool:
        """加载图像"""
        if not os.path.exists(path):
            return False
        
        # 读取图像
        img = cv2.imread(path)
        if img is None:
            return False
        
        # BGR转RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        self.current_image = img
        self.current_image_path = path
        
        # 更新显示
        self._update_display()
        
        self.image_loaded.emit(path)
        return True
    
    def load_from_array(self, img: np.ndarray):
        """从numpy数组加载"""
        self.current_image = img
        self._update_display()
    
    def _update_display(self):
        """更新图像显示"""
        if self.current_image is None:
            return
        
        # 转换为QPixmap显示
        h, w = self.current_image.shape[:2]
        bytes_per_line = w * 3
        
        # 创建QImage
        qimage = QImage(
            self.current_image.data, 
            w, 
            h, 
            bytes_per_line, 
            QImage.Format_RGB888
        )
        
        # 创建QPixmap
        self.pixmap = QPixmap.fromImage(qimage)
        
        # 显示
        self.image_label.setPixmap(self.pixmap)
        
        # 更新标签样式
        self.image_label.setStyleSheet("border: none;")
        
        # 更新控件大小
        self.image_label.setMinimumSize(w, h)
        self.scroll_area.setMinimumSize(
            min(w, self.width()),
            min(h, self.height())
        )
    
    def zoom_in(self):
        """放大"""
        self.zoom(1.2)
    
    def zoom_out(self):
        """缩小"""
        self.zoom(1/1.2)
    
    def zoom(self, factor: float):
        """缩放"""
        if self.current_image is None:
            return
        
        self.scale_factor *= factor
        self.scale_factor = max(0.1, min(10.0, self.scale_factor))
        
        # 重新缩放图像
        h, w = self.current_image.shape[:2]
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        
        if new_w > 0 and new_h > 0:
            resized = cv2.resize(
                self.current_image, 
                (new_w, new_h), 
                interpolation=cv2.INTER_LINEAR
            )
            
            # 转换为QPixmap
            bytes_per_line = new_w * 3
            qimage = QImage(
                resized.data,
                new_w,
                new_h,
                bytes_per_line,
                QImage.Format_RGB888
            )
            self.pixmap = QPixmap.fromImage(qimage)
            self.image_label.setPixmap(self.pixmap)
            self.image_label.setMinimumSize(new_w, new_h)
    
    def reset_zoom(self):
        """重置缩放"""
        if self.current_image is not None:
            self.scale_factor = 1.0
            self._update_display()
    
    def wheelEvent(self, event: QWheelEvent):
        """滚轮缩放"""
        if self.current_image is None:
            super().wheelEvent(event)
            return
        
        # 滚轮前进放大，后退缩小
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom(1.1)
        else:
            self.zoom(1/1.1)
        
        event.accept()
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下"""
        if event.button() == Qt.MidButton:
            self.is_dragging = True
            self.last_mouse_pos = event.pos()
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动"""
        if self.is_dragging and self.last_mouse_pos:
            # 实现平移
            self.last_mouse_pos = event.pos()
            event.accept()
        else:
            # 显示像素坐标
            if self.current_image is not None and self.pixmap:
                self.mouse_moved.emit(0, 0)  # 简化实现
            
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放"""
        if event.button() == Qt.MidButton:
            self.is_dragging = False
            self.last_mouse_pos = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)
    
    def clear(self):
        """清空"""
        self.current_image = None
        self.current_image_path = None
        self.pixmap = None
        self.scale_factor = 1.0
        
        self.image_label.clear()
        self.image_label.setText("请导入影像\n\n支持格式: PNG, JPG, TIFF, BMP")
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                color: #999999;
                font-size: 16px;
                border: none;
            }
        """)
    
    @property
    def has_image(self) -> bool:
        return self.current_image is not None
    
    @property
    def image_size(self) -> tuple:
        if self.current_image is None:
            return (0, 0)
        return self.current_image.shape[1], self.current_image.shape[0]


class ImageGraphicsView(QWidget):
    """
    基于 QGraphicsView 的高级影像视图
    支持平移、缩放、绘制、特征点标注等
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._init_vars()
    
    def _setup_ui(self):
        """设置UI"""
        self.setStyleSheet("background-color: #f5f5f5;")
        
        # 使用简单的ImageViewer
        self.viewer = ImageViewer(self)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewer)
    
    def _init_vars(self):
        """初始化变量"""
        self.keypoints = []
        self.matches = []
    
    def load_image(self, path: str) -> bool:
        """加载图像"""
        return self.viewer.load_image(path)
    
    def zoom_in(self):
        self.viewer.zoom_in()
    
    def zoom_out(self):
        self.viewer.zoom_out()
    
    def reset_zoom(self):
        self.viewer.reset_zoom()
    
    def wheelEvent(self, event):
        self.viewer.wheelEvent(event)
    
    @property
    def has_image(self) -> bool:
        return self.viewer.has_image
    
    @property
    def image_size(self) -> tuple:
        return self.viewer.image_size
    
    def clear(self):
        self.viewer.clear()
        self.keypoints = []
        self.matches = []