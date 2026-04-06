"""
3D 点云视图组件
"""
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class PointCloudViewer(QWidget):
    """3D 点云视图"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.pointcloud_data = None
    
    def _setup_ui(self):
        """设置 UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel("暂无点云\n\n支持导入 .ply .pcd .las .laz 格式")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            background-color: #121212;
            color: #757575;
            font-size: 14px;
        """)
        layout.addWidget(self.label)
    
    def load_pointcloud(self, path: str):
        """加载点云"""
        ext = path.split('.')[-1].lower()
        if ext in ['ply', 'pcd', 'las', 'laz']:
            self.pointcloud_data = path
            self.label.setText(f"已加载点云: {path}\n\n渲染功能开发中...")
        else:
            self.label.setText("不支持的格式")
    
    def set_background_color(self, r: int, g: int, b: int):
        """设置背景色"""
        self.label.setStyleSheet(f"""
            background-color: rgb({r},{g},{b});
            color: #757575;
        """)
    
    def filter_ground_points(self, enable: bool):
        """过滤地面点"""
        # TODO: 实现 PointNet 地面点过滤
        pass
    
    def color_by_height(self, enable: bool):
        """按高度着色"""
        # TODO: 实现按高度渲染
        pass