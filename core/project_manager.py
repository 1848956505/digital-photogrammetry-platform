"""
项目持久化管理器 (Project Manager)
实现工程的保存与加载
"""
import os
import json
from typing import Optional, Dict, Any
from PySide6.QtCore import QObject, Signal

from core.workspace import get_workspace


class ProjectManager(QObject):
    """
    项目持久化管理器
    
    功能：
    - 工程的保存（序列化）
    - 工程的加载（反序列化）
    - 自动保存支持
    """
    
    project_saved = Signal(str)     # (file_path)
    project_loaded = Signal(str)     # (file_path)
    project_modified = Signal(bool)  # (is_modified)
    
    def __init__(self):
        super().__init__()
        self._workspace = get_workspace()
        self._current_file_path: Optional[str] = None
        self._is_modified = False
        
        # 订阅工作空间变化
        self._workspace.data_added.connect(self._on_data_changed)
        self._workspace.data_removed.connect(self._on_data_changed)
        self._workspace.data_updated.connect(self._on_data_changed)
    
    def _on_data_changed(self, *args, **kwargs):
        """工作空间数据变化"""
        if not self._is_modified:
            self._is_modified = True
            self.project_modified.emit(True)
    
    @property
    def current_file_path(self) -> Optional[str]:
        return self._current_file_path
    
    @property
    def is_modified(self) -> bool:
        return self._is_modified
    
    @property
    def project_name(self) -> str:
        if self._current_file_path:
            return os.path.splitext(os.path.basename(self._current_file_path))[0]
        return "未命名项目"
    
    def save_project(self, file_path: str = None) -> bool:
        """
        保存工程
        
        Args:
            file_path: 文件路径，为 None 时使用当前路径
            
        Returns:
            bool: 是否保存成功
        """
        if file_path is None:
            file_path = self._current_file_path
        
        if not file_path:
            return False
        
        try:
            # 准备数据
            project_data = self._prepare_project_data()
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, ensure_ascii=False, indent=2)
            
            # 更新状态
            self._current_file_path = file_path
            self._is_modified = False
            
            # 发送信号
            self.project_saved.emit(file_path)
            self.project_modified.emit(False)
            
            return True
            
        except Exception as e:
            print(f"保存工程失败: {e}")
            return False
    
    def load_project(self, file_path: str) -> bool:
        """
        加载工程
        
        Args:
            file_path: 工程文件路径
            
        Returns:
            bool: 是否加载成功
        """
        if not os.path.exists(file_path):
            return False
        
        try:
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
            
            # 解析数据
            self._parse_project_data(project_data)
            
            # 更新状态
            self._current_file_path = file_path
            self._is_modified = False
            
            # 发送信号
            self.project_loaded.emit(file_path)
            self.project_modified.emit(False)
            
            return True
            
        except Exception as e:
            print(f"加载工程失败: {e}")
            return False
    
    def _prepare_project_data(self) -> Dict[str, Any]:
        """准备工程数据"""
        return {
            "version": "1.0.0",
            "project_name": self._workspace.project_name or "未命名项目",
            "images": self._serialize_images(),
            "processed_images": self._serialize_processed_images(),
            "point_clouds": self._serialize_pointclouds(),
            "vectors": self._serialize_vectors(),
            "masks": self._serialize_masks(),
            "dom": self._workspace.get_dom(),
            "dem": self._workspace.get_dem(),
            "models": self._workspace.get("models", {}),
            "metadata": {
                "created": self._get_timestamp(),
                "modified": self._get_timestamp(),
                "app": "数字摄影测量实习平台"
            }
        }
    
    def _serialize_images(self) -> Dict[str, Any]:
        """序列化影像数据"""
        images = {}
        for name, data in self._workspace.get_all_images().items():
            images[name] = {
                "path": data,
                "type": "image"
            }
        return images

    def _serialize_processed_images(self) -> Dict[str, Any]:
        """序列化处理结果"""
        processed = {}
        for name, data in self._workspace.get_all_processed_images().items():
            processed[name] = {
                "path": data,
                "type": "processed_image"
            }
        return processed
    
    def _serialize_pointclouds(self) -> Dict[str, Any]:
        """序列化点云数据"""
        pointclouds = {}
        for name, data in self._workspace.get_all_pointclouds().items():
            pointclouds[name] = {
                "path": data,
                "type": "pointcloud"
            }
        return pointclouds
    
    def _serialize_vectors(self) -> Dict[str, Any]:
        """序列化矢量数据"""
        vectors = {}
        for name, data in self._workspace.get_all_vectors().items():
            vectors[name] = {
                "path": data,
                "type": "vector"
            }
        return vectors
    
    def _serialize_masks(self) -> Dict[str, Any]:
        """序列化掩膜数据"""
        masks = {}
        for name, data in self._workspace.get_all_masks().items():
            masks[name] = {
                "path": data,
                "type": "mask"
            }
        return masks
    
    def _parse_project_data(self, data: Dict[str, Any]):
        """解析工程数据"""
        # 清空当前数据
        self._workspace.clear_all()
        
        # 设置项目名
        self._workspace.project_name = data.get("project_name", "未命名项目")
        
        # 加载影像
        images = data.get("images", {})
        for name, info in images.items():
            path = info.get("path")
            if path and os.path.exists(path):
                self._workspace.add_image(name, path)

        # 加载处理结果
        processed_images = data.get("processed_images", {})
        for name, info in processed_images.items():
            path = info.get("path")
            if path and os.path.exists(path):
                import cv2
                import numpy as np
                img_array = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if img_array is not None:
                    self._workspace.add_processed_image(name, img_array)
        
        # 加载点云
        pointclouds = data.get("point_clouds", {})
        for name, info in pointclouds.items():
            path = info.get("path")
            if path and os.path.exists(path):
                self._workspace.add_pointcloud(name, path)
        
        # 加载矢量
        vectors = data.get("vectors", {})
        for name, info in vectors.items():
            path = info.get("path")
            if path and os.path.exists(path):
                self._workspace.add_vector(name, path)
        
        # 加载掩膜
        masks = data.get("masks", {})
        for name, info in masks.items():
            path = info.get("path")
            if path and os.path.exists(path):
                self._workspace.add_mask(name, path)

        dom_ref = data.get("dom")
        if isinstance(dom_ref, dict):
            self._workspace.set_dom(dom_ref.get("name"), dom_ref.get("path"))

        dem_ref = data.get("dem")
        if isinstance(dem_ref, dict):
            self._workspace.set_dem(dem_ref.get("name"), dem_ref.get("path"))
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def new_project(self):
        """新建项目"""
        self._workspace.clear_all()
        self._current_file_path = None
        self._is_modified = False
        self.project_modified.emit(False)


# 全局单例
_project_manager_instance = None


def get_project_manager() -> ProjectManager:
    """获取全局项目管理器实例"""
    global _project_manager_instance
    if _project_manager_instance is None:
        _project_manager_instance = ProjectManager()
    return _project_manager_instance
