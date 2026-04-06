"""
全局工作空间 (Workspace)
管理全生命周期的数据，作为内存中的"中央数据库"
"""
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Callable
from PySide6.QtCore import QObject, Signal


class Workspace(QObject):
    """
    工作空间 - 全局数据管理中心
    
    所有插件不直接读写本地文件，而是向 Workspace 要数据。
    支持数据的增删改查（CRUD）接口。
    """
    
    # 信号
    data_added = Signal(str, str, object)      # (type, id, data)
    data_removed = Signal(str, str)           # (type, id)
    data_updated = Signal(str, str, object)  # (type, id, data)
    project_loaded = Signal(str)                # (project_path)
    project_saved = Signal(str)                 # (project_path)
    
    def __init__(self):
        super().__init__()
        # 全局数据结构
        self._data = {
            "images": {},           # {id: {"path": str, "array": np.array}}
            "processed_images": {}, # 处理后的图像 {id: {"path": str, "array": np.array}}
            "point_clouds": {},      # {id: {"path": str}}
            "vectors": {},           # {id: {"path": str}}
            "masks": {},             # {id: {"path": str}}
            "dom": None,             # 项目级 DOM 引用位
            "dem": None,             # 项目级 DEM 引用位
            "models": {},             # {id: {"path": str}}
        }
        
        # 项目信息
        self._project_path: Optional[str] = None
        self._project_name: str = "未命名项目"
        
        # 缓存管理
        self._cache_enabled = True
        self._max_cache_size = 100  # MB
        
        # 回调函数
        self._callbacks: Dict[str, List[Callable]] = {}
    
    # ==================== 通用 get 方法（兼容插件）====================
    
    def get(self, key: str, default=None):
        """
        通用获取方法，兼容旧插件代码
        
        Args:
            key: 数据类型键，如 'images', 'point_clouds' 等
            default: 默认值
            
        Returns:
            数据字典或默认值
        """
        if key in self._data:
            return self._data[key]
        return default
    
    # ==================== 数据 CRUD ====================
    
    def add_image(self, name: str, path: str) -> bool:
        """
        添加影像
        
        Args:
            name: 影像名称（作为ID）
            path: 影像文件路径
            
        Returns:
            bool: 是否添加成功
        """
        if not os.path.exists(path):
            return False
        
        # 检查是否已存在
        if name in self._data["images"]:
            # 更新
            self._data["images"][name]["path"] = path
            self.data_updated.emit("images", name, path)
        else:
            # 新增
            self._data["images"][name] = {
                "path": path,
                "array": None  # 延迟加载
            }
            self.data_added.emit("images", name, path)
        
        return True
    
    # ==================== 处理结果管理 ====================
    
    def add_processed_image(self, name: str, image_array) -> bool:
        """
        添加处理后的图像
        
        Args:
            name: 图像名称
            image_array: numpy 数组形式的图像数据
            
        Returns:
            bool: 是否添加成功
        """
        if image_array is None:
            return False
        
        # 临时保存到文件
        import tempfile
        import cv2
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"processed_{name}.png")
        
        # 保存图像
        result, encoded = cv2.imencode('.png', image_array)
        if result:
            encoded.tofile(temp_path)
        
        # 存储到处理结果字典
        self._data["processed_images"][name] = {
            "path": temp_path,
            "array": image_array
        }
        self.data_added.emit("processed_images", name, temp_path)
        
        return True

    def set_dom(self, name: Optional[str], path: Optional[str] = None) -> None:
        """设置项目级 DOM 引用位"""
        self._data["dom"] = {"name": name, "path": path} if name or path else None
        self.data_updated.emit("dom", name or "", self._data["dom"])

    def get_dom(self) -> Optional[Dict[str, Any]]:
        """获取项目级 DOM 引用位"""
        return self._data.get("dom")

    def set_dem(self, name: Optional[str], path: Optional[str] = None) -> None:
        """设置项目级 DEM 引用位"""
        self._data["dem"] = {"name": name, "path": path} if name or path else None
        self.data_updated.emit("dem", name or "", self._data["dem"])

    def get_dem(self) -> Optional[Dict[str, Any]]:
        """获取项目级 DEM 引用位"""
        return self._data.get("dem")
    
    def get_processed_image(self, name: str) -> Optional[str]:
        """获取处理后图像的路径"""
        if name in self._data["processed_images"]:
            return self._data["processed_images"][name]["path"]
        return None
    
    def get_all_processed_images(self) -> Dict[str, str]:
        """获取所有处理后图像"""
        return {name: data["path"] for name, data in self._data["processed_images"].items()}
    
    def clear_processed_images(self):
        """清空处理结果"""
        self._data["processed_images"].clear()
    
    # ==================== 原始图像 CRUD ====================
    
    def get_image(self, name: str) -> Optional[str]:
        """获取影像路径"""
        if name in self._data["images"]:
            return self._data["images"][name]["path"]
        return None
    
    def get_all_images(self) -> Dict[str, str]:
        """获取所有影像"""
        return {name: data["path"] for name, data in self._data["images"].items()}
    
    def remove_image(self, name: str) -> bool:
        """移除影像"""
        if name in self._data["images"]:
            del self._data["images"][name]
            self.data_removed.emit("images", name)
            return True
        return False
    
    def add_pointcloud(self, name: str, path: str) -> bool:
        """添加点云"""
        if not os.path.exists(path):
            return False
        
        self._data["point_clouds"][name] = {"path": path}
        self.data_added.emit("point_clouds", name, path)
        return True
    
    def get_all_pointclouds(self) -> Dict[str, str]:
        """获取所有点云"""
        return {name: data["path"] for name, data in self._data["point_clouds"].items()}
    
    def add_vector(self, name: str, vector_data: Dict[str, Any]) -> bool:
        """添加矢量成果对象。"""
        if not name:
            return False
        if not isinstance(vector_data, dict):
            return False

        data = deepcopy(vector_data)
        data["name"] = name
        self._data["vectors"][name] = data
        self.data_added.emit("vectors", name, data)
        return True

    def update_vector(self, name: str, vector_data: Dict[str, Any]) -> bool:
        """更新矢量成果对象。"""
        if not name:
            return False
        if not isinstance(vector_data, dict):
            return False

        data = deepcopy(vector_data)
        data["name"] = name
        self._data["vectors"][name] = data
        self.data_updated.emit("vectors", name, data)
        return True

    def remove_vector(self, name: str) -> bool:
        """移除矢量成果对象。"""
        if name in self._data["vectors"]:
            del self._data["vectors"][name]
            self.data_removed.emit("vectors", name)
            return True
        return False

    def get_vector(self, name: str) -> Optional[Dict[str, Any]]:
        """获取单个矢量成果对象。"""
        return self._data["vectors"].get(name)

    def list_vectors(self) -> Dict[str, Dict[str, Any]]:
        """获取所有矢量成果对象。"""
        return self._data["vectors"]

    def get_all_vectors(self) -> Dict[str, Dict[str, Any]]:
        """获取所有矢量成果对象（兼容旧接口）。"""
        return self.list_vectors()
    
    def add_mask(self, name: str, path: str) -> bool:
        """添加掩膜"""
        if not os.path.exists(path):
            return False
        
        self._data["masks"][name] = {"path": path}
        self.data_added.emit("masks", name, path)
        return True
    
    def get_all_masks(self) -> Dict[str, str]:
        """获取所有掩膜"""
        return {name: data["path"] for name, data in self._data["masks"].items()}
    
    # ==================== 项目管理 ====================
    
    @property
    def project_path(self) -> Optional[str]:
        return self._project_path
    
    @project_path.setter
    def project_path(self, path: str):
        self._project_path = path
        if path:
            self.project_loaded.emit(path)
    
    @property
    def project_name(self) -> str:
        return self._project_name
    
    @project_name.setter
    def project_name(self, name: str):
        self._project_name = name
    
    def get_data_dict(self) -> Dict[str, Any]:
        """获取用于序列化的数据结构"""
        return {
            "project_name": self._project_name,
            "images": self._data["images"],
            "processed_images": self._data["processed_images"],
            "point_clouds": self._data["point_clouds"],
            "vectors": self._data["vectors"],
            "masks": self._data["masks"],
            "dom": self._data["dom"],
            "dem": self._data["dem"],
            "models": self._data["models"]
        }
    
    def load_from_dict(self, data: Dict[str, Any]):
        """从序列化数据加载"""
        self._project_name = data.get("project_name", "未命名项目")
        
        # 清理旧数据
        self._data = {
            "images": {},
            "processed_images": {},
            "point_clouds": {},
            "vectors": {},
            "masks": {},
            "dom": None,
            "dem": None,
            "models": {}
        }
        
        # 加载影像
        for name, item in data.get("images", {}).items():
            path = item.get("path")
            if path and os.path.exists(path):
                self._data["images"][name] = {"path": path, "array": None}

        # 加载处理结果
        for name, item in data.get("processed_images", {}).items():
            path = item.get("path")
            if path and os.path.exists(path):
                self._data["processed_images"][name] = {"path": path, "array": None}
        
        # 加载点云
        for name, item in data.get("point_clouds", {}).items():
            path = item.get("path")
            if path and os.path.exists(path):
                self._data["point_clouds"][name] = {"path": path}
        
        # 加载矢量
        for name, item in data.get("vectors", {}).items():
            if isinstance(item, dict):
                vector_data = deepcopy(item)
                vector_data["name"] = name
                self._data["vectors"][name] = vector_data
            elif isinstance(item, str) and os.path.exists(item):
                self._data["vectors"][name] = {"name": name, "path": item}
        
        # 加载掩膜
        for name, item in data.get("masks", {}).items():
            path = item.get("path")
            if path and os.path.exists(path):
                self._data["masks"][name] = {"path": path}

        for name, item in data.get("models", {}).items():
            path = item.get("path")
            if path and os.path.exists(path):
                self._data["models"][name] = {"path": path}

        self._data["dom"] = data.get("dom")
        self._data["dem"] = data.get("dem")
    
    # ==================== 事件回调 ====================
    
    def subscribe(self, event_name: str, callback: Callable):
        """订阅事件"""
        if event_name not in self._callbacks:
            self._callbacks[event_name] = []
        if callback not in self._callbacks[event_name]:
            self._callbacks[event_name].append(callback)
    
    def unsubscribe(self, event_name: str, callback: Callable):
        """取消订阅"""
        if event_name in self._callbacks:
            if callback in self._callbacks[event_name]:
                self._callbacks[event_name].remove(callback)
    
    def emit_event(self, event_name: str, data: Any = None):
        """发送事件"""
        if event_name in self._callbacks:
            for callback in self._callbacks[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Event callback error: {e}")
    
    # ==================== 缓存管理 ====================
    
    def clear_cache(self):
        """清空缓存"""
        for data_type in self._data:
            bucket = self._data[data_type]
            if isinstance(bucket, dict):
                for item in bucket.values():
                    if isinstance(item, dict) and "array" in item:
                        item["array"] = None
    
    @property
    def image_count(self) -> int:
        return len(self._data["images"])
    
    @property
    def pointcloud_count(self) -> int:
        return len(self._data["point_clouds"])
    
    def clear_all(self):
        """清空所有数据"""
        self._data = {
            "images": {},
            "processed_images": {},
            "point_clouds": {},
            "vectors": {},
            "masks": {},
            "dom": None,
            "dem": None,
            "models": {}
        }
        self._project_path = None
        self._project_name = "未命名项目"


# 全局单例
_workspace_instance = None

def get_workspace() -> Workspace:
    """获取全局工作空间实例"""
    global _workspace_instance
    if _workspace_instance is None:
        _workspace_instance = Workspace()
    return _workspace_instance
