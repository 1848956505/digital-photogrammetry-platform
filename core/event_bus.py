"""
全局事件总线 (Event Bus)
实现发布-订阅模式，解耦模块间通信
"""
from typing import Callable, Dict, List, Any
from PySide6.QtCore import QObject, Signal


# 预定义事件主题
class EventTopics:
    """事件主题常量"""
    # 影像相关
    TOPIC_IMAGE_ADDED = "image_added"           # 新影像导入
    TOPIC_IMAGE_REMOVED = "image_removed"       # 影像移除
    TOPIC_IMAGE_SELECTED = "image_selected"     # 影像被选中/双击
    TOPIC_IMAGE_UPDATED = "image_updated"       # 影像数据更新
    
    # 点云相关
    TOPIC_POINTCLOUD_ADDED = "pointcloud_added"
    TOPIC_POINTCLOUD_REMOVED = "pointcloud_removed"
    TOPIC_POINTCLOUD_SELECTED = "pointcloud_selected"
    
    # 矢量相关
    TOPIC_VECTOR_ADDED = "vector_added"
    TOPIC_VECTOR_UPDATED = "vector_updated"
    TOPIC_VECTOR_REMOVED = "vector_removed"
    TOPIC_VECTOR_SELECTED = "vector_selected"
    TOPIC_VECTOR_EDIT_MODE_CHANGED = "vector_edit_mode_changed"
    
    # 任务相关
    TOPIC_TASK_STARTED = "task_started"
    TOPIC_TASK_PROGRESS = "task_progress"
    TOPIC_TASK_FINISHED = "task_finished"
    TOPIC_TASK_ERROR = "task_error"
    
    # 视图相关
    TOPIC_VIEW_2D_CHANGED = "view_2d_changed"
    TOPIC_VIEW_3D_CHANGED = "view_3d_changed"
    TOPIC_VIEW_MODE_CHANGED = "view_mode_changed"
    TOPIC_VIEW_SINGLE_REQUEST = "view_single_request"
    TOPIC_VIEW_COMPARE_REQUEST = "view_compare_request"
    TOPIC_VIEW_3D_REQUEST = "view_3d_request"
    TOPIC_VIEW_CLEAR_REQUEST = "view_clear_request"
    TOPIC_VIEW_MOUSE_PRESSED = "view_mouse_pressed"
    TOPIC_VIEW_MOUSE_RELEASED = "view_mouse_released"
    TOPIC_VIEW_MOUSE_DOUBLE_CLICKED = "view_mouse_double_clicked"
    TOPIC_VIEW_KEY_PRESSED = "view_key_pressed"
    
    # 项目相关
    TOPIC_PROJECT_OPENED = "project_opened"
    TOPIC_PROJECT_SAVED = "project_saved"
    TOPIC_PROJECT_NEW = "project_new"
    
    # 坐标相关
    TOPIC_COORDINATE_CHANGED = "coordinate_changed"  # (col, row, r, g, b)


class EventBus(QObject):
    """
    全局事件总线 - 发布-订阅模式
    
    用于解耦组件间通信。例如：
    - 模块一处理完图片，左侧树和右侧视图需要自动刷新
    - 它们之间不应该互相调用，而是通过 EventBus 通信
    """
    
    # 信号（可选，用于Qt信号槽兼容）
    event_triggered = Signal(str, object)  # (topic, data)
    
    def __init__(self):
        super().__init__()
        self._subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, topic: str, callback: Callable):
        """
        订阅事件
        
        Args:
            topic: 事件主题
            callback: 回调函数，签名为 callback(data)
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        
        if callback not in self._subscribers[topic]:
            self._subscribers[topic].append(callback)
    
    def unsubscribe(self, topic: str, callback: Callable):
        """
        取消订阅
        
        Args:
            topic: 事件主题
            callback: 回调函数
        """
        if topic in self._subscribers:
            if callback in self._subscribers[topic]:
                self._subscribers[topic].remove(callback)
    
    def publish(self, topic: str, data: Any = None):
        """
        发布事件
        
        Args:
            topic: 事件主题
            data: 传递给订阅者的数据
        """
        # 发送Qt信号
        self.event_triggered.emit(topic, data)
        
        # 调用Python回调
        if topic in self._subscribers:
            for callback in self._subscribers[topic]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"EventBus callback error for topic '{topic}': {e}")
    
    def clear(self, topic: str = None):
        """
        清空订阅
        
        Args:
            topic: 指定主题，为None则清空所有
        """
        if topic is None:
            self._subscribers.clear()
        elif topic in self._subscribers:
            self._subscribers[topic].clear()
    
    def get_subscribers_count(self, topic: str) -> int:
        """获取指定主题的订阅者数量"""
        return len(self._subscribers.get(topic, []))


# ==================== 全局单例 ====================

_event_bus_instance = None


def get_event_bus() -> EventBus:
    """获取全局事件总线实例"""
    global _event_bus_instance
    if _event_bus_instance is None:
        _event_bus_instance = EventBus()
    return _event_bus_instance
