"""
插件基类接口
所有模块插件必须继承此类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING
from PySide6.QtWidgets import QWidget

if TYPE_CHECKING:
    from core.workspace import Workspace


class IPlugin(ABC):
    """插件接口基类"""
    
    def __init__(self, workspace: "Workspace"):
        """
        初始化插件
        
        Args:
            workspace: 全局工作空间实例，用于数据共享
        """
        self.workspace = workspace
    
    @abstractmethod
    def plugin_info(self) -> Dict[str, Any]:
        """
        返回插件元数据
        
        Returns:
            Dict: {
                'name': '模块显示名称',
                'group': '菜单分组名称', 
                'icon': '图标路径 (可选)',
                'version': '版本号',
                'description': '模块描述'
            }
        """
        pass
    
    @abstractmethod
    def get_ui_panel(self) -> QWidget:
        """
        返回该模块在右侧属性栏显示的 UI 界面
        
        Returns:
            QWidget: 参数设置面板
        """
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        该模块的核心算法执行入口
        """
        pass
    
    def on_activate(self):
        """插件激活时调用"""
        pass
    
    def on_deactivate(self):
        """插件停用时调用"""
        pass