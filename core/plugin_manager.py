"""
插件管理器 - 动态加载引擎
"""
import os
import sys
import importlib
import inspect
from typing import List, Optional, Dict, Any
from core.base_interface import IPlugin
from core.log_manager import log_manager


class PluginManager:
    """插件动态加载引擎"""
    
    def __init__(self, workspace: "Workspace"):
        """
        初始化插件管理器
        
        Args:
            workspace: 全局工作空间
        """
        self.workspace = workspace
        self.plugins: List[IPlugin] = []
        self._plugin_registry: Dict[str, IPlugin] = {}
    
    def discover_plugins(self, plugin_dir: str = "plugins") -> List[IPlugin]:
        """动态扫描 plugins 目录并加载符合接口规范的类"""
        self.plugins = []
        self._plugin_registry = {}
        
        # 获取插件目录的绝对路径
        if not os.path.isabs(plugin_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            plugin_dir = os.path.join(project_root, plugin_dir)
        
        log_manager.info(f"正在扫描插件目录: {plugin_dir}")
        
        if not os.path.exists(plugin_dir):
            log_manager.warning(f"插件目录不存在: {plugin_dir}")
            return self.plugins
        
        # 遍历 plugins 目录下的所有文件夹
        for item in os.listdir(plugin_dir):
            item_path = os.path.join(plugin_dir, item)
            if os.path.isdir(item_path) and not item.startswith("__"):
                self._load_plugin(item, item_path)
        
        log_manager.info(f"共加载 {len(self.plugins)} 个插件")
        return self.plugins
    
    def _load_plugin(self, name: str, path: str):
        """加载单个插件"""
        try:
            # 添加项目根目录到 sys.path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # 动态导入模块
            module = importlib.import_module(f"plugins.{name}")
            
            # 先检查模块本身，再检查 plugin 子模块
            found_class = self._find_plugin_class(module)
            
            if not found_class:
                try:
                    sub_module = importlib.import_module(f"plugins.{name}.plugin")
                    found_class = self._find_plugin_class(sub_module)
                except ImportError:
                    pass
            
            if found_class:
                plugin_instance = found_class(self.workspace)
                self.plugins.append(plugin_instance)
                self._plugin_registry[name] = plugin_instance
                info = plugin_instance.plugin_info()
                log_manager.info(f"成功加载插件: {info.get('name', name)}")
            else:
                log_manager.warning(f"插件 {name} 中未找到 IPlugin 类")
                    
        except Exception as e:
            log_manager.error(f"无法加载插件 {name}: {str(e)}")
    
    def _find_plugin_class(self, module):
        """在模块中查找 IPlugin 子类"""
        for obj_name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, IPlugin) and obj is not IPlugin and hasattr(obj, 'plugin_info'):
                return obj
        return None
    
    def get_plugin(self, name: str) -> Optional[IPlugin]:
        """根据名称获取插件"""
        for p in self.plugins:
            if p.plugin_info()['name'] == name:
                return p
        return self._plugin_registry.get(name)
    
    def get_plugins_by_group(self, group: str) -> List[IPlugin]:
        """获取指定分组的插件"""
        return [p for p in self.plugins if p.plugin_info()['group'] == group]
    
    def get_all_groups(self) -> List[str]:
        """获取所有插件分组"""
        groups = set()
        for p in self.plugins:
            groups.add(p.plugin_info()['group'])
        return sorted(list(groups))


# 避免循环导入
from core.workspace import Workspace