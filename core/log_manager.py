"""
日志管理器
"""
import logging
from datetime import datetime
from typing import Optional


class LogManager:
    """日志管理器"""
    
    _instance: Optional["LogManager"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._logger = logging.getLogger("PhotogrammetryPlatform")
        self._logger.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        self._logger.addHandler(console_handler)
        
        # 消息回调
        self._callbacks = []
    
    def add_callback(self, callback):
        """添加日志回调"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback):
        """移除日志回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _emit(self, level: str, message: str):
        """发送日志到回调"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for cb in self._callbacks:
            try:
                cb(timestamp, level, message)
            except Exception:
                pass
    
    def debug(self, message: str):
        self._logger.debug(message)
        self._emit("DEBUG", message)
    
    def info(self, message: str):
        self._logger.info(message)
        self._emit("INFO", message)
    
    def warning(self, message: str):
        self._logger.warning(message)
        self._emit("WARNING", message)
    
    def error(self, message: str):
        self._logger.error(message)
        self._emit("ERROR", message)
    
    def critical(self, message: str):
        self._logger.critical(message)
        self._emit("CRITICAL", message)
    
    @property
    def logger(self):
        return self._logger


# 全局日志实例
log_manager = LogManager()