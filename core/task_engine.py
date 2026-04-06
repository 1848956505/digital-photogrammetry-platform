"""
多线程任务引擎 (Task Engine)
基于 QThreadPool 实现，防止耗时操作阻塞 UI
"""
import traceback
from typing import Callable, Any, Optional
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool


class TaskSignals(QObject):
    """任务信号定义"""
    started = Signal()                      # 任务开始
    progress = Signal(int)                  # 进度更新 (0-100)
    finished = Signal(object)               # 任务完成，返回结果
    error = Signal(str)                     # 任务错误


class TaskWorker(QRunnable):
    """
    可运行的任务包装器
    
    支持：
    - 任意函数执行
    - 进度实时更新
    - 错误捕获
    - 结果返回
    """
    
    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs
        self.signals = TaskSignals()
        
        # 允许任务在主线程被终止（可选）
        self._cancelled = False
    
    def run(self):
        """任务执行入口"""
        try:
            # 发送开始信号
            self.signals.started.emit()
            
            # 检查是否已取消
            if self._cancelled:
                self.signals.finished.emit(None)
                return
            
            # 执行函数
            # 如果函数支持进度回调，则注入 progress 回调
            result = self._execute_with_progress()
            
            # 发送完成信号
            self.signals.finished.emit(result)
            
        except Exception as e:
            # 发送错误信号
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.signals.error.emit(error_msg)
            print(f"Task error: {error_msg}")
            traceback.print_exc()
    
    def _execute_with_progress(self) -> Any:
        """执行函数，支持进度回调"""
        # 检查函数是否接受 progress 回调参数
        import inspect
        sig = inspect.signature(self._func)
        
        if 'progress_callback' in sig.parameters:
            # 注入进度回调
            self._kwargs['progress_callback'] = self._signals_progress_wrapper
            return self._func(*self._args, **self._kwargs)
        else:
            # 直接执行
            return self._func(*self._args, **self._kwargs)
    
    def _signals_progress_wrapper(self, value: int):
        """进度回调包装器"""
        # 确保进度值在 0-100 范围内
        value = max(0, min(100, value))
        self.signals.progress.emit(value)
    
    def cancel(self):
        """取消任务"""
        self._cancelled = True


class TaskEngine(QObject):
    """
    多线程任务引擎管理器
    
    功能：
    - 任务队列管理
    - 进度跟踪
    - 任务状态查询
    """
    
    # 信号
    task_started = Signal(str)               # (task_id)
    task_progress = Signal(str, int)         # (task_id, progress)
    task_finished = Signal(str, object)       # (task_id, result)
    task_error = Signal(str, str)            # (task_id, error)
    all_tasks_finished = Signal()            # 所有任务完成
    
    def __init__(self):
        super().__init__()
        
        # 线程池
        self._thread_pool = QThreadPool.globalInstance()
        self._thread_pool.setMaxThreadCount(4)  # 最多4个并发任务
        
        # 任务追踪
        self._running_tasks: dict = {}  # {id: TaskWorker}
        self._task_counter = 0
        
        # 当前活跃任务
        self._active_task_id: Optional[str] = None
    
    def run_task(self, func: Callable, *args, task_name: str = None, **kwargs) -> str:
        """
        运行任务
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            task_name: 任务名称（可选）
            **kwargs: 关键字参数
            
        Returns:
            str: 任务ID
        """
        # 生成任务ID
        self._task_counter += 1
        task_id = task_name or f"task_{self._task_counter}"
        
        # 创建任务
        worker = TaskWorker(func, *args, **kwargs)
        
        # 连接信号
        worker.signals.started.connect(lambda: self._on_task_started(task_id))
        worker.signals.progress.connect(lambda p: self._on_task_progress(task_id, p))
        worker.signals.finished.connect(lambda r: self._on_task_finished(task_id, r))
        worker.signals.error.connect(lambda e: self._on_task_error(task_id, e))
        
        # 存储任务引用
        self._running_tasks[task_id] = worker
        self._active_task_id = task_id
        
        # 提交到线程池
        self._thread_pool.start(worker)
        
        return task_id
    
    def run_background_task(self, func: Callable, *args, **kwargs) -> str:
        """
        运行后台任务（无任务名）
        """
        return self.run_task(func, *args, **kwargs)
    
    def cancel_task(self, task_id: str):
        """取消任务"""
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
    
    def cancel_all(self):
        """取消所有任务"""
        for worker in self._running_tasks.values():
            worker.cancel()
    
    def get_active_task_id(self) -> Optional[str]:
        """获取当前活跃任务ID"""
        return self._active_task_id
    
    def is_busy(self) -> bool:
        """检查是否有任务在运行"""
        return len(self._running_tasks) > 0
    
    def clear_finished_tasks(self):
        """清理已完成的任务记录"""
        # 移除已完成的任务（QRunnable 会自动从池中释放）
        finished = [tid for tid, worker in self._running_tasks.items()]
        for tid in finished:
            del self._running_tasks[tid]
        
        if not self._running_tasks:
            self._active_task_id = None
    
    # ==================== 内部回调 ====================
    
    def _on_task_started(self, task_id: str):
        """任务开始回调"""
        self.task_started.emit(task_id)
    
    def _on_task_progress(self, task_id: str, progress: int):
        """任务进度回调"""
        self.task_progress.emit(task_id, progress)
    
    def _on_task_finished(self, task_id: str, result: Any):
        """任务完成回调"""
        self.task_finished.emit(task_id, result)
        
        # 清理
        if task_id in self._running_tasks:
            del self._running_tasks[task_id]
        
        if self._active_task_id == task_id:
            self._active_task_id = None
        
        # 检查是否所有任务完成
        if not self._running_tasks:
            self.all_tasks_finished.emit()
    
    def _on_task_error(self, task_id: str, error: str):
        """任务错误回调"""
        self.task_error.emit(task_id, error)
        
        # 清理
        if task_id in self._running_tasks:
            del self._running_tasks[task_id]
        
        if self._active_task_id == task_id:
            self._active_task_id = None


# ==================== 便捷函数 ====================

_task_engine_instance = None


def get_task_engine() -> TaskEngine:
    """获取全局任务引擎实例"""
    global _task_engine_instance
    if _task_engine_instance is None:
        _task_engine_instance = TaskEngine()
    return _task_engine_instance


# ==================== 模拟任务函数 ====================

def simulate_long_task(progress_callback: Callable = None, duration: int = 5) -> str:
    """
    模拟长时间任务
    
    Args:
        progress_callback: 进度回调函数
        duration: 持续时间（秒）
        
    Returns:
        str: 完成消息
    """
    import time
    
    total_steps = duration * 10  # 每秒10次更新
    for i in range(total_steps):
        time.sleep(0.1)  # 100ms
        
        # 更新进度
        progress = int((i + 1) / total_steps * 100)
        if progress_callback:
            progress_callback(progress)
    
    return f"任务完成，耗时 {duration} 秒"


def simple_simulate_task() -> str:
    """简单模拟任务（无进度回调）"""
    import time
    time.sleep(2)
    return "简单任务完成"