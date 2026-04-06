"""
数字摄影测量实习平台 - 启动入口
"""
import sys
import os

# 确保项目根目录在 Python 路径中
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from ui.main_window import MainWindow
from core.workspace import get_workspace
from core.log_manager import log_manager


def main():
    """主函数"""
    log_manager.info("=" * 50)
    log_manager.info("数字摄影测量实习平台启动")
    log_manager.info("=" * 50)
    
    # 创建 Qt 应用
    app = QApplication(sys.argv)
    app.setApplicationName("数字摄影测量实习平台")
    app.setApplicationVersion("1.0.0")
    
    # 默认加载浅色主题（可在菜单中切换）
    theme_file = "ui/styles/light1_theme.qss"
    try:
        with open(theme_file, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except Exception as e:
        log_manager.warning(f"无法加载主题文件: {e}")
    
    # 创建工作空间
    workspace = get_workspace()
    
    # 创建主窗口
    window = MainWindow(workspace)
    window.show()
    
    log_manager.info("主窗口已显示")
    
    # 运行应用
    exit_code = app.exec()
    
    log_manager.info(f"应用退出，代码: {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    # 修复 Windows 上 Ctrl+C 无法退出的问题
    import signal
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal.SIG_IGN)
    
    main()
