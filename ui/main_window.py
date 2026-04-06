"""
主窗口 - 数字摄影测量实习平台
第一阶段核心功能：Workspace + 2D渲染引擎
"""
import os
from typing import Any
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QTabWidget, QDockWidget, QTreeWidget, QTreeWidgetItem,
    QTextEdit, QStackedWidget, QLabel, QMenuBar, QMenu,
    QToolBar, QStatusBar, QFileDialog, QProgressBar, QComboBox,
    QMessageBox, QInputDialog, QLineEdit
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent

from core.workspace import Workspace, get_workspace
from core.plugin_manager import PluginManager
from core.base_interface import IPlugin
from core.log_manager import log_manager
from core.event_bus import EventBus, EventTopics, get_event_bus
from core.task_engine import get_task_engine, simulate_long_task
from core.project_manager import get_project_manager


class MainWindow(QMainWindow):
    """主窗口类 - 第一阶段实现"""
    
    # 信号
    image_selected = Signal(str)
    
    def __init__(self, workspace: Workspace = None):
        super().__init__()
        
        # 工作空间
        self.workspace = workspace or get_workspace()
        
        # 事件总线
        self.event_bus = get_event_bus()
        
        # 任务引擎
        self.task_engine = get_task_engine()
        
        # 项目管理器
        self.project_manager = get_project_manager()
        
        # 插件管理器
        self.plugin_manager = PluginManager(self.workspace)
        self.plugins = []
        self._active_plugin_index = None
        
        # 窗口设置
        self.setWindowTitle("数字摄影测量实习平台")
        self.resize(1400, 900)
        self.setMinimumSize(1000, 700)
        self.setAcceptDrops(True)
        
        # 初始化
        self._setup_ui()
        self._subscribe_events()
        self._load_plugins()

        # 在 MainWindow.__init__ 中添加
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        
        log_manager.info("主窗口初始化完成 (Sprint 1)")
    
    def _setup_ui(self):
        """初始化UI顺序：菜单->状态栏->中央->面板"""
        self._create_menu_bar()
        self._create_toolbar()
        self._create_status_bar()
        self._create_central_area()

        # 创建面板并建立【状态同步】
        self._create_left_panel()
        self._create_right_panel()
        self._create_bottom_panel()

    
    def _create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        new_action = QAction("新建项目", self)
        new_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction("打开项目", self)
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存工程", self)
        save_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        import_action = QAction("导入影像", self)
        import_action.triggered.connect(self._on_import_image)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图(&V)")

        # 定义控制Action并存为成员变量
        self.toggle_left_action = QAction("左侧资源管理器", self, checkable=True)
        self.toggle_left_action.setChecked(True)
        self.toggle_left_action.triggered.connect(self._toggle_left_panel)

        self.toggle_right_action = QAction("右侧工具箱", self, checkable=True)
        self.toggle_right_action.setChecked(True)
        self.toggle_right_action.triggered.connect(self._toggle_right_panel)

        self.toggle_log_action = QAction("底部日志", self, checkable=True)
        self.toggle_log_action.setChecked(True)
        self.toggle_log_action.triggered.connect(self._toggle_log_panel)

        view_menu.addAction(self.toggle_left_action)
        view_menu.addAction(self.toggle_right_action)
        view_menu.addAction(self.toggle_log_action)
        
        view_menu.addSeparator()
        
        # 2D/3D视图切换
        self.view_2d_action = QAction("二维视图", self, checkable=True)
        self.view_2d_action.setChecked(True)
        self.view_2d_action.triggered.connect(lambda: self.view_2d.set_mode("2d"))
        view_menu.addAction(self.view_2d_action)
        
        self.view_3d_action = QAction("三维视图", self, checkable=True)
        self.view_3d_action.triggered.connect(lambda: self.view_2d.set_mode("3d"))
        view_menu.addAction(self.view_3d_action)

        self.view_compare_action = QAction("对比视图", self, checkable=True)
        self.view_compare_action.triggered.connect(lambda: self.view_2d.set_mode("compare"))
        view_menu.addAction(self.view_compare_action)
        
        view_menu.addSeparator()
        
        # 放大/缩小
        zoom_in_action = QAction("放大", self)
        zoom_in_action.triggered.connect(self._on_zoom_in)
        view_menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("缩小", self)
        zoom_out_action.triggered.connect(self._on_zoom_out)
        view_menu.addAction(zoom_out_action)

        zoom_fit_action = QAction("重置视图", self)
        zoom_fit_action.setShortcut("F")
        zoom_fit_action.triggered.connect(self._on_zoom_fit)
        view_menu.addAction(zoom_fit_action)
        
        # 工具菜单
        tool_menu = menubar.addMenu("工具(&T)")
        
        test_task_action = QAction("测试长耗时任务", self)
        test_task_action.triggered.connect(self._on_test_long_task)
        tool_menu.addAction(test_task_action)
        
        # 主题菜单
        theme_menu = menubar.addMenu("主题(&T)")
        
        # 扫描样式文件
        import os
        styles_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "styles")
        if os.path.exists(styles_dir):
            for f in os.listdir(styles_dir):
                if f.endswith("_theme.qss"):
                    theme_name = f.replace("_theme.qss", "")
                    theme_labels = {
                        "light": "浅色主题",
                        "light1": "浅色主题二",
                        "dark": "深色主题",
                    }
                    theme_action = QAction(theme_labels.get(theme_name, f"主题-{theme_name}"), self, checkable=True)
                    theme_action.setChecked(theme_name == "light")  # 默认选中浅色
                    theme_action.triggered.connect(lambda checked, t=theme_name: self._on_change_theme(t))
                    theme_menu.addAction(theme_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        new_action = QAction("新建", self)
        new_action.triggered.connect(self._on_new_project)
        toolbar.addAction(new_action)
        
        open_action = QAction("打开", self)
        open_action.triggered.connect(self._on_import_image)
        toolbar.addAction(open_action)

    def _create_central_area(self):
        from ui.central_display import CentralDisplayWidget

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view_2d = CentralDisplayWidget()
        self.view_2d.mouse_moved.connect(self._on_mouse_moved)
        self.view_2d.mouse_pressed.connect(self._on_view_mouse_pressed)
        self.view_2d.mouse_released.connect(self._on_view_mouse_released)
        self.view_2d.mouse_double_clicked.connect(self._on_view_mouse_double_clicked)
        self.view_2d.key_pressed.connect(self._on_view_key_pressed)
        self.view_2d.image_loaded.connect(self._on_image_loaded)
        self.view_2d.mode_changed.connect(self._on_view_mode_changed)

        layout.addWidget(self.view_2d)
        self.setCentralWidget(central_widget)
    
    def _create_left_panel(self):
        """创建左侧资源管理面板"""
        self.dock_layers = QDockWidget("资源管理器", self)
        self.dock_layers.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # 当用户点击面板右上角 'X' 关闭时，同步更新菜单对勾
        self.dock_layers.visibilityChanged.connect(self.toggle_left_action.setChecked)
        
        # 资源树
        self.layer_tree = QTreeWidget()
        self.layer_tree.setHeaderLabel("数据资源")
        self.layer_tree.setColumnCount(1)
        
        # 根节点
        self.root_images = QTreeWidgetItem(["📂 影像"])
        self.root_processed = QTreeWidgetItem(["📂 处理结果"])
        self.root_pointclouds = QTreeWidgetItem(["📂 点云"])
        self.root_vectors = QTreeWidgetItem(["📂 矢量"])
        self.root_masks = QTreeWidgetItem(["📂 掩膜"])
        self.root_dom = QTreeWidgetItem(["📂 DOM"])
        self.root_dem = QTreeWidgetItem(["📂 DEM"])
        self.layer_tree.addTopLevelItems([
            self.root_images,
            self.root_processed,
            self.root_pointclouds,
            self.root_vectors,
            self.root_masks,
            self.root_dom,
            self.root_dem,
        ])
        
        # 双击事件
        self.layer_tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        
        # 右键菜单设置
        self.layer_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.layer_tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        
        self.dock_layers.setWidget(self.layer_tree)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_layers)
    
    def _create_right_panel(self):
        """创建右侧工具面板"""
        self.dock_tools = QDockWidget("工具箱", self)
        self.dock_tools.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # 状态同步
        self.dock_tools.visibilityChanged.connect(self.toggle_right_action.setChecked)
        
        # 工具箱主容器
        tools_container = QWidget()
        tools_layout = QVBoxLayout(tools_container)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        
        # 插件选择器
        self.plugin_combo = QComboBox()
        self.plugin_combo.setMinimumHeight(30)
        self.plugin_combo.addItem("欢迎", 0)  # 默认欢迎页
        
        # 欢迎面板（索引0）
        welcome = QLabel("欢迎使用数字摄影测量实习平台\n\n请先导入影像文件")
        welcome.setAlignment(Qt.AlignCenter)
        welcome.setStyleSheet("color: #666; padding: 50px;")
        
        # 面板堆栈（使用 QScrollArea 包裹实现滚动）
        from PySide6.QtWidgets import QScrollArea
        self.tool_stack = QStackedWidget()
        self.tool_stack.addWidget(welcome)  # 索引0：欢迎页

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.tool_stack)
        scroll_area.setWidgetResizable(True)  # 内容自适应大小
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 水平滚动条（需要时显示）
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)    # 垂直滚动条（需要时显示）
        
        # 选择器信号连接
        self.plugin_combo.currentIndexChanged.connect(self._on_plugin_selected)
        
        tools_layout.addWidget(self.plugin_combo)
        tools_layout.addWidget(scroll_area)  # 使用 scroll_area 替代 tool_stack
        
        self.dock_tools.setWidget(tools_container)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_tools)

    def _create_bottom_panel(self):
        """创建底部日志面板"""
        self.dock_log = QDockWidget("日志", self)
        self.dock_log.setAllowedAreas(Qt.BottomDockWidgetArea)

        # 【关键】状态同步
        self.dock_log.visibilityChanged.connect(self.toggle_log_action.setChecked)

        # 设置 DockWidget 的最小高度 ---
        self.dock_log.setMinimumHeight(120)  # 确保它不会被压缩到 0 而消失

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        # 也可以设置最大高度防止日志栏占用太多空间
        # self.log_text.setMaximumHeight(400)

        self.dock_log.setWidget(self.log_text)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_log)

        log_manager.add_callback(self._on_log_message)
    
    def _create_status_bar(self):
        """创建状态栏"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        self.status_label = QLabel("就绪")
        self.statusbar.addWidget(self.status_label, 1)
        
        # 进度条（默认隐藏）
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("%p%")
        self.statusbar.addPermanentWidget(self.progress_bar)
        
        self.size_label = QLabel("")
        self.statusbar.addPermanentWidget(self.size_label)
        
        self.coord_label = QLabel("坐标: --")
        self.statusbar.addPermanentWidget(self.coord_label)
    
    def _load_plugins(self):
        """加载插件"""
        self.plugins = self.plugin_manager.discover_plugins()
        
        if self.plugins:
            for i, plugin in enumerate(self.plugins):
                try:
                    panel = plugin.get_ui_panel()
                    self.tool_stack.addWidget(panel)  # 仍添加到 tool_stack（scroll_area 里的 widget）
                    
                    plugin_name = plugin.plugin_info().get('name', f'模块{i+1}')
                    self.plugin_combo.addItem(plugin_name, i + 1)
                    
                except Exception as e:
                    log_manager.error(f"插件面板加载失败: {plugin.plugin_info().get('name', '未知')} - {e}")

    def _on_plugin_selected(self, index):
        """插件选择改变"""
        if index >= 0 and index < self.tool_stack.count():
            self.tool_stack.setCurrentIndex(index)

            # --- 修复索引逻辑 ---
            if index == 0:
                if self._active_plugin_index is not None and 0 <= self._active_plugin_index < len(self.plugins):
                    old_plugin = self.plugins[self._active_plugin_index]
                    if hasattr(old_plugin, 'on_deactivate'):
                        try:
                            old_plugin.on_deactivate()
                        except Exception as e:
                            log_manager.error(f"插件停用失败: {e}")
                self._active_plugin_index = None
                return  # 欢迎页，直接返回

            # 插件在 self.plugins 列表中的索引应该是 combo索引减 1
            plugin_idx = index - 1
            if self._active_plugin_index is not None and self._active_plugin_index != plugin_idx and 0 <= self._active_plugin_index < len(self.plugins):
                old_plugin = self.plugins[self._active_plugin_index]
                if hasattr(old_plugin, 'on_deactivate'):
                    try:
                        old_plugin.on_deactivate()
                    except Exception as e:
                        log_manager.error(f"插件停用失败: {e}")
            if 0 <= plugin_idx < len(self.plugins):
                plugin = self.plugins[plugin_idx]
                if hasattr(plugin, 'on_activate'):
                    try:
                        plugin.on_activate()
                        log_manager.info(f"插件已激活: {plugin.plugin_info()['name']}")
                        self._active_plugin_index = plugin_idx
                    except Exception as e:
                        log_manager.error(f"插件激活失败: {e}")

    # ==================== 统一的显隐控制逻辑 ====================

    def _toggle_left_panel(self):
        # 菜单项被点击时，根据其 check 状态设置面板可见性
        self.dock_layers.setVisible(self.toggle_left_action.isChecked())

    def _toggle_right_panel(self):
        self.dock_tools.setVisible(self.toggle_right_action.isChecked())

    def _toggle_log_panel(self):
        self.dock_log.setVisible(self.toggle_log_action.isChecked())
    
    # ==================== 事件处理 ====================
    
    def _on_new_project(self):
        """新建项目"""
        self.workspace.clear_all()
        self._refresh_layer_tree()
        self.view_2d.clear() if hasattr(self.view_2d, 'clear') else None
        self.status_label.setText("新项目已创建")
        log_manager.info("新项目")
    
    def _on_open_project(self):
        """打开项目"""
        path, _ = QFileDialog.getOpenFileName(self, "打开工程", "", "工程文件 (*.gai)")
        if path:
            if self.project_manager.load_project(path):
                self.status_label.setText(f"已打开: {os.path.basename(path)}")
                log_manager.info(f"打开工程: {path}")
                
                # 刷新资源树
                self._refresh_layer_tree()
                
                # 发布事件
                self.event_bus.publish(EventTopics.TOPIC_PROJECT_OPENED, path)
            else:
                log_manager.error(f"打开工程失败: {path}")
    
    def _on_save_project(self):
        """保存工程"""
        path, _ = QFileDialog.getSaveFileName(self, "保存工程", "", "工程文件 (*.gai)")
        if path:
            if self.project_manager.save_project(path):
                self.status_label.setText(f"已保存: {os.path.basename(path)}")
                log_manager.info(f"保存工程: {path}")
                
                # 发布事件
                self.event_bus.publish(EventTopics.TOPIC_PROJECT_SAVED, path)
            else:
                log_manager.error(f"保存工程失败: {path}")
    
    def _on_import_image(self):
        """导入影像"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "导入影像", "",
            "影像文件 (*.tif *.tiff *.jpg *.png *.jpeg *.bmp)"
        )
        
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            self.workspace.add_image(name, f)
            
            # 发布事件 - 通知其他模块
            self.event_bus.publish(EventTopics.TOPIC_IMAGE_ADDED, {"name": name, "path": f})
        
        if files:
            # 显示第一张图像
            first_file = files[0]
            success = self.view_2d.load_image(first_file) if hasattr(self.view_2d, 'load_image') else False
            if success:
                name = os.path.splitext(os.path.basename(first_file))[0]
                h, w = self.view_2d.image_size
                self.size_label.setText(f"{w} x {h}")
                self.status_label.setText(f"已加载: {name}")
                log_manager.info(f"图像已加载: {name}")
                
                # 发布选中事件
                self.event_bus.publish(EventTopics.TOPIC_IMAGE_SELECTED, {"name": name, "path": first_file})
    
    def _on_tree_item_double_clicked(self, item, column):
        """双击资源树项目"""
        name = item.text(0)
        parent = item.parent()
        
        # 处理影像文件夹
        if parent == self.root_images:
            images = self.workspace.get_all_images()
            if name in images:
                path = images[name]
                if os.path.exists(path):
                    self.event_bus.publish(EventTopics.TOPIC_IMAGE_SELECTED, {"name": name, "path": path})
        
        # 处理结果文件夹
        elif parent == self.root_processed:
            processed = self.workspace.get_all_processed_images()
            if name in processed:
                path = processed[name]
                if os.path.exists(path):
                    self.event_bus.publish(EventTopics.TOPIC_IMAGE_UPDATED, {"name": name, "path": path})
        elif parent == self.root_pointclouds:
            pointclouds = self.workspace.get_all_pointclouds()
            path = pointclouds.get(name)
            if path and os.path.exists(path):
                self.event_bus.publish(EventTopics.TOPIC_VIEW_3D_REQUEST, {"kind": "point_cloud", "path": path, "title": name})
        elif parent == self.root_dom:
            dom = self.workspace.get_dom()
            if dom and dom.get("path") and os.path.exists(dom["path"]):
                path = dom["path"]
                ext = os.path.splitext(path)[1].lower()
                if ext in {".npy", ".csv", ".txt", ".xyz", ".pts", ".ply"}:
                    self.event_bus.publish(EventTopics.TOPIC_VIEW_3D_REQUEST, {"kind": "point_cloud", "path": path, "title": name})
                else:
                    self.event_bus.publish(EventTopics.TOPIC_IMAGE_SELECTED, {"name": name, "path": path})
        elif parent == self.root_dem:
            dem = self.workspace.get_dem()
            if dem and dem.get("path") and os.path.exists(dem["path"]):
                path = dem["path"]
                ext = os.path.splitext(path)[1].lower()
                if ext in {".npy", ".csv", ".txt"}:
                    self.event_bus.publish(EventTopics.TOPIC_VIEW_3D_REQUEST, {"kind": "surface", "path": path, "title": name})
                else:
                    self.event_bus.publish(EventTopics.TOPIC_IMAGE_SELECTED, {"name": name, "path": path})
        elif parent == self.root_vectors:
            vector = self.workspace.get_vector(name)
            if vector:
                self.event_bus.publish(EventTopics.TOPIC_VECTOR_SELECTED, {"name": name, "vector": vector, "title": name})
    
    def _on_tree_context_menu(self, position):
        """右键菜单"""
        item = self.layer_tree.itemAt(position)
        if item is None:
            return
        
        # 检查是否是根节点（影像/点云/处理结果）
        if item.parent() is None:
            return
        
        # 获取文件名称
        name = item.text(0)
        
        # 判断是影像还是处理结果
        parent = item.parent()
        if parent == self.root_images:
            self._show_image_context_menu(position, name)
        elif parent == self.root_processed:
            self._show_processed_context_menu(position, name)
        elif parent == self.root_vectors:
            self._show_vector_context_menu(position, name)
    
    def _show_image_context_menu(self, position, name):
        """影像右键菜单"""
        from PySide6.QtWidgets import QMenu
        
        menu = QMenu(self)
        
        # 重命名
        rename_action = menu.addAction("重命名")
        # 导出
        export_action = menu.addAction("导出...")
        # 删除
        delete_action = menu.addAction("删除")
        
        # 显示菜单
        action = menu.exec(self.layer_tree.mapToGlobal(position))
        
        if action == rename_action:
            self._rename_image_file(name)
        elif action == export_action:
            self._export_image_file(name)
        elif action == delete_action:
            self._delete_image_file(name)
    
    def _show_processed_context_menu(self, position, name):
        """处理结果右键菜单"""
        from PySide6.QtWidgets import QMenu
        
        menu = QMenu(self)
        
        # 重命名
        rename_action = menu.addAction("重命名")
        # 导出
        export_action = menu.addAction("导出...")
        # 删除
        delete_action = menu.addAction("删除")
        
        # 显示菜单
        action = menu.exec(self.layer_tree.mapToGlobal(position))
        
        if action == rename_action:
            self._rename_processed_file(name)
        elif action == export_action:
            self._export_processed_file(name)
        elif action == delete_action:
            self._delete_processed_file(name)
        elif action == delete_action:
            self._delete_processed_file(name)

    def _show_vector_context_menu(self, position, name):
        """矢量右键菜单"""
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        select_action = menu.addAction("在中央视图显示")
        export_action = menu.addAction("导出...")
        delete_action = menu.addAction("删除")
        action = menu.exec(self.layer_tree.mapToGlobal(position))

        if action == select_action:
            vector = self.workspace.get_vector(name)
            if vector:
                self.event_bus.publish(EventTopics.TOPIC_VECTOR_SELECTED, {"name": name, "vector": vector, "title": name})
        elif action == export_action:
            vector = self.workspace.get_vector(name)
            if vector:
                self.event_bus.publish(EventTopics.TOPIC_VECTOR_SELECTED, {"name": name, "vector": vector, "title": name, "summary": "可在右侧面板导出"})
        elif action == delete_action:
            if self.workspace.remove_vector(name):
                log_manager.info(f"删除矢量: {name}")
                self.status_label.setText(f"已删除矢量: {name}")
    
    def _rename_image_file(self, old_name):
        """重命名影像文件"""
        from PySide6.QtWidgets import QInputDialog, QLineEdit
        
        new_name, ok = QInputDialog.getText(
            self, "重命名", "请输入新名称:",
            QLineEdit.EchoMode.Normal, old_name
        )
        
        if ok and new_name and new_name != old_name:
            # 获取旧路径
            images = self.workspace.get_all_images()
            if old_name in images:
                old_path = images[old_name]
                
                # 检查新名称是否已存在
                if new_name in images:
                    QMessageBox.warning(self, "警告", "名称已存在")
                    return
                
                # 从 Workspace 删除旧的，添加新的
                self.workspace.remove_image(old_name)
                self.workspace.add_image(new_name, old_path)
                
                # 刷新资源树
                self._refresh_layer_tree()
                
                log_manager.info(f"重命名影像: {old_name} -> {new_name}")
                self.status_label.setText(f"已重命名: {new_name}")
    
    def _export_image_file(self, name):
        """导出影像文件"""
        images = self.workspace.get_all_images()
        if name not in images:
            return
        
        source_path = images[name]
        
        # 选择保存位置
        from PySide6.QtWidgets import QFileDialog
        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出影像", name, 
            "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        
        if save_path:
            import shutil
            try:
                shutil.copy2(source_path, save_path)
                log_manager.info(f"导出影像: {save_path}")
                self.status_label.setText(f"已导出: {os.path.basename(save_path)}")
                QMessageBox.information(self, "成功", f"已导出到:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
                log_manager.error(f"导出影像失败: {e}")
    
    def _delete_image_file(self, name):
        """删除影像文件"""
        reply = QMessageBox.question(
            self, "确认删除", 
            f"确定要从工作空间中删除 '{name}' 吗？\n（不会删除原文件）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.workspace.remove_image(name)
            self._refresh_layer_tree()
            log_manager.info(f"删除影像: {name}")
            self.status_label.setText(f"已删除: {name}")
    
    def _export_processed_file(self, name):
        """导出的处理结果"""
        processed = self.workspace.get_all_processed_images()
        if name not in processed:
            return
        
        source_path = processed[name]
        
        from PySide6.QtWidgets import QFileDialog
        save_path, _ = QFileDialog.getSaveFileName(
            self, "导出的处理结果", name, 
            "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)"
        )
        
        if save_path:
            import shutil
            try:
                shutil.copy2(source_path, save_path)
                log_manager.info(f"导出的处理结果: {save_path}")
                self.status_label.setText(f"已导出: {os.path.basename(save_path)}")
                QMessageBox.information(self, "成功", f"已导出到:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
                log_manager.error(f"导出的处理结果失败: {e}")
    
    def _rename_processed_file(self, old_name):
        """重命名处理结果"""
        new_name, ok = QInputDialog.getText(
            self, "重命名", "请输入新名称:",
            QLineEdit.EchoMode.Normal, old_name
        )
        
        if ok and new_name and new_name != old_name:
            processed = self.workspace.get_all_processed_images()
            if old_name in processed:
                old_path = processed[old_name]
                
                # 检查新名称是否已存在
                if new_name in processed:
                    QMessageBox.warning(self, "警告", "名称已存在")
                    return
                
                # 更新
                self.workspace._data["processed_images"][new_name] = self.workspace._data["processed_images"].pop(old_name)
                
                # 刷新资源树
                self._refresh_layer_tree()
                
                log_manager.info(f"重命名处理结果: {old_name} -> {new_name}")
                self.status_label.setText(f"已重命名: {new_name}")
    
    def _delete_processed_file(self, name):
        """删除处理结果"""
        reply = QMessageBox.question(
            self, "确认删除", 
            f"确定要删除处理结果 '{name}' 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 从 Workspace 的 processed_images 中删除
            # 需要直接操作 _data
            if name in self.workspace._data.get("processed_images", {}):
                del self.workspace._data["processed_images"][name]
                self._refresh_layer_tree()
                log_manager.info(f"删除处理结果: {name}")
                self.status_label.setText(f"已删除: {name}")
    
    def _on_mouse_moved(self, col, row, r, g, b):
        """鼠标移动更新坐标"""
        self.coord_label.setText(f"像素: ({col}, {row}) RGB: ({r},{g},{b})")
        self.event_bus.publish(EventTopics.TOPIC_COORDINATE_CHANGED, {"col": col, "row": row, "r": r, "g": g, "b": b})
    
    def _on_image_loaded(self, path):
        """图像加载完成"""
        name = os.path.splitext(os.path.basename(path))[0]
        h, w = self.view_2d.image_size
        self.size_label.setText(f"{w} x {h}")
        log_manager.info(f"图像加载完成: {name}")
    
    def _on_log_message(self, timestamp, level, message):
        """日志消息"""
        self.log_text.append(f"[{timestamp}] {level}: {message}")
    
    def _refresh_layer_tree(self):
        """刷新资源树"""
        # 清空所有子节点
        self.root_images.takeChildren()
        
        # 刷新原始影像
        images = self.workspace.get_all_images()
        for name in images:
            item = QTreeWidgetItem([name])
            self.root_images.addChild(item)
        
        if images:
            self.root_images.setExpanded(True)
        
        # 刷新处理结果
        self.root_processed.takeChildren()
        processed_images = self.workspace.get_all_processed_images()
        for name in processed_images:
            item = QTreeWidgetItem([name])
            self.root_processed.addChild(item)
        
        if processed_images:
            self.root_processed.setExpanded(True)

        self.root_pointclouds.takeChildren()
        pointclouds = self.workspace.get_all_pointclouds()
        for name in pointclouds:
            self.root_pointclouds.addChild(QTreeWidgetItem([name]))
        if pointclouds:
            self.root_pointclouds.setExpanded(True)

        self.root_vectors.takeChildren()
        vectors = self.workspace.get_all_vectors()
        for name in vectors:
            self.root_vectors.addChild(QTreeWidgetItem([name]))
        if vectors:
            self.root_vectors.setExpanded(True)

        self.root_masks.takeChildren()
        masks = self.workspace.get_all_masks()
        for name in masks:
            self.root_masks.addChild(QTreeWidgetItem([name]))
        if masks:
            self.root_masks.setExpanded(True)

        self.root_dom.takeChildren()
        dom = self.workspace.get_dom()
        if dom and dom.get("name"):
            self.root_dom.addChild(QTreeWidgetItem([dom["name"]]))
            self.root_dom.setExpanded(True)

        self.root_dem.takeChildren()
        dem = self.workspace.get_dem()
        if dem and dem.get("name"):
            self.root_dem.addChild(QTreeWidgetItem([dem["name"]]))
            self.root_dem.setExpanded(True)
    
    # ==================== 事件订阅 ====================
    
    def _subscribe_events(self):
        """订阅 EventBus 事件"""
        self.event_bus.subscribe(EventTopics.TOPIC_IMAGE_ADDED, self._on_image_added_event)
        self.event_bus.subscribe(EventTopics.TOPIC_IMAGE_SELECTED, self._on_image_selected_event)
        self.event_bus.subscribe(EventTopics.TOPIC_IMAGE_UPDATED, self._on_image_updated_event)
        self.event_bus.subscribe(EventTopics.TOPIC_VECTOR_ADDED, self._on_vector_changed_event)
        self.event_bus.subscribe(EventTopics.TOPIC_VECTOR_UPDATED, self._on_vector_changed_event)
        self.event_bus.subscribe(EventTopics.TOPIC_VECTOR_REMOVED, self._on_vector_removed_event)
        self.event_bus.subscribe(EventTopics.TOPIC_VECTOR_SELECTED, self._on_vector_selected_event)
        self.event_bus.subscribe(EventTopics.TOPIC_VECTOR_EDIT_MODE_CHANGED, self._on_vector_edit_mode_changed_event)
        self.event_bus.subscribe(EventTopics.TOPIC_VIEW_SINGLE_REQUEST, self._on_view_single_request)
        self.event_bus.subscribe(EventTopics.TOPIC_VIEW_COMPARE_REQUEST, self._on_view_compare_request)
        self.event_bus.subscribe(EventTopics.TOPIC_VIEW_3D_REQUEST, self._on_view_3d_request)
        self.event_bus.subscribe(EventTopics.TOPIC_VIEW_MODE_CHANGED, self._on_view_mode_changed_event)
    
    def _on_image_added_event(self, data):
        """处理影像添加事件"""
        # data 可能是 dict {"name": name, "path": path} 或直接是 name
        if isinstance(data, dict):
            name = data.get("name", "未知")
        else:
            name = data
        
        # 刷新资源树
        self._refresh_layer_tree()
        
        log_manager.info(f"事件: 新增影像 {name}")
    
    def _on_image_selected_event(self, data):
        """处理影像选中事件"""
        if isinstance(data, dict):
            name = data.get("name")
            path = data.get("path")
        else:
            name = data
            path = self.workspace.get_image(name)
        
        if path and os.path.exists(path):
            self.view_2d.load_image(path) if hasattr(self.view_2d, 'load_image') else None
            h, w = self.view_2d.image_size
            self.size_label.setText(f"{w} x {h}")
            self.status_label.setText(f"显示: {name}")
            log_manager.info(f"事件: 选中影像 {name}")
    
    def _on_image_updated_event(self, data):
        """处理图像更新事件（模块处理结果）"""
        # 刷新资源树，显示处理结果
        self._refresh_layer_tree()
        
        name = data.get("name", "处理结果") if isinstance(data, dict) else "处理结果"
        path = data.get("path") if isinstance(data, dict) else None
        title = None
        summary = None
        if isinstance(data, dict):
            title = data.get("title")
            summary = data.get("summary")
            if not title and data.get("kind") == "dom":
                title = f"DOM 结果: {name}"
        if path and os.path.exists(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                self.view_2d.show_single_image(path, name=name)
                if summary:
                    self.view_2d.set_context_text(f"{title or name} · {summary}")
                elif title:
                    self.view_2d.set_context_text(title)
                if title and hasattr(self.view_2d, "set_context_text"):
                    self.status_label.setText(summary and f"{title} · {summary}" or title)
                else:
                    self.status_label.setText(f"处理结果: {name}")
                if isinstance(data, dict) and data.get("size_text"):
                    self.size_label.setText(data.get("size_text"))
                log_manager.info(f"结果已显示: {title or name} ({path})")
                return
        self.status_label.setText((title + (f" · {summary}" if summary else "")) if title else f"处理结果: {name}")
        log_manager.info(f"事件: 图像已更新 {name}")

    def _on_vector_changed_event(self, data):
        self._refresh_layer_tree()
        if not isinstance(data, dict):
            return
        name = data.get("name", "矢量")
        vector = data.get("vector") or self.workspace.get_vector(name)
        summary = data.get("summary") or data.get("status")
        tool = data.get("current_tool")
        layer_name = data.get("current_layer_name")
        draft_hint = ""
        if tool == "line":
            draft_hint = "线草稿: 继续点击添加顶点, 双击完成, Esc 取消"
        elif tool == "polygon":
            draft_hint = "面草稿: 继续点击添加顶点, 双击闭合完成, Esc 取消"
        status_bits = [bit for bit in [tool and f"工具: {tool}", layer_name and f"图层: {layer_name}", summary] if bit]
        if summary:
            self.status_label.setText(" · ".join(status_bits) if status_bits else summary)
        else:
            self.status_label.setText(" · ".join(status_bits) if status_bits else f"矢量已更新: {name}")
        if vector:
            self.view_2d.set_vector_interaction_mode(True)
            base_path = vector.get("source_image_path") or vector.get("meta", {}).get("source_image_path")
            if not base_path:
                source_image = vector.get("source_image")
                if source_image:
                    images = self.workspace.get_all_images()
                    base_path = images.get(source_image, source_image)
            if base_path and os.path.exists(base_path):
                self.view_2d.show_single_image(base_path, name=vector.get("source_image") or name)
            self.view_2d.render_vector_collection(
                vector,
                selected_feature_ids=data.get("selected_feature_ids"),
                hover_feature_id=data.get("hover_feature_id"),
                draft_geometry=data.get("draft_geometry"),
            )
            if data.get("title") or summary:
                title = data.get("title") or name
                context_bits = [title]
                if tool:
                    context_bits.append(f"工具: {tool}")
                if layer_name:
                    context_bits.append(f"图层: {layer_name}")
                if summary:
                    context_bits.append(summary)
                if draft_hint:
                    context_bits.append(draft_hint)
                self.view_2d.set_context_text(" · ".join(context_bits))
                self.status_label.setText(" · ".join(context_bits))
        log_manager.info(f"矢量已更新: {name}")

    def _on_vector_removed_event(self, data):
        self._refresh_layer_tree()
        name = data.get("name", "矢量") if isinstance(data, dict) else str(data)
        self.status_label.setText(f"矢量已删除: {name}")
        log_manager.info(f"矢量已删除: {name}")

    def _on_vector_selected_event(self, data):
        if not isinstance(data, dict):
            return
        name = data.get("name", "矢量")
        vector = data.get("vector") or self.workspace.get_vector(name)
        if not vector:
            return
        self.view_2d.set_vector_interaction_mode(True)
        base_path = vector.get("source_image_path") or vector.get("meta", {}).get("source_image_path")
        if not base_path:
            source_image = vector.get("source_image")
            if source_image:
                images = self.workspace.get_all_images()
                base_path = images.get(source_image, source_image)
        if base_path and os.path.exists(base_path):
            self.view_2d.show_single_image(base_path, name=vector.get("source_image") or name)
        self.view_2d.render_vector_collection(
            vector,
            selected_feature_ids=data.get("selected_feature_ids"),
            hover_feature_id=data.get("hover_feature_id"),
            draft_geometry=data.get("draft_geometry"),
        )
        title = data.get("title") or f"矢量: {name}"
        summary = data.get("summary")
        tool = data.get("current_tool")
        layer_name = data.get("current_layer_name")
        draft_hint = ""
        if tool == "line":
            draft_hint = "线草稿: 继续点击添加顶点, 双击完成, Esc 取消"
        elif tool == "polygon":
            draft_hint = "面草稿: 继续点击添加顶点, 双击闭合完成, Esc 取消"
        context_bits = [title]
        if tool:
            context_bits.append(f"工具: {tool}")
        if layer_name:
            context_bits.append(f"图层: {layer_name}")
        if summary:
            context_bits.append(summary)
        if draft_hint:
            context_bits.append(draft_hint)
        self.view_2d.set_context_text(" · ".join(context_bits))
        self.status_label.setText(" · ".join(context_bits))
        log_manager.info(f"矢量已选中: {name}")

    def _on_vector_edit_mode_changed_event(self, data):
        enabled = bool(data.get("enabled")) if isinstance(data, dict) else bool(data)
        self.view_2d.set_vector_interaction_mode(enabled)
        tool = data.get("current_tool") if isinstance(data, dict) else None
        layer_name = data.get("current_layer_name") if isinstance(data, dict) else None
        summary = data.get("summary") if isinstance(data, dict) else None
        bits = ["矢量编辑模式" if enabled else "浏览模式"]
        if tool:
            bits.append(f"工具: {tool}")
        if layer_name:
            bits.append(f"图层: {layer_name}")
        if summary:
            bits.append(summary)
        if tool == "line":
            bits.append("线草稿: 继续点击添加顶点, 双击完成, Esc 取消")
        elif tool == "polygon":
            bits.append("面草稿: 继续点击添加顶点, 双击闭合完成, Esc 取消")
        self.status_label.setText(" · ".join(bits))

    def _on_view_mouse_pressed(self, x, y, button):
        self.event_bus.publish(EventTopics.TOPIC_VIEW_MOUSE_PRESSED, {"x": x, "y": y, "button": button})

    def _on_view_mouse_released(self, x, y, button):
        self.event_bus.publish(EventTopics.TOPIC_VIEW_MOUSE_RELEASED, {"x": x, "y": y, "button": button})

    def _on_view_mouse_double_clicked(self, x, y, button):
        self.event_bus.publish(EventTopics.TOPIC_VIEW_MOUSE_DOUBLE_CLICKED, {"x": x, "y": y, "button": button})

    def _on_view_key_pressed(self, key):
        self.event_bus.publish(EventTopics.TOPIC_VIEW_KEY_PRESSED, {"key": key})

    def _on_view_mode_changed_event(self, data):
        mode = data.get("mode") if isinstance(data, dict) else data
        self._on_view_mode_changed(mode)

    def _on_view_mode_changed(self, mode: str):
        if mode == "compare":
            self.view_2d_action.setChecked(False)
            self.view_compare_action.setChecked(True)
            self.view_3d_action.setChecked(False)
            self.status_label.setText("对比模式")
        elif mode == "3d":
            self.view_2d_action.setChecked(False)
            self.view_compare_action.setChecked(False)
            self.view_3d_action.setChecked(True)
            self.status_label.setText("三维视图")
        else:
            self.view_2d_action.setChecked(True)
            self.view_compare_action.setChecked(False)
            self.view_3d_action.setChecked(False)
            self.status_label.setText("二维视图")

    def _on_view_single_request(self, data):
        if not isinstance(data, dict):
            return
        image = data.get("image")
        path = data.get("path")
        name = data.get("name", "")
        keypoints = data.get("keypoints")
        mask = data.get("mask")
        if image is not None:
            self.view_2d.show_single_image(image, name=name, keypoints=keypoints, mask=mask)
        elif path and os.path.exists(path):
            self.view_2d.show_single_image(path, name=name, keypoints=keypoints, mask=mask)
        self.status_label.setText(data.get("title", "二维显示"))

    def _on_view_compare_request(self, data):
        if not isinstance(data, dict):
            return
        left = data.get("left")
        right = data.get("right")
        if left is None or right is None:
            return
        self.view_2d.show_compare(
            left.get("image") if isinstance(left, dict) else left,
            right.get("image") if isinstance(right, dict) else right,
            left_name=left.get("name", "") if isinstance(left, dict) else "",
            right_name=right.get("name", "") if isinstance(right, dict) else "",
            left_keypoints=left.get("keypoints") if isinstance(left, dict) else None,
            right_keypoints=right.get("keypoints") if isinstance(right, dict) else None,
            matches=data.get("matches"),
            left_mask=left.get("mask") if isinstance(left, dict) else None,
            right_mask=right.get("mask") if isinstance(right, dict) else None,
            sync=data.get("sync", True),
        )
        self.status_label.setText(data.get("title", "对比显示"))

    def _on_view_3d_request(self, data):
        if not isinstance(data, dict):
            return
        data_type = data.get("data_type")
        payload = data.get("payload") if isinstance(data.get("payload"), dict) else None
        if data_type == "surface_grid" and payload:
            grid = payload.get("z_grid")
            if grid is not None:
                self.view_2d.show_surface(grid, title=data.get("title", payload.get("name", "DSM/DEM")))
                self.status_label.setText(data.get("title", payload.get("name", "三维显示")))
                return
        if data_type == "point_cloud" and payload:
            points = payload.get("points")
            if points is not None:
                self.view_2d.show_point_cloud(points, title=data.get("title", payload.get("name", "点云")))
                self.status_label.setText(data.get("title", payload.get("name", "三维显示")))
                return
        kind = data.get("kind", "point_cloud")
        if kind == "surface":
            grid = data.get("grid")
            path = data.get("path")
            if grid is not None:
                self.view_2d.show_surface(grid, title=data.get("title", "DSM/DEM"))
            elif path and os.path.exists(path):
                self.view_2d.load_surface_from_path(path)
        else:
            points = data.get("points")
            path = data.get("path")
            if points is not None:
                self.view_2d.show_point_cloud(points, colors=data.get("colors"), title=data.get("title", "点云"))
            elif path and os.path.exists(path):
                self.view_2d.load_point_cloud(path)
        self.status_label.setText(data.get("title", "三维显示"))
    
    # ==================== 视图菜单功能 ====================
    
    def _toggle_left_panel(self):
        """切换左侧面板显示"""
        if self.dock_layers.isVisible():
            self.dock_layers.hide()
            self.toggle_left_action.setChecked(False)
        else:
            self.dock_layers.show()
            self.toggle_left_action.setChecked(True)
    
    def _toggle_right_panel(self):
        """切换右侧面板显示"""
        if self.dock_tools.isVisible():
            self.dock_tools.hide()
            self.toggle_right_action.setChecked(False)
        else:
            self.dock_tools.show()
            self.toggle_right_action.setChecked(True)
    
    def _toggle_log_panel(self):
        """切换日志面板显示"""
        if self.dock_log.isVisible():
            self.dock_log.hide()
            self.toggle_log_action.setChecked(False)
        else:
            self.dock_log.show()
            self.toggle_log_action.setChecked(True)
    
    def _on_zoom_in(self):
        """放大"""
        if hasattr(self.view_2d, 'zoom_in'):
            self.view_2d.zoom_in()
    
    def _on_zoom_out(self):
        """缩小"""
        if hasattr(self.view_2d, 'zoom_out'):
            self.view_2d.zoom_out()


    # 在 MainWindow 类中添加对应的处理函数
    def _on_zoom_fit(self):
        """全图显示"""
        if hasattr(self.view_2d, 'fit_to_window'):
            self.view_2d.fit_to_window()
            log_manager.info("视图已重置为全图显示")

    def _on_change_theme(self, theme_name: str):
        """切换主题"""
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        
        theme_file = f"ui/styles/{theme_name}_theme.qss"
        try:
            with open(theme_file, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())
            log_manager.info(f"主题已切换: {theme_name}")
        except Exception as e:
            log_manager.error(f"切换主题失败: {e}")
    
    def _on_about(self):
        """关于对话框"""
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.about(self, "关于",
            "数字摄影测量实习平台 v1.0\n\n"
            "融合传统摄影测量与深度学习的综合实习平台\n\n"
            "支持模块化功能，可自由增删模块")
    
    # ==================== 任务引擎 ====================
    
    def _on_test_long_task(self):
        """测试长耗时任务"""
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在执行测试任务...")
        
        # 订阅任务信号
        self.task_engine.task_progress.connect(self._on_task_progress)
        self.task_engine.task_finished.connect(self._on_task_finished)
        self.task_engine.task_error.connect(self._on_task_error)
        
        # 启动任务
        task_id = self.task_engine.run_task(
            simulate_long_task,
            duration=5,
            task_name="测试任务"
        )
        
        log_manager.info(f"启动测试任务: {task_id}")
    
    def _on_task_progress(self, task_id: str, progress: int):
        """任务进度更新"""
        self.progress_bar.setValue(progress)
        log_manager.debug(f"任务进度: {progress}%")
    
    def _on_task_finished(self, task_id: str, result: Any):
        """任务完成"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"任务完成: {result}")
        log_manager.info(f"任务完成: {task_id} -> {result}")
        
        # 断开信号连接（避免信号累积）
        self.task_engine.task_progress.disconnect(self._on_task_progress)
        self.task_engine.task_finished.disconnect(self._on_task_finished)
        self.task_engine.task_error.disconnect(self._on_task_error)
    
    def _on_task_error(self, task_id: str, error: str):
        """任务错误"""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"任务出错: {error}")
        log_manager.error(f"任务错误: {task_id} -> {error}")
        
        # 断开信号连接
        self.task_engine.task_progress.disconnect(self._on_task_progress)
        self.task_engine.task_finished.disconnect(self._on_task_finished)
        self.task_engine.task_error.disconnect(self._on_task_error)
    
    # ==================== 拖放支持 ====================
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                files.append(path)
        
        if files:
            # 导入并显示
            for f in files:
                name = os.path.splitext(os.path.basename(f))[0]
                self.workspace.add_image(name, f)
            
            self.view_2d.load_image(files[0]) if hasattr(self.view_2d, 'load_image') else None
            self._refresh_layer_tree()
            self.status_label.setText(f"拖放导入: {os.path.basename(files[0])}")
            log_manager.info(f"拖放导入: {files[0]}")


def main():
    """启动函数"""
    from PySide6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    workspace = get_workspace()
    window = MainWindow(workspace)
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
