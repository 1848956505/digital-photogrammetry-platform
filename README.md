# 数字摄影测量实习平台

> 融合传统摄影测量与深度学习的综合实习平台

## 📋 项目简介

本项目是一个模块化的数字摄影测量实习平台，采用**微内核（插件式）**架构设计。平台将传统摄影测量算法与深度学习技术相结合，提供从基础影像处理到高级三维重建的完整功能链。

**核心特性**：
- 🏗️ 微内核架构 - 模块可自由增删
- 🎨 简洁UI - Fusion风格浅色设计
- 🔄 插件机制 - 热插拔式功能模块
- 🤖 AI融合 - 支持深度学习与传统算法对比

## 

---

## 🏗️ 整体架构

```
数字摄影测量实习平台/
│
├── main.py                      # 程序启动入口
├── requirements.txt             # 依赖管理
├── SPEC.md                      # 技术规格文档
│
├── core/                        # 核心框架层
│   ├── base_interface.py       # IPlugin 接口规范
│   ├── workspace.py            # 全局工作空间
│   ├── log_manager.py          # 日志管理
│   ├── plugin_manager.py       # 插件加载器
│   ├── event_bus.py            # 事件总线
│   └── task_engine.py          # 任务引擎
│
├── ui/                         # 界面层
│   ├── main_window.py         # 主窗口
│   ├── image_viewer.py        # 2D影像视图
│   └── styles/                 # 主题样式
│
├── widgets/                    # 公共组件
│   └── ...
│
└── plugins/                    # 插件模块层
    ├── mod1_image_process/    # 模块一：基础影像处理
    ├── mod2_aerial_tri/       # 模块二：空中三角测量
    ├── mod3_dsm_dem/          # 模块三：DSM/DEM生产
    ├── mod4_dom/              # 模块四：DOM生产
    ├── mod5_dlg/              # 模块五：DLG数字线划图
    ├── mod6_dl_interpret/     # 模块六：深度学习解译
    └── mod7_mipmap_3d/        # 模块七：MipMap三维重建
```



---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- PySide6 6.5.0+
- OpenCV 4.8.0+
- NumPy 1.24.0+

### 安装

```bash
# 克隆或下载项目
cd "项目目录"

# 使用虚拟环境（推荐）
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 最小安装（仅测试UI）
pip install PySide6 numpy opencv-python Pillow scipy

# 完整安装
pip install -r requirements.txt
```

### 运行

```bash
python main.py
```

### 使用流程

1. **启动**：运行 `python main.py`
2. **导入图像**：
   - 菜单：文件 → 导入影像
   - 拖放：直接将图像文件拖到窗口
3. **使用功能**：
   - 右侧"工具箱"面板选择模块
   - 设置参数后点击"应用"
4. **查看结果**：
   - 中央2D视图自动显示处理结果
   - 左侧"处理结果"文件夹显示历史记录



---

## 📄 License

MIT License

---

*更新时间：2026-04-03*
*版本：1.0.1*