from __future__ import annotations
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QCheckBox,QFrame,QGraphicsEllipseItem,QGraphicsLineItem,QGraphicsPathItem,QGraphicsPixmapItem,QGraphicsScene,QGraphicsSimpleTextItem,QGraphicsView,QHBoxLayout,QLabel,QPushButton,QSplitter,QStackedWidget,QToolButton,QVBoxLayout,QWidget
from core.log_manager import log_manager
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
except Exception:
    FigureCanvas = NavigationToolbar = Figure = None

def _load_image_from_path(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None: raise ValueError(path)
    return img

def _to_bgr(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2: return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if a.shape[2] == 4: return cv2.cvtColor(a, cv2.COLOR_BGRA2BGR)
    return a.copy()

def _pixmap(img: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(_to_bgr(img), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy())

def _pts(points: Sequence[Any]) -> List[Tuple[float, float]]:
    out = []
    for p in points or []:
        if hasattr(p, 'pt'): x, y = p.pt
        elif isinstance(p, dict): x, y = p.get('x'), p.get('y')
        else: x, y = p[:2]
        out.append((float(x), float(y)))
    return out

def _xyz_array(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        arr = data
    else:
        seq = []
        for item in data or []:
            if isinstance(item, dict):
                pos = item.get('position', item.get('pos', item.get('xyz', item)))
                if isinstance(pos, dict):
                    seq.append((float(pos.get('x', 0.0)), float(pos.get('y', 0.0)), float(pos.get('z', 0.0))))
                else:
                    seq.append(tuple(pos[:3]))
            else:
                seq.append(tuple(item[:3]))
        arr = np.asarray(seq, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def _button_value(button: Any) -> int:
    return int(getattr(button, "value", button))

def _safe_remove_item(scene: QGraphicsScene, item: Any):
    if item is None:
        return
    try:
        scene.removeItem(item)
    except Exception:
        pass

def _normalize_scatter_colors(colors: Any, count: int):
    arr = np.asarray(colors)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size == count:
            return arr.astype(float)
        if arr.size in (3, 4):
            rgba = arr.astype(float)
            if float(np.nanmax(np.abs(rgba))) > 1.0:
                rgba = rgba / 255.0
            return tuple(rgba.tolist())
        return arr.astype(float)
    if arr.ndim == 2 and arr.shape[0] in (1, count) and arr.shape[1] in (3, 4):
        rgba = arr.astype(float)
        if float(np.nanmax(np.abs(rgba))) > 1.0:
            rgba = rgba / 255.0
        if rgba.shape[0] == 1:
            return tuple(rgba[0].tolist())
        return rgba
    return arr.astype(float)

class LayeredImageView(QGraphicsView):
    mouse_moved = Signal(int, int, int, int, int)
    mouse_pressed = Signal(float, float, int)
    mouse_released = Signal(float, float, int)
    mouse_double_clicked = Signal(float, float, int)
    key_pressed = Signal(int)
    image_loaded = Signal(str)
    zoom_changed = Signal(float)
    view_state_changed = Signal(float, object)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self); self.setScene(self._scene)
        self._base = None; self._img_item = None; self._kp_items = []; self._mask_item = None
        self._zoom = 1.0; self._min = 0.1; self._max = 10.0; self._step = 1.15
        self._panning = False; self._pan_start = None; self._vis = {'keypoints': True, 'mask': True}
        self._interaction_mode = 'view'
        self._vector_items = []
        self._vector_draft_item = None
        self._vector_hint_item = None
        self.setRenderHint(QPainter.RenderHint.Antialiasing); self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor('#f5f5f5')); self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded); self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse); self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag); self.setMouseTracking(True); self.viewport().setMouseTracking(True)
        self._welcome()
    @property
    def has_image(self): return self._base is not None
    @property
    def image_size(self): return (0,0) if self._base is None else (self._base.shape[1], self._base.shape[0])
    @property
    def zoom_factor(self): return self._zoom
    def _welcome(self):
        self._scene.clear(); self._img_item = None; self._kp_items = []; self._mask_item = None
        t = self._scene.addText('导入图像后开始使用\n支持叠加显示与对比模式')
        t.setDefaultTextColor(QColor('#9a9a9a')); t.setPos(-t.boundingRect().width()/2, -t.boundingRect().height()/2)
    def load_image(self, path: str):
        try: self.set_image(_load_image_from_path(path), os.path.basename(path)); self.image_loaded.emit(path); return True
        except Exception as e: log_manager.error(f'图像加载失败：{e}'); return False
    def set_image(self, img: np.ndarray, name: str=''):
        self.clear_vector_overlays()
        self._base = np.asarray(img); self._scene.clear(); self._img_item = QGraphicsPixmapItem(_pixmap(self._base)); self._scene.addItem(self._img_item); self._scene.setSceneRect(self._img_item.boundingRect()); self._kp_items=[]; self._mask_item=None; self.fit_to_window(); self.image_loaded.emit(name or '')
    def set_processed_image(self, img: np.ndarray, name: str=''): self.set_image(img, name)
    def _del_kp(self):
        for i in self._kp_items: self._scene.removeItem(i)
        self._kp_items=[]
    def _del_mask(self):
        if self._mask_item is not None: self._scene.removeItem(self._mask_item); self._mask_item=None
    def set_interaction_mode(self, mode: str):
        self._interaction_mode = mode if mode in {'view', 'vector_edit'} else 'view'
    def clear_vector_overlays(self):
        for item in self._vector_items:
            _safe_remove_item(self._scene, item)
        self._vector_items = []
        if self._vector_draft_item is not None:
            _safe_remove_item(self._scene, self._vector_draft_item)
            self._vector_draft_item = None
        if self._vector_hint_item is not None:
            _safe_remove_item(self._scene, self._vector_hint_item)
            self._vector_hint_item = None
    def _geometry_to_path(self, geometry: dict) -> QPainterPath | None:
        if not geometry:
            return None
        gtype = (geometry.get('type') or '').lower()
        coords = geometry.get('coordinates') or []
        path = QPainterPath()
        if gtype == 'point':
            if not coords:
                return None
            x, y = coords[:2]
            path.addEllipse(QPointF(float(x), float(y)), 3.0, 3.0)
            return path
        if gtype == 'linestring':
            if len(coords) < 2:
                return None
            first = coords[0]
            path.moveTo(float(first[0]), float(first[1]))
            for x, y in coords[1:]:
                path.lineTo(float(x), float(y))
            return path
        if gtype == 'polygon':
            ring = coords[0] if coords and isinstance(coords[0], (list, tuple)) and coords and isinstance(coords[0][0], (list, tuple)) else coords
            if len(ring) < 3:
                return None
            first = ring[0]
            path.moveTo(float(first[0]), float(first[1]))
            for x, y in ring[1:]:
                path.lineTo(float(x), float(y))
            path.closeSubpath()
            return path
        return None
    def render_vector_collection(self, vector_data: dict, selected_feature_ids=None, hover_feature_id=None, draft_geometry=None):
        self.clear_vector_overlays()
        if self._base is None or not vector_data:
            return
        selected_ids = set(selected_feature_ids or [])
        layers = vector_data.get('layers') or []
        for layer in layers:
            if not layer.get('visible', True):
                continue
            for feature in layer.get('features', []):
                geometry = feature.get('geometry') or {}
                path = self._geometry_to_path(geometry)
                if path is None:
                    continue
                feature_id = feature.get('feature_id')
                style = feature.get('style') or {}
                gtype = (geometry.get('type') or '').lower()
                selected = feature_id in selected_ids
                hover = feature_id == hover_feature_id
                stroke = QColor(style.get('stroke_color', '#1976d2'))
                fill = QColor(style.get('fill_color', '#1976d2'))
                fill.setAlpha(int(style.get('fill_opacity', 70)))
                pen = QPen(stroke, float(style.get('stroke_width', 2)) + (2 if selected or hover else 0))
                if hover:
                    pen.setColor(QColor('#ff9800'))
                elif selected:
                    pen.setColor(QColor('#ffd54f'))
                if gtype == 'point':
                    item = self._scene.addPath(path, pen, fill)
                    item.setZValue(25)
                    self._vector_items.append(item)
                elif gtype == 'polygon':
                    item = QGraphicsPathItem(path)
                    item.setPen(pen)
                    item.setBrush(fill if not selected else QColor(255, 235, 59, 80))
                    item.setZValue(20)
                    self._scene.addItem(item)
                    self._vector_items.append(item)
                else:
                    item = QGraphicsPathItem(path)
                    item.setPen(pen)
                    item.setBrush(Qt.BrushStyle.NoBrush)
                    item.setZValue(20)
                    self._scene.addItem(item)
                    self._vector_items.append(item)
                if style.get('label_visible'):
                    label_text = feature.get('properties', {}).get('名称') or feature.get('properties', {}).get('name') or feature_id[:8]
                    text_item = QGraphicsSimpleTextItem(str(label_text))
                    text_item.setBrush(QColor('#263238'))
                    bounds = path.boundingRect()
                    text_item.setPos(bounds.center())
                    text_item.setZValue(30)
                    self._scene.addItem(text_item)
                    self._vector_items.append(text_item)
        if draft_geometry:
            draft_path = self._geometry_to_path(draft_geometry)
            if draft_path is not None:
                draft_item = QGraphicsPathItem(draft_path)
                draft_pen = QPen(QColor('#d32f2f'), 2, Qt.PenStyle.DashLine)
                draft_item.setPen(draft_pen)
                draft_item.setBrush(Qt.BrushStyle.NoBrush)
                draft_item.setZValue(40)
                self._scene.addItem(draft_item)
                self._vector_draft_item = draft_item
                hint = '草稿绘制中: 双击完成, Esc 取消'
                gtype = (draft_geometry.get('type') or '').lower()
                if gtype == 'linestring':
                    hint = '线草稿: 继续点击添加顶点, 双击完成, Esc 取消'
                elif gtype == 'polygon':
                    hint = '面草稿: 继续点击添加顶点, 双击闭合完成, Esc 取消'
                hint_item = QGraphicsSimpleTextItem(hint)
                hint_item.setBrush(QColor('#d32f2f'))
                hint_item.setZValue(45)
                hint_item.setPos(12, 12)
                self._scene.addItem(hint_item)
                self._vector_hint_item = hint_item
    def set_overlay_visible(self, name: str, visible: bool):
        self._vis[name] = visible
        if name == 'keypoints':
            for i in self._kp_items: i.setVisible(visible)
        elif name == 'mask' and self._mask_item is not None: self._mask_item.setVisible(visible)
    def set_overlay_visible(self, name: str, visible: bool):
        self._vis[name] = visible
        if name == 'keypoints':
            for i in self._kp_items: i.setVisible(visible)
        elif name == 'mask' and self._mask_item is not None: self._mask_item.setVisible(visible)
    def clear_overlays(self): self._del_kp(); self._del_mask()
    def set_keypoints(self, points, color=(0,255,0), radius=4):
        self._del_kp()
        if self._base is None: return
        for x,y in _pts(points):
            it = QGraphicsEllipseItem(x-radius, y-radius, radius*2, radius*2); it.setPen(QPen(QColor(*color), 1.2)); it.setBrush(QColor(*color)); it.setZValue(20); self._scene.addItem(it); self._kp_items.append(it)
        self.set_overlay_visible('keypoints', self._vis.get('keypoints', True))
    def set_mask_overlay(self, mask, color=(255,0,0), opacity=0.35):
        self._del_mask();
        if mask is None or self._base is None: return
        m = np.asarray(mask); m = m[:,:,0] if m.ndim == 3 else m; alpha = (np.clip(m.astype(np.float32),0,255) > 0).astype(np.uint8) * int(255*opacity)
        rgba = np.zeros((alpha.shape[0], alpha.shape[1], 4), dtype=np.uint8); rgba[:,:,:3]=color; rgba[:,:,3]=alpha; qimg = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888).copy(); self._mask_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg)); self._mask_item.setZValue(10); self._scene.addItem(self._mask_item); self.set_overlay_visible('mask', self._vis.get('mask', True))
    def fit_to_window(self):
        if self._img_item is None: return
        r = self._img_item.boundingRect();
        if r.isEmpty(): return
        self.fitInView(r, Qt.AspectRatioMode.KeepAspectRatio); self._zoom = self.transform().m11(); self.zoom_changed.emit(self._zoom); self.view_state_changed.emit(self._zoom, self.mapToScene(self.viewport().rect().center()))
    def _apply_zoom(self):
        c = self.mapToScene(self.viewport().rect().center()); self.resetTransform(); self.scale(self._zoom, self._zoom); self.centerOn(c); self.zoom_changed.emit(self._zoom); self.view_state_changed.emit(self._zoom, c)
    def zoom_in(self):
        if self._img_item is None: return
        nz = min(self._max, self._zoom*self._step)
        if nz != self._zoom: self._zoom = nz; self._apply_zoom()
    def zoom_out(self):
        if self._img_item is None: return
        nz = max(self._min, self._zoom/self._step)
        if nz != self._zoom: self._zoom = nz; self._apply_zoom()
    def apply_view_state(self, zoom_factor: float, center_point: QPointF):
        self._zoom = max(self._min, min(self._max, zoom_factor)); self.resetTransform(); self.scale(self._zoom, self._zoom); self.centerOn(center_point)
    def wheelEvent(self, event):
        if self._img_item is None: return super().wheelEvent(event)
        old = self.mapToScene(event.position().toPoint()); f = self._step if event.angleDelta().y() > 0 else 1/self._step; nz = self._zoom * f
        if self._min <= nz <= self._max:
            self._zoom = nz; self.resetTransform(); self.scale(self._zoom, self._zoom); new = self.mapToScene(event.position().toPoint()); d = new - old; self.translate(d.x(), d.y()); self.zoom_changed.emit(self._zoom); self.view_state_changed.emit(self._zoom, self.mapToScene(self.viewport().rect().center()))
        event.accept()
    def mousePressEvent(self, event):
        p = self.mapToScene(event.pos())
        self.mouse_pressed.emit(float(p.x()), float(p.y()), _button_value(event.button()))
        if self._interaction_mode == 'vector_edit' and event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton): self._panning=True; self._pan_start=event.pos(); self.setCursor(Qt.CursorShape.OpenHandCursor); event.accept(); return
        super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if self._panning and self._pan_start is not None:
            d = event.pos() - self._pan_start; self._pan_start = event.pos(); self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-d.x()); self.verticalScrollBar().setValue(self.verticalScrollBar().value()-d.y()); self.view_state_changed.emit(self._zoom, self.mapToScene(self.viewport().rect().center()))
        if self._img_item is not None and self._base is not None:
            p = self.mapToScene(event.pos()); c, r = int(p.x()), int(p.y())
            if 0 <= c < self._base.shape[1] and 0 <= r < self._base.shape[0]:
                px = self._base[r, c]
                if isinstance(px, np.ndarray) and px.size >= 3: rr, gg, bb = int(px[0]), int(px[1]), int(px[2])
                else: rr = gg = bb = int(px)
                self.mouse_moved.emit(c, r, rr, gg, bb)
            else: self.mouse_moved.emit(c, r, -1, -1, -1)
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event):
        p = self.mapToScene(event.pos())
        self.mouse_released.emit(float(p.x()), float(p.y()), _button_value(event.button()))
        if self._interaction_mode == 'vector_edit' and event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton): self._panning=False; self._pan_start=None; self.setCursor(Qt.CursorShape.ArrowCursor); event.accept(); return
        super().mouseReleaseEvent(event)
    def mouseDoubleClickEvent(self, event):
        p = self.mapToScene(event.pos())
        self.mouse_double_clicked.emit(float(p.x()), float(p.y()), _button_value(event.button()))
        if self._interaction_mode == 'vector_edit' and event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
    def keyPressEvent(self, event):
        self.key_pressed.emit(int(event.key()))
        super().keyPressEvent(event)
    def clear(self): self._base=None; self._zoom=1.0; self.clear_vector_overlays(); self._scene.clear(); self._kp_items=[]; self._mask_item=None; self._welcome()

class MatchOverlayWidget(QWidget):
    def __init__(self, compare_view, parent=None):
        super().__init__(parent or compare_view); self._v = compare_view; self._matches=[]; self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True); self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True); self.setStyleSheet('background: transparent;')
    def set_matches(self, matches): self._matches=list(matches or []); self.update()
    def clear(self): self._matches=[]; self.update()
    def paintEvent(self, event):
        if not self._matches: return
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for m in self._matches:
            l, r = m.get('left'), m.get('right')
            if not l or not r: continue
            inlier = bool(m.get('inlier', True)); pen = QPen(QColor(46,125,50) if inlier else QColor(198,40,40), 1.2)
            if not inlier: pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            lp = self._v.left_view.viewport().mapTo(self, self._v.left_view.mapFromScene(QPointF(*l)).toPoint())
            rp = self._v.right_view.viewport().mapTo(self, self._v.right_view.mapFromScene(QPointF(*r)).toPoint())
            p.drawLine(lp, rp)
        p.end()

class CompareView(QWidget):
    mouse_moved = Signal(int,int,int,int,int)
    image_loaded = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent); self._sync_enabled=True; self._busy=False; self._active='left'; self._build_ui(); self._wire(); self._overlay = MatchOverlayWidget(self, self)
    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        self.title_label = QLabel('对比模式'); self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.title_label.setStyleSheet('font-weight:600;color:#345;padding:6px;'); lay.addWidget(self.title_label)
        bar = QHBoxLayout(); self.sync_check = QCheckBox('同步浏览'); self.sync_check.setChecked(True); self.sync_check.toggled.connect(self.set_sync_enabled); self.reset_btn = QPushButton('重置视图'); self.reset_btn.clicked.connect(self.fit_to_window); bar.addWidget(self.sync_check); bar.addWidget(self.reset_btn); bar.addStretch(1); lay.addLayout(bar)
        sp = QSplitter(Qt.Orientation.Horizontal); self.left_title = QLabel('左图'); self.right_title = QLabel('右图'); self.left_view = LayeredImageView(); self.right_view = LayeredImageView(); sp.addWidget(self._wrap(self.left_view, self.left_title)); sp.addWidget(self._wrap(self.right_view, self.right_title)); sp.setSizes([1,1]); lay.addWidget(sp, 1); self._splitter = sp
    def _wrap(self, view, title):
        panel = QFrame(); panel.setFrameShape(QFrame.Shape.StyledPanel); panel.setStyleSheet('QFrame { background: #f8f9fb; border: 1px solid #d9e1ec; }'); l = QVBoxLayout(panel); l.setContentsMargins(0,0,0,0); title.setAlignment(Qt.AlignmentFlag.AlignCenter); title.setStyleSheet('font-weight:600;color:#345;padding:6px;'); l.addWidget(title); l.addWidget(view, 1); return panel
    def _wire(self):
        self.left_view.mouse_moved.connect(self._on_left_mouse_moved); self.right_view.mouse_moved.connect(self._on_right_mouse_moved); self.left_view.image_loaded.connect(self._on_image_loaded); self.right_view.image_loaded.connect(self._on_image_loaded); self.left_view.view_state_changed.connect(lambda z,c: self._sync_view('left', z, c)); self.right_view.view_state_changed.connect(lambda z,c: self._sync_view('right', z, c))
    def resizeEvent(self, e): super().resizeEvent(e); self._overlay.setGeometry(self.rect()); self._overlay.raise_()
    def _on_left_mouse_moved(self, c,r,rr,gg,bb): self._active='left'; self.mouse_moved.emit(c,r,rr,gg,bb)
    def _on_right_mouse_moved(self, c,r,rr,gg,bb): self._active='right'; self.mouse_moved.emit(c,r,rr,gg,bb)
    def _on_image_loaded(self, path): self.image_loaded.emit(path)
    def set_sync_enabled(self, enabled: bool): self._sync_enabled = enabled; self.sync_check.setChecked(enabled)
    def set_title(self, title: str): self.title_label.setText(title or '对比模式')
    def _sync_view(self, src, z, c):
        if not self._sync_enabled or self._busy: return
        tgt = self.right_view if src == 'left' else self.left_view; self._busy=True
        try: tgt.apply_view_state(z, c)
        finally: self._busy=False
    def load_left_image(self, image, name=''): self.left_view.load_image(str(image)) if isinstance(image, (str,os.PathLike)) else self.left_view.set_image(np.asarray(image), name)
    def load_right_image(self, image, name=''): self.right_view.load_image(str(image)) if isinstance(image, (str,os.PathLike)) else self.right_view.set_image(np.asarray(image), name)
    def set_left_keypoints(self, pts, color=(76,175,80)): self.left_view.set_keypoints(pts, color=color)
    def set_right_keypoints(self, pts, color=(33,150,243)): self.right_view.set_keypoints(pts, color=color)
    def set_left_mask(self, mask): self.left_view.set_mask_overlay(mask)
    def set_right_mask(self, mask): self.right_view.set_mask_overlay(mask)
    def set_matches(self, matches): self._overlay.set_matches(matches)
    def clear(self): self.left_view.clear(); self.right_view.clear(); self._overlay.clear(); self.set_title('对比模式')
    def fit_to_window(self): self.left_view.fit_to_window(); self.right_view.fit_to_window()
    def zoom_in(self):
        if self._active == 'right': self.right_view.zoom_in(); self.left_view.apply_view_state(self.right_view.zoom_factor, self.right_view.mapToScene(self.right_view.viewport().rect().center())) if self._sync_enabled else None
        else: self.left_view.zoom_in(); self.right_view.apply_view_state(self.left_view.zoom_factor, self.left_view.mapToScene(self.left_view.viewport().rect().center())) if self._sync_enabled else None
    def zoom_out(self):
        if self._active == 'right': self.right_view.zoom_out(); self.left_view.apply_view_state(self.right_view.zoom_factor, self.right_view.mapToScene(self.right_view.viewport().rect().center())) if self._sync_enabled else None
        else: self.left_view.zoom_out(); self.right_view.apply_view_state(self.left_view.zoom_factor, self.left_view.mapToScene(self.left_view.viewport().rect().center())) if self._sync_enabled else None
    def set_vector_interaction_mode(self, enabled: bool): self.single_view.set_interaction_mode('vector_edit' if enabled else 'view')
    def render_vector_collection(self, vector_data: dict, selected_feature_ids=None, hover_feature_id=None, draft_geometry=None):
        self.set_mode('2d')
        self.single_view.render_vector_collection(vector_data, selected_feature_ids=selected_feature_ids, hover_feature_id=hover_feature_id, draft_geometry=draft_geometry)
    def clear_vector_overlays(self): self.single_view.clear_vector_overlays()

class ThreeDView(QWidget):
    def __init__(self, parent=None): super().__init__(parent); self._build_ui()
    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); self.header = QLabel('三维视图'); self.header.setStyleSheet('font-weight:600;color:#345;padding:6px;'); lay.addWidget(self.header)
        if FigureCanvas is None or Figure is None: self.placeholder = QLabel('当前环境未安装 matplotlib，无法显示三维视图'); self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter); lay.addWidget(self.placeholder, 1); self.canvas = self.toolbar = self.axes = None; self.status = QLabel(''); return
        self.figure = Figure(figsize=(6,4), dpi=100); self.canvas = FigureCanvas(self.figure); self.axes = self.figure.add_subplot(111, projection='3d'); self.axes.set_facecolor('#ffffff'); self.figure.patch.set_facecolor('#f5f7fa'); lay.addWidget(self.canvas, 1); self.toolbar = NavigationToolbar(self.canvas, self) if NavigationToolbar is not None else None; self.toolbar and lay.addWidget(self.toolbar); self.status = QLabel('暂无三维数据'); self.status.setStyleSheet('color:#666;padding:4px;'); lay.addWidget(self.status)
    def clear(self):
        if self.axes is None: return
        self.axes.cla(); self.axes.set_xlabel('X'); self.axes.set_ylabel('Y'); self.axes.set_zlabel('Z'); self.status.setText('暂无三维数据'); self.canvas.draw_idle()
    def _scale(self, factor: float):
        if self.axes is None: return
        for get, set_ in ((self.axes.get_xlim3d, self.axes.set_xlim3d), (self.axes.get_ylim3d, self.axes.set_ylim3d), (self.axes.get_zlim3d, self.axes.set_zlim3d)):
            lo, hi = get(); c=(lo+hi)/2.0; h=max((hi-lo)/2.0*factor, 1e-6); set_(c-h, c+h)
        self.canvas.draw_idle()
    def zoom_in(self): self._scale(0.85)
    def zoom_out(self): self._scale(1.15)
    def reset_view(self): self.axes and (self.axes.view_init(elev=30, azim=-60), self.canvas.draw_idle())
    def _bounds(self, xyz):
        mn, mx = xyz.min(axis=0), xyz.max(axis=0); c=(mn+mx)/2.0; r=max(float(np.max(mx-mn)/2.0), 1e-6); self.axes.set_xlim(c[0]-r, c[0]+r); self.axes.set_ylim(c[1]-r, c[1]+r); self.axes.set_zlim(c[2]-r, c[2]+r); hasattr(self.axes,'set_box_aspect') and self.axes.set_box_aspect((1,1,1)); self.axes.grid(True, alpha=0.2)
    def show_point_cloud(self, points, colors=None, title='点云'):
        if self.axes is None: return
        arr = np.asarray(points)
        if arr.ndim != 2 or arr.shape[1] < 3: raise ValueError('点云数据需要 Nx3 格式')
        xyz = arr[:, :3]
        self.axes.cla()
        if colors is None:
            scatter_colors = xyz[:, 2]
            cmap = 'viridis'
        else:
            scatter_colors = _normalize_scatter_colors(colors, xyz.shape[0])
            cmap = 'viridis' if np.asarray(scatter_colors).ndim == 1 else None
        self.axes.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=scatter_colors,
            cmap=cmap,
            s=1.8,
            alpha=0.92,
        )
        self._bounds(xyz); self.axes.view_init(elev=28, azim=-60); self.axes.set_title(title); self.axes.set_xlabel('X'); self.axes.set_ylabel('Y'); self.axes.set_zlabel('Z'); self.status.setText(f'点云：{xyz.shape[0]} 个点'); self.canvas.draw_idle()
    def show_surface(self, grid, title='DSM/DEM'):
        if self.axes is None: return
        z = np.asarray(grid)
        if z.ndim != 2: raise ValueError('地表网格数据必须是二维数组')
        y, x = np.mgrid[0:z.shape[0], 0:z.shape[1]]; self.axes.cla(); self.axes.plot_surface(x, y, z, cmap='terrain', linewidth=0, antialiased=True, alpha=0.96); hasattr(self.axes,'set_box_aspect') and self.axes.set_box_aspect((1,1,0.55)); self.axes.view_init(elev=32, azim=-60); self.axes.grid(True, alpha=0.2); self.axes.set_title(title); self.axes.set_xlabel('X'); self.axes.set_ylabel('Y'); self.axes.set_zlabel('Z'); self.status.setText(f'地表网格：{z.shape[1]} × {z.shape[0]}'); self.canvas.draw_idle()
    def show_camera_poses(self, poses, title='相机位姿'):
        if self.axes is None: return
        arr = _xyz_array(poses)
        if arr.ndim != 2 or arr.shape[1] < 3: raise ValueError('位姿数据需要 Nx3 格式')
        xyz = arr[:, :3]; self.axes.cla(); self.axes.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c=np.linspace(0,1,xyz.shape[0]), cmap='plasma', s=18)
        if arr.shape[1] >= 6:
            d = arr[:, 3:6]; self.axes.quiver(xyz[:,0], xyz[:,1], xyz[:,2], d[:,0], d[:,1], d[:,2], length=1.0, normalize=True, color='tab:blue', alpha=0.6)
        for i, (x, y, z) in enumerate(xyz[:40]): self.axes.text(x, y, z, str(i+1), fontsize=8)
        self._bounds(xyz); self.axes.view_init(elev=28, azim=-60); self.axes.set_title(title); self.axes.set_xlabel('X'); self.axes.set_ylabel('Y'); self.axes.set_zlabel('Z'); self.status.setText(f'相机位姿：{xyz.shape[0]} 个'); self.canvas.draw_idle()
    def _read_ascii_ply(self, path: str):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            if f.readline().strip() != 'ply': raise ValueError('不是有效的 PLY 文件')
            ok = False; n = None
            while True:
                line = f.readline()
                if not line: raise ValueError('Incomplete PLY header')
                line = line.strip()
                if line.startswith('format ascii'): ok = True
                if line.startswith('element vertex'): n = int(line.split()[-1])
                if line == 'end_header': break
            if not ok: raise ValueError('仅支持 ASCII PLY 文件')
            if n is None: raise ValueError('未找到 PLY 顶点数量')
            rows=[]
            for _ in range(n):
                parts = f.readline().split()
                if len(parts) >= 3: rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            return np.asarray(rows, dtype=np.float32)
    def load_point_cloud(self, path: str):
        ext = Path(path).suffix.lower()
        if ext == '.npy': pts = np.load(path)
        elif ext == '.csv': pts = np.loadtxt(path, delimiter=',')
        elif ext in {'.txt', '.xyz', '.pts'}: pts = np.loadtxt(path)
        elif ext == '.ply': pts = self._read_ascii_ply(path)
        else: raise ValueError(f'不支持的点云格式：{ext}')
        self.show_point_cloud(pts, title=os.path.basename(path))
    def load_surface_from_path(self, path: str):
        ext = Path(path).suffix.lower()
        if ext == '.npy': grid = np.load(path)
        elif ext in {'.csv', '.txt'}: grid = np.loadtxt(path, delimiter=',' if ext == '.csv' else None)
        else: img = _load_image_from_path(path); grid = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        self.show_surface(np.asarray(grid), title=os.path.basename(path))
    def load_camera_poses_from_path(self, path: str):
        ext = Path(path).suffix.lower()
        poses = np.load(path) if ext == '.npy' else np.loadtxt(path, delimiter=',' if ext == '.csv' else None)
        self.show_camera_poses(np.asarray(poses), title=os.path.basename(path))

class CentralDisplayWidget(QWidget):
    mouse_moved = Signal(int,int,int,int,int)
    mouse_pressed = Signal(float, float, int)
    mouse_released = Signal(float, float, int)
    mouse_double_clicked = Signal(float, float, int)
    key_pressed = Signal(int)
    image_loaded = Signal(str)
    mode_changed = Signal(str)
    def __init__(self, parent=None): super().__init__(parent); self.current_mode = '2d'; self._build_ui(); self._wire()
    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)
        bar = QFrame(); bar.setStyleSheet('QFrame { background:#eef3f8; border-bottom:1px solid #d9e1ec; }'); bl = QHBoxLayout(bar); bl.setContentsMargins(8,6,8,6); bl.setSpacing(6)
        self.btn_2d = QToolButton(); self.btn_2d.setText('二维视图'); self.btn_2d.setCheckable(True)
        self.btn_compare = QToolButton(); self.btn_compare.setText('对比模式'); self.btn_compare.setCheckable(True)
        self.btn_3d = QToolButton(); self.btn_3d.setText('三维视图'); self.btn_3d.setCheckable(True)
        self.btn_keypoints = QCheckBox('关键点'); self.btn_keypoints.setChecked(True)
        self.btn_mask = QCheckBox('掩膜'); self.btn_mask.setChecked(True)
        self.btn_matches = QCheckBox('匹配线'); self.btn_matches.setChecked(True)
        self.btn_sync = QCheckBox('同步浏览'); self.btn_sync.setChecked(True)
        self.btn_2d.clicked.connect(lambda: self.set_mode('2d')); self.btn_compare.clicked.connect(lambda: self.set_mode('compare')); self.btn_3d.clicked.connect(lambda: self.set_mode('3d'))
        self.btn_keypoints.toggled.connect(lambda v: self._apply_overlay_visibility('keypoints', v)); self.btn_mask.toggled.connect(lambda v: self._apply_overlay_visibility('mask', v)); self.btn_matches.toggled.connect(lambda v: self._apply_overlay_visibility('matches', v)); self.btn_sync.toggled.connect(self.set_compare_sync)
        for w in (self.btn_2d, self.btn_compare, self.btn_3d): bl.addWidget(w)
        bl.addSpacing(16)
        for w in (self.btn_keypoints, self.btn_mask, self.btn_matches): bl.addWidget(w)
        bl.addStretch(1); bl.addWidget(self.btn_sync); lay.addWidget(bar)
        self.context_label = QLabel('当前显示：等待结果')
        self.context_label.setStyleSheet('color:#567;padding:6px 8px;background:#f8fbff;border-bottom:1px solid #d9e1ec;')
        lay.addWidget(self.context_label)
        self.stack = QStackedWidget(); self.single_view = LayeredImageView(); self.compare_view = CompareView(); self.three_d_view = ThreeDView(); self.stack.addWidget(self.single_view); self.stack.addWidget(self.compare_view); self.stack.addWidget(self.three_d_view); lay.addWidget(self.stack, 1)
        self._mode_buttons = {'2d': self.btn_2d, 'compare': self.btn_compare, '3d': self.btn_3d}; self._apply_mode_button_state('2d'); self._update_overlay_control_state()
    def _wire(self):
        self.single_view.mouse_moved.connect(self.mouse_moved); self.single_view.mouse_pressed.connect(self.mouse_pressed); self.single_view.mouse_released.connect(self.mouse_released); self.single_view.mouse_double_clicked.connect(self.mouse_double_clicked); self.single_view.key_pressed.connect(self.key_pressed); self.single_view.image_loaded.connect(self.image_loaded); self.compare_view.mouse_moved.connect(self.mouse_moved); self.compare_view.image_loaded.connect(self.image_loaded)
    def _apply_mode_button_state(self, mode: str):
        for k,b in self._mode_buttons.items(): b.setChecked(k == mode)
    def _apply_overlay_visibility(self, name: str, visible: bool):
        if self.current_mode == 'compare': self.compare_view.left_view.set_overlay_visible(name, visible); self.compare_view.right_view.set_overlay_visible(name, visible)
        else: self.single_view.set_overlay_visible(name, visible)
    def _update_overlay_control_state(self): self.btn_sync.setEnabled(self.current_mode == 'compare'); self.btn_matches.setEnabled(True)
    def set_context_text(self, text: str):
        self.context_label.setText(text or '当前显示：等待结果')
    def set_mode(self, mode: str):
        mode = (mode or '2d').lower(); mode = mode if mode in {'2d','compare','3d'} else '2d'; self.current_mode = mode; self.stack.setCurrentIndex({'2d':0,'compare':1,'3d':2}[mode]); self._apply_mode_button_state(mode); self._update_overlay_control_state(); self.mode_changed.emit(mode)
    def set_compare_sync(self, enabled: bool): self.compare_view.set_sync_enabled(enabled); self.btn_sync.setChecked(enabled)
    @property
    def has_image(self): return self.compare_view.left_view.has_image or self.compare_view.right_view.has_image if self.current_mode == 'compare' else (False if self.current_mode == '3d' else self.single_view.has_image)
    @property
    def image_size(self): return self.compare_view.left_view.image_size if self.current_mode == 'compare' else ((0,0) if self.current_mode == '3d' else self.single_view.image_size)
    def load_image(self, path: str): self.set_mode('2d'); self.set_context_text(f'原始影像：{os.path.basename(path)}'); return self.single_view.load_image(path)
    def load_image_from_array(self, image: np.ndarray, name: str = '', mode: str = '2d'): self.set_mode(mode); self.set_context_text(f'单图结果：{name or "数组影像"}'); self.single_view.set_image(image, name); self.image_loaded.emit(name)
    def show_single_image(self, image: Any, name: str = '', keypoints: Optional[Sequence[Any]] = None, mask: Optional[np.ndarray] = None):
        self.set_mode('2d')
        display_name = name or (os.path.basename(str(image)) if isinstance(image, (str, os.PathLike)) else '单图结果')
        self.set_context_text(f'单图结果：{display_name}')
        self.single_view.load_image(str(image)) if isinstance(image, (str, os.PathLike)) else self.single_view.set_image(np.asarray(image), name)
        keypoints is not None and self.single_view.set_keypoints(keypoints); mask is not None and self.single_view.set_mask_overlay(mask)
    def show_compare(self, left: Any, right: Any, left_name: str = '', right_name: str = '', title: str = '', left_keypoints: Optional[Sequence[Any]] = None, right_keypoints: Optional[Sequence[Any]] = None, matches: Optional[Sequence[dict]] = None, left_mask: Optional[np.ndarray] = None, right_mask: Optional[np.ndarray] = None, sync: bool = True):
        self.set_mode('compare'); self.set_compare_sync(sync); title and self.compare_view.set_title(title)
        compare_title = title
        if not compare_title:
            names = [n for n in (left_name, right_name) if n]
            compare_title = f"对比结果：{' vs '.join(names)}" if names else "对比结果"
        self.set_context_text(compare_title)
        self.compare_view.load_left_image(str(left)) if isinstance(left, (str, os.PathLike)) else self.compare_view.load_left_image(np.asarray(left), left_name); self.compare_view.load_right_image(str(right)) if isinstance(right, (str, os.PathLike)) else self.compare_view.load_right_image(np.asarray(right), right_name); left_keypoints is not None and self.compare_view.set_left_keypoints(left_keypoints); right_keypoints is not None and self.compare_view.set_right_keypoints(right_keypoints); left_mask is not None and self.compare_view.set_left_mask(left_mask); right_mask is not None and self.compare_view.set_right_mask(right_mask); self.compare_view.set_matches(matches or [])
    def show_point_cloud(self, points: np.ndarray, colors: Optional[np.ndarray] = None, title: str = '点云'): self.set_mode('3d'); self.set_context_text(f'3D 点云：{title}'); self.three_d_view.show_point_cloud(points, colors=colors, title=title)
    def show_surface(self, grid: np.ndarray, title: str = 'DSM/DEM'): self.set_mode('3d'); self.set_context_text(f'3D 表面：{title}'); self.three_d_view.show_surface(grid, title=title)
    def show_camera_poses(self, poses: np.ndarray, title: str = '相机位姿'): self.set_mode('3d'); self.set_context_text(f'3D 相机位姿：{title}'); self.three_d_view.show_camera_poses(poses, title=title)
    def load_point_cloud(self, path: str): self.set_mode('3d'); self.set_context_text(f'3D 点云：{os.path.basename(path)}'); self.three_d_view.load_point_cloud(path)
    def load_surface_from_path(self, path: str): self.set_mode('3d'); self.set_context_text(f'3D 表面：{os.path.basename(path)}'); self.three_d_view.load_surface_from_path(path)
    def load_camera_poses_from_path(self, path: str): self.set_mode('3d'); self.set_context_text(f'3D 相机位姿：{os.path.basename(path)}'); self.three_d_view.load_camera_poses_from_path(path)
    def clear(self): self.single_view.clear(); self.compare_view.clear(); self.three_d_view.clear(); self.set_context_text('当前显示：等待结果'); self.set_mode('2d')
    def clear_overlays(self): self.compare_view.left_view.clear_overlays(); self.compare_view.right_view.clear_overlays(); self.compare_view.set_matches([]) if self.current_mode == 'compare' else self.single_view.clear_overlays()
    def set_vector_interaction_mode(self, enabled: bool): self.single_view.set_interaction_mode('vector_edit' if enabled else 'view')
    def render_vector_collection(self, vector_data: dict, selected_feature_ids=None, hover_feature_id=None, draft_geometry=None):
        self.set_mode('2d')
        self.single_view.render_vector_collection(vector_data, selected_feature_ids=selected_feature_ids, hover_feature_id=hover_feature_id, draft_geometry=draft_geometry)
    def clear_vector_overlays(self): self.single_view.clear_vector_overlays()
    def zoom_in(self): self.compare_view.zoom_in() if self.current_mode == 'compare' else (self.three_d_view.zoom_in() if self.current_mode == '3d' else self.single_view.zoom_in())
    def zoom_out(self): self.compare_view.zoom_out() if self.current_mode == 'compare' else (self.three_d_view.zoom_out() if self.current_mode == '3d' else self.single_view.zoom_out())
    def fit_to_window(self): self.compare_view.fit_to_window() if self.current_mode == 'compare' else (self.three_d_view.reset_view() if self.current_mode == '3d' else self.single_view.fit_to_window())
