from __future__ import annotations

import math
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox, QInputDialog, QWidget

from core.base_interface import IPlugin
from core.event_bus import EventTopics, get_event_bus
from core.log_manager import log_manager
from core.workspace import Workspace

from .algorithms.vector_exporters import export_geojson_all_layers, export_geojson_layer
from .models import build_vector_collection, create_feature, create_layer, default_schema, find_feature, find_layer, summarize_vector_collection, update_derived_properties
from .ui import DlgControlPanel


class DlgPlugin(IPlugin):
    def __init__(self, workspace: Workspace):
        super().__init__(workspace)
        self.event_bus = get_event_bus()
        self.panel = DlgControlPanel()
        self._subs: List[Tuple[str, Any]] = []
        self._name = ""
        self._base = ""
        self._state = {"active_layer_id": None, "current_tool": "select", "draft_vertices": [], "draft_geometry": None, "selected_feature_ids": [], "hover_feature_id": None, "is_dirty": False, "mouse_pos": [0.0, 0.0]}
        self._wire_panel()
        self._refresh_workspace(True)

    def plugin_info(self) -> Dict[str, Any]:
        return {"name": "DLG", "group": "模块五", "version": "1.0.0", "description": "基于底图的 DLG 数字化、编辑、管理与 GeoJSON 导出"}

    def get_ui_panel(self) -> QWidget:
        return self.panel

    def execute(self, *args, **kwargs):
        self._refresh_workspace(True)
        if not self._name:
            return {"success": False, "message": "请先导入底图，再开始 DLG 数字化"}
        vector = self.workspace.get_vector(self._name)
        summary = summarize_vector_collection(vector) if vector else ""
        self._publish_state(EventTopics.TOPIC_VECTOR_SELECTED, vector, summary, f"DLG 结果: {self._name}")
        return {"success": True, "message": f"DLG 模块已就绪: {self._name}", "collection": self._name, "summary": summary}

    def on_activate(self):
        self._subscribe()
        self._refresh_workspace(True)
        self._publish_mode(True)
        self._refresh_panel()
        log_manager.info("模块五 DLG 已激活")

    def on_deactivate(self):
        self._unsubscribe()
        self._publish_mode(False)
        self._state["draft_vertices"] = []
        self._state["draft_geometry"] = None
        self.panel.set_status_message("模块已退出编辑模式")
        log_manager.info("模块五 DLG 已停用")

    def _wire_panel(self):
        p = self.panel
        p.refresh_requested.connect(lambda: self._refresh_workspace(True))
        p.base_image_changed.connect(self._on_base_changed)
        p.tool_changed.connect(self._on_tool_changed)
        p.new_layer_requested.connect(self._new_layer)
        p.delete_layer_requested.connect(self._delete_layer)
        p.layer_selected.connect(self._select_layer)
        p.layer_visibility_changed.connect(self._toggle_visibility)
        p.save_feature_requested.connect(self._save_feature)
        p.add_field_requested.connect(self._add_field)
        p.rename_field_requested.connect(self._rename_field)
        p.delete_field_requested.connect(self._delete_field)
        p.save_result_requested.connect(self._save_result)
        p.export_requested.connect(self._export)

    def _subscribe(self):
        if self._subs:
            return
        items = [
            (EventTopics.TOPIC_IMAGE_SELECTED, self._on_image_selected),
            (EventTopics.TOPIC_IMAGE_ADDED, self._on_image_added),
            (EventTopics.TOPIC_COORDINATE_CHANGED, self._on_coord_changed),
            (EventTopics.TOPIC_VIEW_MOUSE_PRESSED, self._on_mouse_pressed),
            (EventTopics.TOPIC_VIEW_MOUSE_RELEASED, self._on_mouse_released),
            (EventTopics.TOPIC_VIEW_MOUSE_DOUBLE_CLICKED, self._on_mouse_double),
            (EventTopics.TOPIC_VIEW_KEY_PRESSED, self._on_key_pressed),
            (EventTopics.TOPIC_VECTOR_SELECTED, self._on_vector_selected),
            (EventTopics.TOPIC_VECTOR_ADDED, self._on_vector_changed),
            (EventTopics.TOPIC_VECTOR_UPDATED, self._on_vector_changed),
            (EventTopics.TOPIC_VECTOR_REMOVED, self._on_vector_removed),
        ]
        for topic, cb in items:
            self.event_bus.subscribe(topic, cb)
            self._subs.append((topic, cb))

    def _unsubscribe(self):
        for topic, cb in self._subs:
            self.event_bus.unsubscribe(topic, cb)
        self._subs.clear()

    def _refresh_workspace(self, activate_default: bool = False):
        self._refresh_base_images()
        if not self._name:
            existing = self._find_existing_collection()
            if existing:
                self._load_collection(existing)
        if activate_default and not self._name and self.panel.get_current_base_image():
            self._on_base_changed(self.panel.get_current_base_image())
        self._refresh_panel()

    def _find_existing_collection(self) -> str:
        for name, vector in self.workspace.get_all_vectors().items():
            if isinstance(vector, dict) and vector.get("type") == "vector_layer_collection" and vector.get("meta", {}).get("module") == "mod5_dlg":
                return name
        return ""
    def _refresh_base_images(self):
        names = sorted(self.workspace.get_all_images().keys())
        self.panel.set_base_images(names)
        if self._base and self._base in names:
            self.panel.base_image_combo.blockSignals(True)
            self.panel.base_image_combo.setCurrentText(self._base)
            self.panel.base_image_combo.blockSignals(False)

    def _base_path(self, name: str) -> str:
        return self.workspace.get_image(name) or ""

    def _collection_name(self, base: str) -> str:
        safe = os.path.splitext(os.path.basename(base or "DLG"))[0]
        return f"DLG_{safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _current_collection(self) -> Optional[Dict[str, Any]]:
        return self.workspace.get_vector(self._name) if self._name else None

    def _current_layer(self) -> Optional[Dict[str, Any]]:
        c = self._current_collection()
        if not c:
            return None
        lid = self._state.get("active_layer_id")
        if lid:
            layer = find_layer(c, lid)
            if layer:
                return layer
        layers = c.get("layers") or []
        return layers[0] if layers else None

    def _active_layer_for_geometry(self, gtype: str) -> Optional[Dict[str, Any]]:
        c = self._ensure_collection()
        if not c:
            return None
        layer = self._current_layer()
        if layer and layer.get("geometry_type") == gtype:
            return layer
        for item in c.get("layers") or []:
            if item.get("geometry_type") == gtype:
                self._state["active_layer_id"] = item.get("layer_id")
                return item
        name = f"{gtype}图层"
        idx = 1
        names = {item.get("layer_name") for item in c.get("layers") or []}
        while name in names:
            idx += 1
            name = f"{gtype}图层{idx}"
        layer = create_layer(name, gtype, schema=default_schema(gtype))
        c.setdefault("layers", []).append(layer)
        self._state["active_layer_id"] = layer["layer_id"]
        self._state["is_dirty"] = True
        self._commit(summary=f"新建图层: {name}")
        return layer

    def _ensure_collection(self, base: Optional[str] = None):
        base = base or self._base or self.panel.get_current_base_image()
        if not base:
            return None
        c = self._current_collection()
        if c:
            return c
        name = self._collection_name(base)
        c = build_vector_collection(name=name, source_image=base, coordinate_mode="pixel")
        c["source_image_path"] = self._base_path(base)
        c["meta"]["source_image_path"] = c["source_image_path"]
        c["meta"]["base_image_name"] = base
        c["meta"]["created_at"] = datetime.now().isoformat(timespec="seconds")
        self.workspace.add_vector(name, c)
        self._name, self._base = name, base
        self._state.update({"active_layer_id": None, "selected_feature_ids": [], "hover_feature_id": None, "draft_vertices": [], "draft_geometry": None, "is_dirty": False})
        self.panel.set_dirty(False)
        self._publish_state(EventTopics.TOPIC_VECTOR_ADDED, c, summarize_vector_collection(c), f"DLG 结果: {name}")
        return c

    def _load_collection(self, name: str):
        c = self.workspace.get_vector(name)
        if not c:
            return
        self._name = name
        self._base = c.get("source_image", "")
        layers = c.get("layers") or []
        self._state["active_layer_id"] = layers[0].get("layer_id") if layers else None
        self._state["selected_feature_ids"] = []
        self._state["hover_feature_id"] = None
        self._state["draft_vertices"] = []
        self._state["draft_geometry"] = None
        self._state["is_dirty"] = False
        self.panel.set_dirty(False)
        self._publish_state(EventTopics.TOPIC_VECTOR_SELECTED, c, summarize_vector_collection(c), f"DLG 结果: {name}")

    def _commit(self, topic: str = EventTopics.TOPIC_VECTOR_UPDATED, summary: str = ""):
        c = self._current_collection()
        if not c:
            return
        c["name"] = self._name
        c.setdefault("meta", {})["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.workspace.update_vector(self._name, c)
        self._state["is_dirty"] = False
        self.panel.set_dirty(False)
        self._publish_state(topic, c, summary or summarize_vector_collection(c), f"DLG 结果: {self._name}")

    def _refresh_panel(self):
        c = self._current_collection()
        self._refresh_base_images()
        if not c:
            self.panel.set_layers([])
            self.panel.set_layer_summary("当前图层: -")
            self.panel.set_feature_editor(None, None, [])
            self.panel.set_status_message("请先导入底图，或新建 DLG 图层集合")
            self.panel.set_dirty(False)
            return
        layers = c.get("layers") or []
        self.panel.set_layers(layers)
        layer = self._current_layer()
        feature = None
        if layer and self._state.get("selected_feature_ids"):
            _, feature = find_feature(c, self._state["selected_feature_ids"][0])
        if layer:
            self.panel.set_layer_summary(f"当前图层: {layer.get('layer_name','-')} | 几何: {layer.get('geometry_type','-')} | 要素: {len(layer.get('features', []))}")
            self.panel.set_feature_editor(layer.get("layer_id"), feature, layer.get("schema", []))
        else:
            self.panel.set_layer_summary("当前图层: -")
            self.panel.set_feature_editor(None, None, [])
        self.panel.set_status_message(summarize_vector_collection(c))
        self.panel.set_dirty(bool(self._state.get("is_dirty")))
        self.panel.set_current_tool(self._state.get("current_tool", "select"))
        self.panel.set_cursor_text(self._cursor_text())

    def _cursor_text(self) -> str:
        x, y = self._state.get("mouse_pos", [0.0, 0.0])
        return f"({int(x)}, {int(y)})" if x or y else "--"

    def _current_layer_name(self) -> str:
        layer = self._current_layer()
        return layer.get("layer_name", "") if layer else ""

    def _publish_mode(self, enabled: bool):
        self.event_bus.publish(
            EventTopics.TOPIC_VECTOR_EDIT_MODE_CHANGED,
            {
                "enabled": bool(enabled),
                "title": "DLG 编辑模式" if enabled else "DLG 浏览模式",
                "current_tool": self._state.get("current_tool", "select"),
                "current_layer_name": self._current_layer_name(),
                "summary": "草稿绘制中" if self._state.get("draft_geometry") else "可开始数字化" if enabled else "",
            },
        )

    def _publish_state(self, topic: str, vector: Optional[Dict[str, Any]], summary: str = "", title: str = "", enabled: Optional[bool] = None):
        feature_count = 0
        if isinstance(vector, dict):
            for layer in vector.get("layers") or []:
                feature_count += len(layer.get("features", []))
        payload = {"name": self._name, "vector": deepcopy(vector) if isinstance(vector, dict) else vector, "summary": summary, "title": title or f"DLG ??: {self._name}", "selected_feature_ids": list(self._state.get("selected_feature_ids") or []), "hover_feature_id": self._state.get("hover_feature_id"), "draft_geometry": deepcopy(self._state.get("draft_geometry")), "active_layer_id": self._state.get("active_layer_id"), "current_tool": self._state.get("current_tool", "select"), "current_layer_name": self._current_layer_name(), "feature_count": feature_count, "origin": "mod5_dlg"}
        if enabled is not None:
            payload["enabled"] = bool(enabled)
        self.event_bus.publish(topic, payload)

    @staticmethod
    def _button_value(button: Any) -> int:
        return int(getattr(button, "value", button))

    def _warn(self, text: str):
        self.panel.set_status_message(text)
        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            log_manager.warning(text)
            return
        QMessageBox.warning(self.panel, "DLG", text)

    def _info(self, text: str):
        self.panel.set_status_message(text)
        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            log_manager.info(text)
            return
        QMessageBox.information(self.panel, "DLG", text)

    def _confirm(self, text: str) -> bool:
        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            return True
        return QMessageBox.question(self.panel, "DLG", text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def _on_image_added(self, data):
        self._refresh_base_images()

    def _on_image_selected(self, data):
        if isinstance(data, dict):
            name = data.get("name") or ""
            path = data.get("path") or self.workspace.get_image(name) or ""
        else:
            name = str(data)
            path = self.workspace.get_image(name) or ""
        if not name:
            return
        self._base = name
        self.panel.base_image_combo.blockSignals(True)
        self.panel.base_image_combo.setCurrentText(name)
        self.panel.base_image_combo.blockSignals(False)
        if path:
            log_manager.info(f"DLG 底图切换: {name} -> {path}")
        self._ensure_collection(name)
        self._publish_mode(True)
        self._refresh_panel()

    def _on_coord_changed(self, data):
        if not isinstance(data, dict):
            return
        self._state["mouse_pos"] = [float(data.get("col", 0.0)), float(data.get("row", 0.0))]
        self.panel.set_cursor_text(self._cursor_text())

    def _on_vector_changed(self, data):
        if not isinstance(data, dict) or data.get("name") != self._name:
            return
        self._refresh_panel()

    def _on_vector_removed(self, data):
        if isinstance(data, dict) and data.get("name") == self._name:
            self._name = ""
            self._base = ""
            self._state.update({"active_layer_id": None, "selected_feature_ids": [], "draft_vertices": [], "draft_geometry": None, "is_dirty": False})
            self._refresh_workspace(True)

    def _on_vector_selected(self, data):
        if not isinstance(data, dict):
            return
        name = data.get("name") or ""
        vector = data.get("vector") or (self.workspace.get_vector(name) if name else None)
        if not vector:
            return
        if name:
            self._name = name
        self._base = vector.get("source_image", self._base)
        self._state["selected_feature_ids"] = list(data.get("selected_feature_ids") or [])
        self._state["hover_feature_id"] = data.get("hover_feature_id")
        self._state["draft_geometry"] = deepcopy(data.get("draft_geometry")) if data.get("draft_geometry") else None
        if data.get("active_layer_id"):
            self._state["active_layer_id"] = data.get("active_layer_id")
        self._refresh_panel()
    def _on_base_changed(self, base: str):
        base = (base or "").strip()
        if not base:
            return
        if base == self._base and self._name:
            self._refresh_panel()
            return
        self._base = base
        self._name = ""
        self._state.update({"active_layer_id": None, "selected_feature_ids": [], "draft_vertices": [], "draft_geometry": None, "is_dirty": False})
        self._ensure_collection(base)
        self._publish_base_selected(base)
        self._refresh_panel()
        log_manager.info(f"DLG 底图已切换: {base}")

    def _publish_base_selected(self, base: str):
        self.event_bus.publish(EventTopics.TOPIC_IMAGE_SELECTED, {"name": base, "path": self._base_path(base), "title": f"DLG 底图: {base}"})

    def _on_tool_changed(self, tool: str):
        self._state["current_tool"] = tool or "select"
        if tool in {"select", "delete"}:
            self._state["draft_vertices"] = []
            self._state["draft_geometry"] = None
        self.panel.set_current_tool(self._state["current_tool"])
        self._publish_mode(True)
        self._publish_current(f"当前工具: {self._state['current_tool']}")
        log_manager.info(f"DLG 工具切换: {self._state['current_tool']}")

    def _publish_current(self, summary: str = ""):
        c = self._current_collection()
        if c:
            self._publish_state(EventTopics.TOPIC_VECTOR_UPDATED, c, summary or summarize_vector_collection(c), f"DLG 结果: {self._name}")

    def _new_layer(self):
        c = self._ensure_collection()
        if not c:
            self._warn("请先选择底图，再新建图层")
            return
        choice, ok = QInputDialog.getItem(self.panel, "新建图层", "几何类型:", ["点 (Point)", "线 (LineString)", "面 (Polygon)"], 0, False)
        if not ok:
            return
        gtype = {"点 (Point)": "Point", "线 (LineString)": "LineString", "面 (Polygon)": "Polygon"}.get(choice, "Point")
        name, ok = QInputDialog.getText(self.panel, "新建图层", "图层名称:")
        if not ok or not name.strip():
            return
        layer = create_layer(name.strip(), gtype, schema=default_schema(gtype))
        c.setdefault("layers", []).append(layer)
        self._state["active_layer_id"] = layer["layer_id"]
        self._state["is_dirty"] = True
        self._commit(summary=f"新建图层: {name.strip()}")
        log_manager.info(f"DLG 新建图层: {name.strip()} / {gtype}")

    def _delete_layer(self):
        c = self._current_collection(); layer = self._current_layer()
        if not c or not layer:
            self._warn("当前没有可删除的图层")
            return
        if not self._confirm(f"确认删除图层「{layer.get('layer_name', '-') }」吗？"):
            return
        c["layers"] = [it for it in c.get("layers", []) if it.get("layer_id") != layer.get("layer_id")]
        self._state.update({"active_layer_id": c["layers"][0]["layer_id"] if c.get("layers") else None, "selected_feature_ids": [], "draft_vertices": [], "draft_geometry": None, "is_dirty": True})
        self._commit(summary=f"删除图层: {layer.get('layer_name', '-')}")
        log_manager.info(f"DLG 删除图层: {layer.get('layer_name', '-')}")

    def _select_layer(self, layer_id: str):
        c = self._current_collection()
        if not c or not layer_id:
            return
        self._state.update({"active_layer_id": layer_id, "selected_feature_ids": [], "draft_vertices": [], "draft_geometry": None})
        self._refresh_panel()
        self._publish_current(f"切换图层: {layer_id}")

    def _toggle_visibility(self, layer_id: str, visible: bool):
        c = self._current_collection(); layer = find_layer(c, layer_id) if c else None
        if not layer:
            return
        layer["visible"] = bool(visible)
        self._state["is_dirty"] = True
        self._commit(summary=f"图层可见性: {layer.get('layer_name', '-')}")

    def _save_feature(self):
        c = self._current_collection(); layer = self._current_layer()
        if not c or not layer:
            self._warn("请先选择图层")
            return
        fid = self._state.get("selected_feature_ids", [None])[0] if self._state.get("selected_feature_ids") else None
        if not fid:
            self._warn("请先选中要素，再编辑属性")
            return
        _, feat = find_feature(c, fid)
        if not feat:
            self._warn("未找到当前要素")
            return
        props = self._normalize_properties(self.panel.read_schema(), self.panel.read_feature_properties())
        feat["properties"] = props
        feat["updated_at"] = datetime.now().isoformat(timespec="seconds")
        update_derived_properties(layer)
        self._state["is_dirty"] = True
        self._commit(summary=f"更新属性: {fid}")

    def _add_field(self):
        layer = self._current_layer()
        if not layer:
            self._warn("请先选择图层")
            return
        name, ok = QInputDialog.getText(self.panel, "新增字段", "字段名称:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if any(f.get("name") == name for f in layer.get("schema", [])):
            self._warn("字段已存在")
            return
        ftype, ok = QInputDialog.getItem(self.panel, "新增字段", "字段类型:", ["string", "float", "bool"], 0, False)
        if not ok:
            return
        layer.setdefault("schema", []).append({"name": name, "type": ftype})
        for feat in layer.get("features", []):
            feat.setdefault("properties", {})[name] = self._default_value(ftype)
        self._state["is_dirty"] = True
        self._commit(summary=f"新增字段: {name}")

    def _rename_field(self):
        layer = self._current_layer(); row = self.panel.schema_table.currentRow()
        if not layer or row < 0:
            self._warn("请先选择图层并在字段表中选中字段")
            return
        old = (self.panel.schema_table.item(row, 0).text() if self.panel.schema_table.item(row, 0) else "").strip()
        new, ok = QInputDialog.getText(self.panel, "重命名字段", f"字段 {old} ->")
        if not ok or not new.strip() or old == new.strip():
            return
        new = new.strip()
        if any(f.get("name") == new for f in layer.get("schema", [])):
            self._warn("新字段名已存在")
            return
        for f in layer.get("schema", []):
            if f.get("name") == old:
                f["name"] = new
        for feat in layer.get("features", []):
            props = feat.setdefault("properties", {})
            props[new] = props.pop(old, self._default_value(self._field_type(layer, new)))
        self._state["is_dirty"] = True
        self._commit(summary=f"重命名字段: {old} -> {new}")

    def _delete_field(self):
        layer = self._current_layer(); row = self.panel.schema_table.currentRow()
        if not layer or row < 0:
            self._warn("请先选择图层并在字段表中选中字段")
            return
        item = self.panel.schema_table.item(row, 0)
        if not item:
            return
        name = item.text().strip()
        if not self._confirm(f"确认删除字段「{name}」吗？"):
            return
        layer["schema"] = [f for f in layer.get("schema", []) if f.get("name") != name]
        for feat in layer.get("features", []):
            feat.setdefault("properties", {}).pop(name, None)
        self._state["is_dirty"] = True
        self._commit(summary=f"删除字段: {name}")

    def _save_result(self):
        c = self._current_collection()
        if not c:
            self._warn("当前没有可保存的 DLG 成果")
            return
        self._state["is_dirty"] = False
        self._commit(summary=summarize_vector_collection(c))
        self._info("DLG 成果已写回 Workspace.vectors")

    def _export(self):
        c = self._current_collection()
        if not c:
            self._warn("当前没有可导出的 DLG 成果")
            return
        cfg = self.panel.get_export_config()
        if (cfg.get("format") or "GeoJSON").lower() != "geojson":
            self._warn("第一版仅支持 GeoJSON 导出")
            return
        out = self._export_geojson(c, cfg.get("path") or "")
        if out:
            c.setdefault("meta", {})["export_path"] = out
            c["meta"]["export_stamp"] = self._stamp()
            self._commit(summary=f"导出完成: {out}")
            self._info(f"GeoJSON 导出成功: {out}")
            log_manager.info(f"DLG 导出成功: {out}")
    def _on_mouse_pressed(self, data):
        if not isinstance(data, dict): return
        x, y = float(data.get('x', 0.0)), float(data.get('y', 0.0))
        btn = self._button_value(data.get('button', 0))
        self._state['mouse_pos'] = [x, y]
        self.panel.set_cursor_text(self._cursor_text())
        if btn != self._button_value(Qt.MouseButton.LeftButton):
            if btn == self._button_value(Qt.MouseButton.RightButton): self._cancel_draft()
            return
        tool = self._state.get('current_tool', 'select')
        if tool == 'point': self._commit_point(x, y)
        elif tool in {'line', 'polygon'}: self._append_vertex(x, y)
        elif tool == 'select': self._select_hit(x, y)
        elif tool == 'delete': self._delete_hit(x, y)

    def _on_mouse_released(self, data):
        if isinstance(data, dict): self._state['mouse_pos'] = [float(data.get('x', 0.0)), float(data.get('y', 0.0))]

    def _on_mouse_double(self, data):
        if not isinstance(data, dict): return
        if self._state.get('current_tool') in {'line', 'polygon'}:
            self._append_vertex(float(data.get('x', 0.0)), float(data.get('y', 0.0)))
            self._finish_draft()

    def _on_key_pressed(self, data):
        key = int(data.get('key', 0)) if isinstance(data, dict) else int(data)
        if key == int(Qt.Key.Key_Escape): self._cancel_draft(); return
        if key in {int(Qt.Key.Key_Backspace), int(Qt.Key.Key_Delete)}:
            if self._state.get('draft_vertices'): self._pop_vertex()
            else: self._delete_selected()
            return
        if key in {int(Qt.Key.Key_Return), int(Qt.Key.Key_Enter)}: self._finish_draft()

    def _commit_point(self, x: float, y: float):
        layer = self._active_layer_for_geometry('Point')
        if not layer:
            self._warn('请先新建点图层')
            return
        feat = create_feature('Point', [x, y], schema=layer.get('schema', []))
        layer.setdefault('features', []).append(feat)
        update_derived_properties(layer)
        self._state.update({'selected_feature_ids': [feat['feature_id']], 'draft_vertices': [], 'draft_geometry': None, 'is_dirty': True})
        self._commit(summary=f"新增点要素: {feat['feature_id']}")
        log_manager.info(f"DLG 新增点要素: {feat['feature_id']} ({x:.1f}, {y:.1f})")

    def _append_vertex(self, x: float, y: float):
        gtype = 'LineString' if self._state.get('current_tool') == 'line' else 'Polygon'
        layer = self._active_layer_for_geometry(gtype)
        if not layer:
            self._warn('请先新建对应几何类型的图层')
            return
        self._state['draft_vertices'].append([float(x), float(y)])
        self._state['draft_geometry'] = self._draft_geometry()
        self._state['active_layer_id'] = layer.get('layer_id')
        self._state['is_dirty'] = True
        self.panel.set_dirty(True)
        self._publish_current(f"正在绘制 {gtype}")

    def _finish_draft(self):
        tool = self._state.get('current_tool')
        vs = self._state.get('draft_vertices') or []
        if tool == 'line' and len(vs) < 2: return self._warn('线要素至少需要 2 个顶点')
        if tool == 'polygon' and len(vs) < 3: return self._warn('面要素至少需要 3 个顶点')
        gtype = 'LineString' if tool == 'line' else 'Polygon'
        layer = self._active_layer_for_geometry(gtype)
        if not layer:
            return self._warn('请先新建对应几何类型的图层')
        feat = create_feature(gtype, vs, schema=layer.get('schema', []))
        layer.setdefault('features', []).append(feat)
        update_derived_properties(layer)
        self._state.update({'selected_feature_ids': [feat['feature_id']], 'draft_vertices': [], 'draft_geometry': None, 'is_dirty': True})
        self._commit(summary=f"新增{gtype}要素: {feat['feature_id']}")
        log_manager.info(f"DLG 新增{gtype}要素: {feat['feature_id']}")

    def _cancel_draft(self):
        if not self._state.get('draft_vertices'): return
        self._state.update({'draft_vertices': [], 'draft_geometry': None, 'is_dirty': False})
        self.panel.set_dirty(False)
        self._publish_current('已取消当前草稿')

    def _pop_vertex(self):
        vs = self._state.get('draft_vertices') or []
        if not vs: return
        vs.pop()
        self._state['draft_geometry'] = self._draft_geometry()
        self._publish_current('撤销最后一个顶点')

    def _select_hit(self, x: float, y: float):
        c = self._current_collection()
        if not c: return
        candidates: List[Tuple[float, str, str]] = []
        layers = [self._current_layer()] if self._current_layer() else []
        layers += [l for l in c.get('layers') or [] if l not in layers]
        for layer in layers:
            if not layer or not layer.get('visible', True): continue
            for feat in layer.get('features', []):
                d = self._feature_distance(feat, x, y)
                if d is not None: candidates.append((d, layer.get('layer_id'), feat.get('feature_id')))
        if not candidates:
            self._state['selected_feature_ids'] = []
            self._refresh_panel(); self._publish_current('未选中要素')
            return
        candidates.sort(key=lambda t: t[0])
        _, lid, fid = candidates[0]
        self._state.update({'active_layer_id': lid, 'selected_feature_ids': [fid]})
        self._refresh_panel(); self._publish_current(f'选中要素: {fid}')

    def _delete_hit(self, x: float, y: float):
        c = self._current_collection()
        if not c: return
        hits: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for layer in c.get('layers') or []:
            if not layer.get('visible', True): continue
            for feat in layer.get('features', []):
                d = self._feature_distance(feat, x, y)
                if d is not None: hits.append((d, layer, feat))
        if not hits:
            self._publish_current('未找到可删除要素')
            return
        hits.sort(key=lambda t: t[0])
        _, layer, feat = hits[0]
        layer['features'] = [f for f in layer.get('features', []) if f.get('feature_id') != feat.get('feature_id')]
        update_derived_properties(layer)
        self._state.update({'selected_feature_ids': [], 'is_dirty': True})
        self._commit(summary=f"删除要素: {feat.get('feature_id')}")

    def _delete_selected(self):
        c = self._current_collection()
        if not c or not self._state.get('selected_feature_ids'): return
        for fid in list(self._state.get('selected_feature_ids') or []):
            layer, feat = find_feature(c, fid)
            if layer and feat:
                layer['features'] = [f for f in layer.get('features', []) if f.get('feature_id') != fid]
                update_derived_properties(layer)
        self._state.update({'selected_feature_ids': [], 'is_dirty': True})
        self._commit(summary='删除选中要素')

    def _draft_geometry(self) -> Optional[Dict[str, Any]]:
        vs = self._state.get('draft_vertices') or []
        tool = self._state.get('current_tool')
        if tool == 'line' and len(vs) >= 2:
            return {'type': 'LineString', 'coordinates': [[float(x), float(y)] for x, y in vs]}
        if tool == 'polygon' and len(vs) >= 2:
            coords = [[float(x), float(y)] for x, y in vs]
            if len(coords) >= 3 and coords[0] != coords[-1]: coords.append(coords[0][:])
            return {'type': 'Polygon', 'coordinates': [coords]}
        return None

    def _feature_distance(self, feat: Dict[str, Any], x: float, y: float, tol: float = 8.0) -> Optional[float]:
        g = feat.get('geometry') or {}; t = (g.get('type') or '').lower(); c = g.get('coordinates') or []
        if t == 'point' and len(c) >= 2:
            d = math.hypot(float(c[0]) - x, float(c[1]) - y)
            return d if d <= tol else None
        if t == 'linestring' and len(c) >= 2:
            d = min(self._dist_seg(x, y, float(a[0]), float(a[1]), float(b[0]), float(b[1])) for a, b in zip(c, c[1:]))
            return d if d <= tol else None
        if t == 'polygon':
            ring = c[0] if c and isinstance(c[0], list) and c and isinstance(c[0][0], list) else c
            if len(ring) < 3: return None
            if self._in_poly(x, y, ring): return 0.0
            d = min(self._dist_seg(x, y, float(a[0]), float(a[1]), float(b[0]), float(b[1])) for a, b in zip(ring, ring[1:]))
            return d if d <= tol else None
        return None

    @staticmethod
    def _dist_seg(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0: return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / float(dx * dx + dy * dy)))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    @staticmethod
    def _in_poly(px, py, ring):
        inside = False
        for i in range(len(ring) - 1):
            x1, y1 = float(ring[i][0]), float(ring[i][1]); x2, y2 = float(ring[i + 1][0]), float(ring[i + 1][1])
            if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / ((y2 - y1) or 1e-9) + x1): inside = not inside
        return inside

    def _normalize_properties(self, schema: Sequence[Dict[str, Any]], values: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in schema or []:
            name = str(f.get('name', '')).strip()
            if name: out[name] = self._parse_value(values.get(name), str(f.get('type', 'string')))
        return out

    def _parse_value(self, raw: Any, ftype: str) -> Any:
        text = '' if raw is None else str(raw).strip(); ftype = (ftype or 'string').lower()
        if ftype == 'float':
            try: return float(text) if text != '' else 0.0
            except Exception: return 0.0
        if ftype == 'bool': return text.lower() in {'1', 'true', 'yes', 'y', 't', '是'}
        return text

    def _default_value(self, ftype: str) -> Any:
        return 0.0 if (ftype or '').lower() == 'float' else False if (ftype or '').lower() == 'bool' else ''

    def _field_type(self, layer: Dict[str, Any], field_name: str) -> str:
        for f in layer.get('schema', []):
            if f.get('name') == field_name: return str(f.get('type', 'string'))
        return 'string'

    def _export_geojson(self, c: Dict[str, Any], path: str):
        if not c.get('layers'):
            return self._warn('当前没有可导出的图层')
        c.setdefault('meta', {})['export_stamp'] = self._stamp()
        if not path:
            path = os.path.join(os.getcwd(), 'output', 'mod5_dlg')
        if os.path.isdir(path) or not os.path.splitext(path)[1]:
            c['meta']['export_path'] = path
            return ';'.join(export_geojson_all_layers(c, path))
        layer = self._current_layer() or (c.get('layers') or [None])[0]
        if not layer: return None
        out = export_geojson_layer(c, layer.get('layer_id'), path)
        c['meta']['export_path'] = out
        return out

    def _stamp(self):
        return datetime.now().strftime('%Y%m%d_%H%M%S')
