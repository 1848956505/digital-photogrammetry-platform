"""模块五 DLG 右侧控制面板。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)


class DlgControlPanel(QWidget):
    """DLG 数字化控制面板。"""

    refresh_requested = Signal()
    base_image_changed = Signal(str)
    tool_changed = Signal(str)
    new_layer_requested = Signal()
    delete_layer_requested = Signal()
    layer_selected = Signal(str)
    layer_visibility_changed = Signal(str, bool)
    save_feature_requested = Signal()
    add_field_requested = Signal()
    rename_field_requested = Signal()
    delete_field_requested = Signal()
    save_result_requested = Signal()
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_layer_id: Optional[str] = None
        self._current_feature_id: Optional[str] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QLabel("DLG 数字化")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976d2;")
        layout.addWidget(title)

        intro = QLabel(
            "基于底图进行点、线、面数字化，结果写回 Workspace.vectors，"
            "导出坐标采用像素坐标。"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #555; padding: 4px 6px; background: #f6f8fb; border-radius: 4px;")
        layout.addWidget(intro)

        base_group = QGroupBox("底图")
        base_layout = QFormLayout(base_group)
        self.base_image_combo = QComboBox()
        self.base_image_combo.currentTextChanged.connect(self.base_image_changed.emit)
        self.refresh_base_btn = QPushButton("刷新底图")
        self.refresh_base_btn.clicked.connect(self.refresh_requested.emit)
        row = QHBoxLayout()
        row.addWidget(self.base_image_combo, 1)
        row.addWidget(self.refresh_base_btn)
        row_widget = QWidget()
        row_widget.setLayout(row)
        base_layout.addRow("当前底图", row_widget)
        layout.addWidget(base_group)

        tool_group = QGroupBox("工具")
        tool_layout = QVBoxLayout(tool_group)
        self.tool_buttons = QButtonGroup(self)
        self.tool_buttons.setExclusive(True)
        for label, tool in (("选择", "select"), ("点", "point"), ("线", "line"), ("面", "polygon"), ("删除", "delete")):
            btn = QPushButton(label)
            btn.setCheckable(True)
            if tool == "select":
                btn.setChecked(True)
            self.tool_buttons.addButton(btn)
            self.tool_buttons.setId(btn, len(self.tool_buttons.buttons()))
            btn.clicked.connect(lambda checked=False, t=tool: self.tool_changed.emit(t))
            tool_layout.addWidget(btn)
        layout.addWidget(tool_group)

        layer_group = QGroupBox("图层")
        layer_layout = QVBoxLayout(layer_group)
        self.layer_tree = QTreeWidget()
        self.layer_tree.setColumnCount(4)
        self.layer_tree.setHeaderLabels(["图层", "几何", "可见", "数量"])
        self.layer_tree.itemSelectionChanged.connect(self._on_layer_selection_changed)
        self.layer_tree.itemChanged.connect(self._on_layer_item_changed)
        layer_layout.addWidget(self.layer_tree)

        layer_btn_row = QHBoxLayout()
        self.new_layer_btn = QPushButton("新建图层")
        self.delete_layer_btn = QPushButton("删除图层")
        self.new_layer_btn.clicked.connect(self.new_layer_requested.emit)
        self.delete_layer_btn.clicked.connect(self.delete_layer_requested.emit)
        layer_btn_row.addWidget(self.new_layer_btn)
        layer_btn_row.addWidget(self.delete_layer_btn)
        layer_layout.addLayout(layer_btn_row)

        self.layer_summary_label = QLabel("当前图层: -")
        self.layer_summary_label.setWordWrap(True)
        layer_layout.addWidget(self.layer_summary_label)
        layout.addWidget(layer_group)

        feature_group = QGroupBox("属性")
        feature_layout = QVBoxLayout(feature_group)
        self.feature_info_label = QLabel("当前要素: -")
        self.feature_info_label.setWordWrap(True)
        feature_layout.addWidget(self.feature_info_label)

        self.feature_table = QTableWidget(0, 2)
        self.feature_table.setHorizontalHeaderLabels(["字段", "值"])
        feature_layout.addWidget(self.feature_table)

        feature_btn_row = QHBoxLayout()
        self.save_feature_btn = QPushButton("保存属性")
        self.add_field_btn = QPushButton("新增字段")
        self.rename_field_btn = QPushButton("重命名字段")
        self.delete_field_btn = QPushButton("删除字段")
        self.save_feature_btn.clicked.connect(self.save_feature_requested.emit)
        self.add_field_btn.clicked.connect(self.add_field_requested.emit)
        self.rename_field_btn.clicked.connect(self.rename_field_requested.emit)
        self.delete_field_btn.clicked.connect(self.delete_field_requested.emit)
        feature_btn_row.addWidget(self.save_feature_btn)
        feature_btn_row.addWidget(self.add_field_btn)
        feature_btn_row.addWidget(self.rename_field_btn)
        feature_btn_row.addWidget(self.delete_field_btn)
        feature_layout.addLayout(feature_btn_row)

        self.schema_table = QTableWidget(0, 2)
        self.schema_table.setHorizontalHeaderLabels(["字段名", "类型"])
        feature_layout.addWidget(self.schema_table)
        layout.addWidget(feature_group)

        export_group = QGroupBox("导出")
        export_layout = QFormLayout(export_group)
        self.export_format = QComboBox()
        self.export_format.addItems(["GeoJSON", "Shapefile", "KML", "DXF"])
        self.export_path_edit = QLineEdit()
        self.export_path_edit.setPlaceholderText("留空则自动保存到 output/ 目录")
        self.export_browse_btn = QPushButton("浏览...")
        self.export_browse_btn.clicked.connect(self._browse_export_path)
        export_path_row = QHBoxLayout()
        export_path_row.addWidget(self.export_path_edit, 1)
        export_path_row.addWidget(self.export_browse_btn)
        export_path_widget = QWidget()
        export_path_widget.setLayout(export_path_row)
        export_layout.addRow("导出格式", self.export_format)
        export_layout.addRow("导出路径", export_path_widget)
        self.export_hint = QLabel("当前版本导出坐标为像素坐标，不含地理参考。")
        self.export_hint.setWordWrap(True)
        self.export_hint.setStyleSheet("color:#b26a00;")
        export_layout.addRow("", self.export_hint)
        self.save_result_btn = QPushButton("保存成果")
        self.save_result_btn.clicked.connect(self.save_result_requested.emit)
        export_layout.addRow("", self.save_result_btn)
        self.export_btn = QPushButton("导出矢量")
        self.export_btn.clicked.connect(self.export_requested.emit)
        export_layout.addRow("", self.export_btn)
        layout.addWidget(export_group)

        status_group = QGroupBox("状态")
        status_layout = QVBoxLayout(status_group)
        self.tool_status = QLabel("当前工具: 选择")
        self.tool_status.setWordWrap(True)
        self.dirty_status = QLabel("未保存修改: 否")
        self.layer_count_status = QLabel("图层: 0")
        self.feature_count_status = QLabel("要素: 0")
        self.cursor_status = QLabel("坐标: --")
        for w in (self.tool_status, self.dirty_status, self.layer_count_status, self.feature_count_status, self.cursor_status):
            status_layout.addWidget(w)
        layout.addWidget(status_group)

        layout.addStretch()

    def _browse_export_path(self):
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(
            self,
            "选择导出路径",
            self.export_path_edit.text().strip() or "",
            "GeoJSON (*.geojson);;Shapefile (*.shp);;KML (*.kml);;DXF (*.dxf)",
        )
        if path:
            self.export_path_edit.setText(path)

    def _on_layer_selection_changed(self):
        items = self.layer_tree.selectedItems()
        if not items:
            self._current_layer_id = None
            return
        layer_id = items[0].data(0, Qt.ItemDataRole.UserRole)
        self._current_layer_id = layer_id
        self.layer_selected.emit(layer_id)

    def _on_layer_item_changed(self, item: QTreeWidgetItem, column: int):
        if column != 2:
            return
        layer_id = item.data(0, Qt.ItemDataRole.UserRole)
        visible = item.checkState(2) == Qt.CheckState.Checked
        if layer_id:
            self.layer_visibility_changed.emit(layer_id, visible)

    def set_base_images(self, names: List[str]):
        current = self.base_image_combo.currentText()
        self.base_image_combo.blockSignals(True)
        self.base_image_combo.clear()
        self.base_image_combo.addItems(names)
        if current in names:
            self.base_image_combo.setCurrentText(current)
        self.base_image_combo.blockSignals(False)
        if names and not self.base_image_combo.currentText():
            self.base_image_combo.setCurrentIndex(0)

    def set_layers(self, layers: List[Dict[str, Any]]):
        current_layer = self._current_layer_id
        self.layer_tree.blockSignals(True)
        self.layer_tree.clear()
        total_features = 0
        for layer in layers:
            item = QTreeWidgetItem([
                layer.get("layer_name", "-"),
                layer.get("geometry_type", "-"),
                "",
                str(len(layer.get("features", []))),
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, layer.get("layer_id"))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(2, Qt.CheckState.Checked if layer.get("visible", True) else Qt.CheckState.Unchecked)
            self.layer_tree.addTopLevelItem(item)
            total_features += len(layer.get("features", []))
            if current_layer and layer.get("layer_id") == current_layer:
                self.layer_tree.setCurrentItem(item)
        self.layer_tree.blockSignals(False)
        self.layer_count_status.setText(f"图层: {len(layers)}")
        self.feature_count_status.setText(f"要素: {total_features}")

    def set_layer_summary(self, text: str):
        self.layer_summary_label.setText(text or "当前图层: -")

    def set_feature_editor(self, layer_id: Optional[str], feature: Optional[Dict[str, Any]], schema: Optional[List[Dict[str, Any]]]):
        self._current_layer_id = layer_id
        self._current_feature_id = feature.get("feature_id") if isinstance(feature, dict) else None
        self.feature_info_label.setText(
            f"当前要素: {self._current_feature_id or '-'}"
        )

        schema = schema or []
        self.schema_table.blockSignals(True)
        self.schema_table.setRowCount(len(schema))
        for row, field in enumerate(schema):
            self.schema_table.setItem(row, 0, QTableWidgetItem(str(field.get("name", ""))))
            self.schema_table.setItem(row, 1, QTableWidgetItem(str(field.get("type", "string"))))
        self.schema_table.blockSignals(False)

        properties = feature.get("properties", {}) if isinstance(feature, dict) else {}
        self.feature_table.blockSignals(True)
        self.feature_table.setRowCount(len(schema))
        for row, field in enumerate(schema):
            name = field.get("name", "")
            item_name = QTableWidgetItem(str(name))
            item_name.setFlags(item_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.feature_table.setItem(row, 0, item_name)
            self.feature_table.setItem(row, 1, QTableWidgetItem(str(properties.get(name, ""))))
        self.feature_table.blockSignals(False)

    def read_feature_properties(self) -> Dict[str, Any]:
        props: Dict[str, Any] = {}
        for row in range(self.feature_table.rowCount()):
            field_item = self.feature_table.item(row, 0)
            value_item = self.feature_table.item(row, 1)
            if not field_item:
                continue
            props[field_item.text().strip()] = value_item.text().strip() if value_item else ""
        return props

    def read_schema(self) -> List[Dict[str, Any]]:
        schema: List[Dict[str, Any]] = []
        for row in range(self.schema_table.rowCount()):
            name_item = self.schema_table.item(row, 0)
            type_item = self.schema_table.item(row, 1)
            if not name_item:
                continue
            schema.append({
                "name": name_item.text().strip(),
                "type": (type_item.text().strip() if type_item else "string") or "string",
            })
        return schema

    def set_current_tool(self, tool: str):
        self.tool_status.setText(f"当前工具: {tool or '选择'}")

    def set_cursor_text(self, text: str):
        self.cursor_status.setText(f"坐标: {text or '--'}")

    def set_dirty(self, dirty: bool):
        self.dirty_status.setText(f"未保存修改: {'是' if dirty else '否'}")

    def set_status_message(self, text: str):
        self.feature_info_label.setText(text or "当前要素: -")

    def get_current_base_image(self) -> str:
        return self.base_image_combo.currentText().strip()

    def get_current_layer_id(self) -> Optional[str]:
        return self._current_layer_id

    def get_current_feature_id(self) -> Optional[str]:
        return self._current_feature_id

    def get_export_config(self) -> Dict[str, str]:
        return {
            "format": self.export_format.currentText(),
            "path": self.export_path_edit.text().strip(),
        }
