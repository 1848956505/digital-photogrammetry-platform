"""模块五 DLG 矢量数据模型。"""
from __future__ import annotations

import math
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ALLOWED_GEOMETRY_TYPES = {"Point", "LineString", "Polygon"}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_id() -> str:
    return str(uuid.uuid4())


def default_value(field_type: str):
    field_type = (field_type or "string").lower()
    if field_type == "float":
        return 0.0
    if field_type == "bool":
        return False
    return ""


def default_schema(geometry_type: str) -> List[Dict[str, str]]:
    geometry_type = geometry_type or "Point"
    if geometry_type == "Point":
        return [
            {"name": "名称", "type": "string"},
            {"name": "类别", "type": "string"},
        ]
    if geometry_type == "LineString":
        return [
            {"name": "名称", "type": "string"},
            {"name": "类别", "type": "string"},
            {"name": "长度", "type": "float"},
        ]
    return [
        {"name": "名称", "type": "string"},
        {"name": "类别", "type": "string"},
        {"name": "面积", "type": "float"},
    ]


def normalize_geometry(geometry_type: str, coordinates: Sequence) -> Dict[str, Any]:
    if geometry_type not in ALLOWED_GEOMETRY_TYPES:
        raise ValueError(f"不支持的几何类型: {geometry_type}")

    if geometry_type == "Point":
        if len(coordinates) < 2:
            raise ValueError("点几何至少需要两个坐标值")
        return {"type": "Point", "coordinates": [float(coordinates[0]), float(coordinates[1])]}

    if geometry_type == "LineString":
        coords = [[float(x), float(y)] for x, y in coordinates]
        if len(coords) < 2:
            raise ValueError("线几何至少需要两个顶点")
        return {"type": "LineString", "coordinates": coords}

    coords = [[float(x), float(y)] for x, y in coordinates]
    if len(coords) < 3:
        raise ValueError("面几何至少需要三个顶点")
    if coords[0] != coords[-1]:
        coords.append(coords[0][:])
    return {"type": "Polygon", "coordinates": [coords]}


def polyline_length(coordinates: Sequence[Sequence[float]]) -> float:
    total = 0.0
    points = list(coordinates or [])
    for prev, cur in zip(points, points[1:]):
        dx = float(cur[0]) - float(prev[0])
        dy = float(cur[1]) - float(prev[1])
        total += math.hypot(dx, dy)
    return float(total)


def polygon_area(coordinates: Sequence[Sequence[float]]) -> float:
    ring = list(coordinates or [])
    if len(ring) < 3:
        return 0.0
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    area = 0.0
    for p1, p2 in zip(ring, ring[1:]):
        area += float(p1[0]) * float(p2[1]) - float(p2[0]) * float(p1[1])
    return abs(area) * 0.5


def create_feature(
    geometry_type: str,
    coordinates: Sequence,
    schema: Optional[List[Dict[str, str]]] = None,
    properties: Optional[Dict[str, Any]] = None,
    style: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    geometry = normalize_geometry(geometry_type, coordinates)
    schema = deepcopy(schema or default_schema(geometry_type))
    props = {field["name"]: default_value(field.get("type", "string")) for field in schema}
    if properties:
        props.update(properties)

    if geometry_type == "LineString":
        length_value = polyline_length(geometry["coordinates"])
        if "长度" in props and (not props.get("长度")):
            props["长度"] = length_value
    elif geometry_type == "Polygon":
        ring = geometry["coordinates"][0]
        area_value = polygon_area(ring)
        if "面积" in props and (not props.get("面积")):
            props["面积"] = area_value

    return {
        "feature_id": _new_id(),
        "geometry": geometry,
        "properties": props,
        "style": style or {
            "stroke_width": 2,
            "stroke_color": "#1976d2",
            "fill_color": "#64b5f6",
            "fill_opacity": 70,
            "label_visible": False,
        },
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def create_layer(
    layer_name: str,
    geometry_type: str,
    schema: Optional[List[Dict[str, str]]] = None,
    visible: bool = True,
    editable: bool = True,
    style_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "layer_id": _new_id(),
        "layer_name": layer_name,
        "geometry_type": geometry_type,
        "visible": visible,
        "editable": editable,
        "schema": deepcopy(schema or default_schema(geometry_type)),
        "features": [],
        "style_profile": style_profile or {},
    }


def update_derived_properties(layer: Dict[str, Any]) -> Dict[str, Any]:
    geometry_type = layer.get("geometry_type")
    schema = layer.get("schema") or []
    derived_fields = {field["name"] for field in schema}
    for feature in layer.get("features", []):
        geometry = feature.get("geometry") or {}
        props = feature.setdefault("properties", {})
        if geometry_type == "LineString" and "长度" in derived_fields:
            props["长度"] = round(polyline_length(geometry.get("coordinates") or []), 3)
        elif geometry_type == "Polygon" and "面积" in derived_fields:
            ring = (geometry.get("coordinates") or [[]])[0]
            props["面积"] = round(polygon_area(ring), 3)
        feature["updated_at"] = _now_iso()
    return layer


def build_vector_collection(
    name: str,
    source_image: str,
    layers: Optional[List[Dict[str, Any]]] = None,
    coordinate_mode: str = "pixel",
    export_path: str = "",
    snapshot_path: str = "",
    module: str = "mod5_dlg",
    version: str = "1.0",
) -> Dict[str, Any]:
    return {
        "name": name,
        "type": "vector_layer_collection",
        "source_image": source_image,
        "coordinate_mode": coordinate_mode,
        "layers": deepcopy(layers or []),
        "meta": {
            "module": module,
            "version": version,
            "export_path": export_path,
            "snapshot_path": snapshot_path,
        },
    }


def summarize_vector_collection(collection: Dict[str, Any]) -> str:
    layers = collection.get("layers") or []
    layer_count = len(layers)
    feature_count = sum(len(layer.get("features", [])) for layer in layers)
    visible_count = sum(1 for layer in layers if layer.get("visible", True))
    geometry_types = sorted({layer.get("geometry_type", "-") for layer in layers})
    geom_text = "、".join(geometry_types) if geometry_types else "-"
    return f"{layer_count} 图层 / {feature_count} 要素 / {visible_count} 可见 / {geom_text}"


def find_layer(collection: Dict[str, Any], layer_id: str) -> Optional[Dict[str, Any]]:
    for layer in collection.get("layers") or []:
        if layer.get("layer_id") == layer_id:
            return layer
    return None


def find_feature(collection: Dict[str, Any], feature_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    for layer in collection.get("layers") or []:
        for feature in layer.get("features", []):
            if feature.get("feature_id") == feature_id:
                return layer, feature
    return None, None
