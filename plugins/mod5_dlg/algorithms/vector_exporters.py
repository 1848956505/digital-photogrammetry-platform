"""模块五 DLG 矢量导出器。"""
from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from typing import Any, Dict, List

from core.log_manager import log_manager


def _ensure_geojson_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    if ext.lower() not in {".geojson", ".json"}:
        return root + ".geojson"
    return path


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text or "").strip())
    return cleaned.strip("._-") or "DLG"


def _feature_to_geojson(feature: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "id": feature.get("feature_id"),
        "geometry": deepcopy(feature.get("geometry")),
        "properties": deepcopy(feature.get("properties") or {}),
    }


def export_geojson_layer(collection: Dict[str, Any], layer_id: str, path: str) -> str:
    """将单个图层导出为 GeoJSON FeatureCollection。"""
    path = _ensure_geojson_path(path)
    layer = None
    for candidate in collection.get("layers") or []:
        if candidate.get("layer_id") == layer_id:
            layer = candidate
            break
    if layer is None:
        raise ValueError("未找到指定图层")

    export_dir = os.path.dirname(path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    payload = {
        "type": "FeatureCollection",
        "name": layer.get("layer_name", "DLG"),
        "features": [_feature_to_geojson(feature) for feature in layer.get("features", [])],
        "properties": {
            "collection_name": collection.get("name", ""),
            "source_image": collection.get("source_image", ""),
            "coordinate_mode": collection.get("coordinate_mode", "pixel"),
            "layer_id": layer_id,
            "geometry_type": layer.get("geometry_type", ""),
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log_manager.info(f"GeoJSON 已导出: {path}")
    return path


def export_geojson_all_layers(collection: Dict[str, Any], directory: str) -> List[str]:
    """按图层导出多个 GeoJSON 文件。"""
    export_dir = directory
    os.makedirs(export_dir, exist_ok=True)
    exported = []
    stamp = str(collection.get("meta", {}).get("export_stamp") or "").strip()
    if not stamp:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for layer in collection.get("layers") or []:
        layer_name = _safe_name(layer.get("layer_name", "DLG"))
        filename = f"{layer_name}_{stamp}.geojson"
        path = os.path.join(export_dir, filename)
        exported.append(export_geojson_layer(collection, layer.get("layer_id"), path))
    return exported


def export_shapefile(*args, **kwargs):
    raise NotImplementedError("第一版暂不提供 Shapefile 导出")


def export_kml(*args, **kwargs):
    raise NotImplementedError("第一版暂不提供 KML 导出")


def export_dxf(*args, **kwargs):
    raise NotImplementedError("第一版暂不提供 DXF 导出")
