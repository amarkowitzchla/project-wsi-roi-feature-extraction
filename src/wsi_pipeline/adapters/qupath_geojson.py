from __future__ import annotations

import json
from pathlib import Path

from wsi_pipeline.adapters.generic_geojson_v1 import RoiRecord, parse_roi_geojson


class QuPathGeoJsonError(ValueError):
    pass


def qupath_to_geojson_v1(path: Path, slide_id: str) -> Path:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise QuPathGeoJsonError("Expected FeatureCollection from QuPath")

    features = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        classification = props.get("classification") or {}
        label = classification.get("name") or props.get("label") or props.get("name")
        if not label:
            raise QuPathGeoJsonError("Missing label/classification in QuPath feature")

        annotation_id = props.get("id") or props.get("objectId") or props.get("annotation_id")
        if not annotation_id:
            annotation_id = f"qupath_{len(features)}"

        props_v1 = {
            "annotation_id": str(annotation_id),
            "label": str(label),
            "annotator": props.get("creator"),
            "tool": "qupath",
            "created_at": props.get("created"),
            "notes": props.get("notes"),
            "confidence": props.get("confidence"),
        }

        features.append({
            "type": "Feature",
            "geometry": feature.get("geometry"),
            "properties": props_v1,
        })

    out = {
        "type": "FeatureCollection",
        "features": features,
    }
    out_path = path.with_suffix(".v1.geojson")
    out_path.write_text(json.dumps(out), encoding="utf-8")
    return out_path


def parse_qupath_geojson(path: Path, slide_id: str) -> list[RoiRecord]:
    v1_path = qupath_to_geojson_v1(path, slide_id)
    return parse_roi_geojson(v1_path, slide_id)
