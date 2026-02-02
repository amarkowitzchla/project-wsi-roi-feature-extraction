from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shapely.geometry import shape

from wsi_pipeline.utils.common import utc_now


class RoiGeoJsonError(ValueError):
    pass


@dataclass
class RoiRecord:
    slide_id: str
    annotation_id: str
    label: str
    geometry_wkt: str
    bbox_x0: float
    bbox_y0: float
    bbox_x1: float
    bbox_y1: float
    area: float
    annotator: str | None
    tool: str | None
    created_at: str | None
    notes: str | None
    confidence: float | None


REQUIRED_PROPS = {"annotation_id", "label"}


def parse_roi_geojson(path: Path, slide_id: str) -> list[RoiRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise RoiGeoJsonError(f"Not a FeatureCollection: {path}")

    records: list[RoiRecord] = []
    for feature in data.get("features", []):
        if feature.get("type") != "Feature":
            raise RoiGeoJsonError("Invalid feature in GeoJSON")
        geometry = feature.get("geometry")
        if not geometry:
            raise RoiGeoJsonError("Feature missing geometry")
        geom = shape(geometry)
        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            raise RoiGeoJsonError(f"Unsupported geometry type: {geom.geom_type}")

        props = feature.get("properties", {})
        missing = REQUIRED_PROPS - set(props.keys())
        if missing:
            raise RoiGeoJsonError(f"Missing properties: {', '.join(sorted(missing))}")

        annotation_id = str(props.get("annotation_id"))
        label = str(props.get("label"))

        minx, miny, maxx, maxy = geom.bounds
        record = RoiRecord(
            slide_id=slide_id,
            annotation_id=annotation_id,
            label=label,
            geometry_wkt=geom.wkt,
            bbox_x0=float(minx),
            bbox_y0=float(miny),
            bbox_x1=float(maxx),
            bbox_y1=float(maxy),
            area=float(geom.area),
            annotator=props.get("annotator"),
            tool=props.get("tool"),
            created_at=props.get("created_at"),
            notes=props.get("notes"),
            confidence=props.get("confidence"),
        )
        records.append(record)

    return records
