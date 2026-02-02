from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import wkt
from shapely.geometry import mapping

from wsi_pipeline.adapters.generic_geojson_v1 import RoiRecord, parse_roi_geojson
from wsi_pipeline.adapters.qupath_geojson import parse_qupath_geojson
from wsi_pipeline.adapters.sectra_adapter import parse_sectra_export
from wsi_pipeline.core.taxonomy import validate_labels
from wsi_pipeline.utils.common import Manifest, ensure_dir, utc_now, write_json


ADAPTERS = {
    "generic_geojson_v1": parse_roi_geojson,
    "qupath_geojson": parse_qupath_geojson,
    "sectra": parse_sectra_export,
}


def build_rois(
    slide_ids: Iterable[str],
    input_dir: Path,
    output_parquet: Path,
    output_geojson: Path | None,
    tool: str,
    taxonomy: dict,
    allow_unknown_labels: bool,
    manifest_path: Path,
) -> None:
    if tool not in ADAPTERS:
        raise ValueError(f"Unknown adapter: {tool}")

    records: list[RoiRecord] = []
    for slide_id in slide_ids:
        geojson_path = input_dir / f"{slide_id}.geojson"
        if not geojson_path.exists():
            continue
        parsed = ADAPTERS[tool](geojson_path, slide_id)
        records.extend(parsed)

    validate_labels([r.label for r in records], taxonomy, allow_unknown_labels)

    rows = [r.__dict__ for r in records]
    ensure_dir(output_parquet.parent)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_parquet)

    if output_geojson:
        features = []
        for r in records:
            geom = wkt.loads(r.geometry_wkt)
            props = {
                "annotation_id": r.annotation_id,
                "label": r.label,
                "annotator": r.annotator,
                "tool": r.tool,
                "created_at": r.created_at,
                "notes": r.notes,
                "confidence": r.confidence,
            }
            features.append({"type": "Feature", "geometry": mapping(geom), "properties": props})

        out = {"type": "FeatureCollection", "features": features}
        ensure_dir(output_geojson.parent)
        output_geojson.write_text(json.dumps(out), encoding="utf-8")

    manifest = Manifest(
        name="rois_manifest",
        created_at=utc_now(),
        version="1.0",
        params={"tool": tool, "allow_unknown_labels": allow_unknown_labels},
        inputs={"input_dir": str(input_dir), "slide_ids": list(slide_ids)},
        outputs={"rois_parquet": str(output_parquet)},
    )
    write_json(manifest_path, manifest.to_dict())
