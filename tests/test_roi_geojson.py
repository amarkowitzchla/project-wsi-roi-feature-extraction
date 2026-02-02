from pathlib import Path
import json

from wsi_pipeline.adapters.generic_geojson_v1 import parse_roi_geojson


def test_parse_roi_geojson(tmp_path: Path):
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                },
                "properties": {
                    "annotation_id": "a1",
                    "label": "tumor",
                    "annotator": "tester",
                    "tool": "generic",
                },
            }
        ],
    }
    path = tmp_path / "slide1.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")

    records = parse_roi_geojson(path, "slide1")
    assert len(records) == 1
    rec = records[0]
    assert rec.slide_id == "slide1"
    assert rec.label == "tumor"
    assert rec.annotation_id == "a1"
