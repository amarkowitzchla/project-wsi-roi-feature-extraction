from __future__ import annotations

from pathlib import Path

from wsi_pipeline.adapters.generic_geojson_v1 import RoiRecord


class SectraAdapterError(NotImplementedError):
    pass


def parse_sectra_export(path: Path, slide_id: str) -> list[RoiRecord]:
    """
    Placeholder for Sectra export parsing.

    TODO:
    - Determine the export format (JSON/XML) from Sectra.
    - Parse polygon coordinates and convert to slide pixel coordinates at level 0.
    - Map labels to ROI GeoJSON v1 properties.
    - Return canonical RoiRecord list.
    """
    raise SectraAdapterError(
        "Sectra adapter is a scaffold. Provide an export sample and implement parsing."
    )
