from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from wsi_pipeline.utils.common import Manifest, ensure_dir, read_csv, sha256_file, utc_now, write_csv, write_json


SUPPORTED_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn"}


def find_slides(paths: Iterable[Path]) -> list[Path]:
    slides: list[Path] = []
    for base in paths:
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                slides.append(p)
    return sorted(slides)


def slide_id_from_path(path: Path) -> str:
    return path.stem


def scan_slides(
    slide_paths: list[Path],
    output_csv: Path,
    output_manifest: Path,
    compute_sha256: bool,
) -> None:
    openslide = None
    openslide_error = None
    try:
        import openslide as _openslide
        openslide = _openslide
    except Exception as exc:
        openslide_error = exc

    rows = []
    seen_ids: dict[str, int] = {}

    for slide_path in slide_paths:
        slide_id = slide_id_from_path(slide_path)
        if slide_id in seen_ids:
            seen_ids[slide_id] += 1
            slide_id = f"{slide_id}_{seen_ids[slide_id]}"
        else:
            seen_ids[slide_id] = 1

        size_bytes = slide_path.stat().st_size
        sha256 = sha256_file(slide_path) if compute_sha256 else ""

        vendor = ""
        mpp_x = ""
        mpp_y = ""
        width = ""
        height = ""
        if openslide is not None:
            slide = openslide.open_slide(str(slide_path))
            vendor = slide.properties.get("openslide.vendor", "")
            mpp_x = slide.properties.get("openslide.mpp-x", "")
            mpp_y = slide.properties.get("openslide.mpp-y", "")
            width, height = slide.dimensions

        row = {
            "slide_id": slide_id,
            "slide_path": str(slide_path),
            "filename": slide_path.name,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "vendor": vendor,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "width": width,
            "height": height,
            "openslide_vendor": vendor,
            "created_at": utc_now(),
        }
        rows.append(row)

    fieldnames = [
        "slide_id",
        "slide_path",
        "filename",
        "size_bytes",
        "sha256",
        "vendor",
        "mpp_x",
        "mpp_y",
        "width",
        "height",
        "openslide_vendor",
        "created_at",
    ]

    write_csv(output_csv, rows, fieldnames)

    manifest = Manifest(
        name="slides_manifest",
        created_at=utc_now(),
        version="1.0",
        params={
            "compute_sha256": compute_sha256,
            "openslide_available": openslide is not None,
            "openslide_error": str(openslide_error) if openslide_error else "",
        },
        inputs={"slide_count": len(slide_paths)},
        outputs={"slides_manifest_csv": str(output_csv)},
    )
    write_json(output_manifest, manifest.to_dict())


def load_slides_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv(path)
