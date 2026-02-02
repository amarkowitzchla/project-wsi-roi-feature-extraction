from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from shapely import wkt
from shapely.geometry import box
from shapely.strtree import STRtree

from wsi_pipeline.utils.common import Manifest, ensure_dir, utc_now, write_json
from wsi_pipeline.core.tiling_math import assign_label, tissue_fraction


@dataclass
class TileRecord:
    slide_id: str
    tile_id: str
    dzi_level: int
    tile_col: int
    tile_row: int
    x0: float
    y0: float
    w: float
    h: float
    label: str | None
    roi_ids: list[str]
    roi_overlap_frac: float
    tissue_frac: float
    qc_flags: list[str]
    tile_png_path: str | None
    split: str | None


def build_tiles(
    slide_id: str,
    slide_path: Path,
    rois_parquet: Path,
    output_parquet: Path,
    tile_png_dir: Path,
    tile_size: int,
    overlap: int,
    limit_bounds: bool,
    label_overlap_threshold: float,
    tissue_frac_threshold: float,
    save_png: bool,
    manifest_path: Path,
    split: str | None,
    annotation_tool: str | None,
    label_version: str | None,
    only_labeled: bool,
    min_dzi_level: int | None,
    max_dzi_level: int | None,
    force: bool,
) -> None:
    import openslide
    from openslide.deepzoom import DeepZoomGenerator

    if output_parquet.exists() and not force:
        return

    ensure_dir(output_parquet.parent)
    if save_png:
        ensure_dir(tile_png_dir)

    roi_table = pq.read_table(rois_parquet)
    roi_df = roi_table.to_pandas()
    roi_df = roi_df[roi_df["slide_id"] == slide_id]

    rois = [wkt.loads(g) for g in roi_df["geometry_wkt"].tolist()]
    roi_labels = roi_df["label"].tolist()
    roi_ids = roi_df["annotation_id"].tolist()
    tree = STRtree(rois) if rois else None

    slide = openslide.open_slide(str(slide_path))
    dz = DeepZoomGenerator(slide, tile_size=tile_size, overlap=overlap, limit_bounds=limit_bounds)

    records: list[TileRecord] = []
    for level in range(dz.level_count):
        if min_dzi_level is not None and level < min_dzi_level:
            continue
        if max_dzi_level is not None and level > max_dzi_level:
            continue
        cols, rows = dz.level_tiles[level]
        level_dir = tile_png_dir / str(level)
        if save_png:
            ensure_dir(level_dir)
        for row in range(rows):
            for col in range(cols):
                tile = dz.get_tile(level, (col, row))
                coords = dz.get_tile_coordinates(level, (col, row))
                (x, y) = coords[0]
                (w, h) = coords[2]
                tile_geom = box(x, y, x + w, y + h)

                label = None
                overlap_frac = 0.0
                roi_id_list: list[str] = []
                if tree:
                    candidates = tree.query(tile_geom)
                    if candidates is not None and len(candidates) > 0:
                        # Shapely STRtree may return geometries or indices depending on version.
                        first = candidates[0]
                        if hasattr(first, "geom_type"):
                            geom_list = list(candidates)
                            labels = []
                            ids = []
                            for geom in geom_list:
                                idx = rois.index(geom)
                                labels.append(roi_labels[idx])
                                ids.append(roi_ids[idx])
                        else:
                            idx_list = [int(i) for i in list(candidates)]
                            geom_list = [rois[i] for i in idx_list]
                            labels = [roi_labels[i] for i in idx_list]
                            ids = [roi_ids[i] for i in idx_list]
                        label, roi_id_list, overlap_frac = assign_label(
                            tile_geom, geom_list, labels, ids, label_overlap_threshold
                        )

                if only_labeled and label is None:
                    continue

                tissue_frac = tissue_fraction(tile)
                qc_flags = []
                if tissue_frac < tissue_frac_threshold:
                    qc_flags.append("low_tissue")

                tile_png_path = None
                if save_png:
                    tile_png_path = str(level_dir / f"{col}_{row}.png")
                    tile.save(tile_png_path, format="PNG")

                tile_id = f"{slide_id}_L{level}_{col}_{row}"

                records.append(
                    TileRecord(
                        slide_id=slide_id,
                        tile_id=tile_id,
                        dzi_level=level,
                        tile_col=col,
                        tile_row=row,
                        x0=float(x),
                        y0=float(y),
                        w=float(w),
                        h=float(h),
                        label=label,
                        roi_ids=roi_id_list,
                        roi_overlap_frac=float(overlap_frac),
                        tissue_frac=float(tissue_frac),
                        qc_flags=qc_flags,
                        tile_png_path=tile_png_path,
                        split=split,
                    )
                )

    slide_ids = [r.slide_id for r in records]
    tile_ids = [r.tile_id for r in records]
    dzi_levels = [r.dzi_level for r in records]
    tile_cols = [r.tile_col for r in records]
    tile_rows = [r.tile_row for r in records]
    x0s = [r.x0 for r in records]
    y0s = [r.y0 for r in records]
    ws = [r.w for r in records]
    hs = [r.h for r in records]
    labels = [r.label for r in records]
    roi_ids = [r.roi_ids for r in records]
    roi_overlap = [r.roi_overlap_frac for r in records]
    tissue_fracs = [r.tissue_frac for r in records]
    qc_flags = [r.qc_flags for r in records]
    tile_png_paths = [r.tile_png_path for r in records]
    splits = [r.split for r in records]

    table = pa.table(
        {
            "slide_id": slide_ids,
            "tile_id": tile_ids,
            "dzi_level": pa.array(dzi_levels, type=pa.int32()),
            "tile_col": pa.array(tile_cols, type=pa.int32()),
            "tile_row": pa.array(tile_rows, type=pa.int32()),
            "x0": pa.array(x0s, type=pa.float64()),
            "y0": pa.array(y0s, type=pa.float64()),
            "w": pa.array(ws, type=pa.float64()),
            "h": pa.array(hs, type=pa.float64()),
            "label": pa.array(labels, type=pa.string()),
            "roi_ids": pa.array(roi_ids, type=pa.list_(pa.string())),
            "roi_overlap_frac": pa.array(roi_overlap, type=pa.float64()),
            "tissue_frac": pa.array(tissue_fracs, type=pa.float64()),
            "qc_flags": pa.array(qc_flags, type=pa.list_(pa.string())),
            "tile_png_path": pa.array(tile_png_paths, type=pa.string()),
            "split": pa.array(splits, type=pa.string()),
        }
    )
    pq.write_table(table, output_parquet)

    manifest = Manifest(
        name="tiles_manifest",
        created_at=utc_now(),
        version="1.0",
        params={
            "tile_size": tile_size,
            "overlap": overlap,
            "limit_bounds": limit_bounds,
            "label_overlap_threshold": label_overlap_threshold,
            "tissue_frac_threshold": tissue_frac_threshold,
            "save_png": save_png,
            "annotation_tool": annotation_tool,
            "label_version": label_version,
            "only_labeled": only_labeled,
            "min_dzi_level": min_dzi_level,
            "max_dzi_level": max_dzi_level,
        },
        inputs={"slide_id": slide_id, "slide_path": str(slide_path)},
        outputs={"tiles_parquet": str(output_parquet), "tile_png_dir": str(tile_png_dir)},
    )
    write_json(manifest_path, manifest.to_dict())
