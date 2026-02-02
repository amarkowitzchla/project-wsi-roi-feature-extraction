from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from wsi_pipeline.utils.common import Manifest, ensure_dir, utc_now, write_json


def build_dataset(
    tiles_parquet: Path,
    features_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    write_csv: bool,
    force: bool,
) -> None:
    ensure_dir(output_dir)
    if any(output_dir.glob("*.parquet")) and not force:
        return

    tiles_table = pq.read_table(tiles_parquet)
    tiles_df = tiles_table.to_pandas()

    features_table = ds.dataset(features_dir, format="parquet").to_table()
    features_df = features_table.to_pandas()
    if "slide_id" not in features_df.columns and "tile_id" in features_df.columns:
        features_df["slide_id"] = features_df["tile_id"].str.split("_L", n=1).str[0]

    merged = tiles_df.merge(features_df, on=["slide_id", "tile_id"], how="inner")

    if "embedding" in merged.columns:
        embed_list = [list(map(float, e)) for e in merged["embedding"].tolist()]
        embed_dim = len(embed_list[0]) if embed_list else 0
        for i in range(embed_dim):
            merged[f"embed_{i}"] = [row[i] for row in embed_list]
        merged = merged.drop(columns=["embedding"])

    # Final dataset schema: slide_id, label, embed_0..embed_N
    keep_cols = ["slide_id", "label"] + [c for c in merged.columns if c.startswith("embed_")]
    merged = merged[keep_cols]
    table_dict = {col: merged[col].tolist() for col in merged.columns}
    table = pa.table(table_dict)
    partitioning = []
    if "split" in merged.columns:
        partitioning.append("split")
    partitioning.append("slide_id")

    existing_behavior = "overwrite_or_ignore" if force else "error"
    ds.write_dataset(
        table,
        output_dir,
        format="parquet",
        partitioning=partitioning,
        existing_data_behavior=existing_behavior,
    )
    if write_csv:
        csv_path = output_dir.parent / "dataset.csv"
        table.to_pandas().to_csv(csv_path, index=False)

    manifest = Manifest(
        name="dataset_manifest",
        created_at=utc_now(),
        version="1.0",
        params={"write_csv": write_csv},
        inputs={"tiles_parquet": str(tiles_parquet), "features_dir": str(features_dir)},
        outputs={"dataset_dir": str(output_dir)},
    )
    write_json(manifest_path, manifest.to_dict())
