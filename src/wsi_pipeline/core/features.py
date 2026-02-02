from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

from wsi_pipeline.models.uni import load_uni_model
from wsi_pipeline.utils.common import Manifest, ensure_dir, sha256_file, utc_now, write_json


def _preprocess(img: Image.Image, image_size: int):
    img = img.convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img).astype("float32") / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    import torch

    return torch.from_numpy(arr)


def extract_features(
    tiles_parquet: Path,
    output_dir: Path,
    model_path: Path,
    batch_size: int,
    device: str,
    image_size: int,
    manifest_path: Path,
    only_labeled: bool,
    min_dzi_level: int | None,
    max_dzi_level: int | None,
    write_csv: bool,
    force: bool,
) -> None:
    ensure_dir(output_dir)
    if any(output_dir.glob("*.parquet")) and not force:
        return

    import torch

    model = load_uni_model(model_path, device)
    model_sha256 = sha256_file(model_path)

    tiles_table = pq.read_table(tiles_parquet)
    tiles_df = tiles_table.to_pandas()
    if only_labeled:
        tiles_df = tiles_df[tiles_df["label"].notna()]
    if min_dzi_level is not None:
        tiles_df = tiles_df[tiles_df["dzi_level"] >= min_dzi_level]
    if max_dzi_level is not None:
        tiles_df = tiles_df[tiles_df["dzi_level"] <= max_dzi_level]
    tiles_df = tiles_df[tiles_df["tile_png_path"].notna()]

    embeddings = []
    slide_ids = []
    tile_ids = []

    batch_imgs = []
    batch_meta = []

    for _, row in tqdm(tiles_df.iterrows(), total=len(tiles_df), desc="Extracting UNI"):
        img_path = Path(row["tile_png_path"])
        if not img_path.exists():
            continue
        img = Image.open(img_path)
        batch_imgs.append(_preprocess(img, image_size))
        batch_meta.append((row["slide_id"], row["tile_id"]))

        if len(batch_imgs) == batch_size:
            batch = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                out = model(batch)
                if isinstance(out, (tuple, list)):
                    out = out[0]
            out_np = out.detach().cpu().numpy()
            embeddings.extend(out_np.tolist())
            slide_ids.extend([m[0] for m in batch_meta])
            tile_ids.extend([m[1] for m in batch_meta])
            batch_imgs = []
            batch_meta = []

    if batch_imgs:
        batch = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            out = model(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
        out_np = out.detach().cpu().numpy()
        embeddings.extend(out_np.tolist())
        slide_ids.extend([m[0] for m in batch_meta])
        tile_ids.extend([m[1] for m in batch_meta])

    embed_list = [np.asarray(e, dtype="float32") for e in embeddings]
    embed_dim = int(embed_list[0].shape[0]) if embed_list else 0
    table_dict = {
        "slide_id": slide_ids,
        "tile_id": tile_ids,
        "encoder": ["UNI"] * len(tile_ids),
        "model_sha256": [model_sha256] * len(tile_ids),
    }
    if embed_dim > 0:
        for i in range(embed_dim):
            table_dict[f"embed_{i}"] = [float(v[i]) for v in embed_list]
    table = pa.table(table_dict)

    existing_behavior = "overwrite_or_ignore" if force else "error"
    ds.write_dataset(
        table,
        output_dir,
        format="parquet",
        partitioning=["slide_id"],
        existing_data_behavior=existing_behavior,
    )
    if write_csv:
        csv_dir = output_dir.parent / "features_csv"
        ensure_dir(csv_dir)
        csv_path = csv_dir / "tile_features.csv"
        table.to_pandas().to_csv(csv_path, index=False)

    manifest = Manifest(
        name="features_manifest",
        created_at=utc_now(),
        version="1.0",
        params={
            "batch_size": batch_size,
            "device": device,
            "image_size": image_size,
            "only_labeled": only_labeled,
            "min_dzi_level": min_dzi_level,
            "max_dzi_level": max_dzi_level,
            "write_csv": write_csv,
        },
        inputs={
            "tiles_parquet": str(tiles_parquet),
            "model_path": str(model_path),
            "model_sha256": model_sha256,
        },
        outputs={"features_dir": str(output_dir)},
    )
    write_json(manifest_path, manifest.to_dict())
