from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from wsi_pipeline.core.project import ProjectPaths, init_project, load_project_config
from wsi_pipeline.core.slides import find_slides, scan_slides, load_slides_manifest
from wsi_pipeline.core.rois import build_rois
from wsi_pipeline.core.taxonomy import load_taxonomy
from wsi_pipeline.core.deepzoom import build_deepzoom
from wsi_pipeline.core.tiles import build_tiles
from wsi_pipeline.core.features import extract_features
from wsi_pipeline.core.dataset import build_dataset
from wsi_pipeline.core.splits import assign_splits
from wsi_pipeline.utils.common import ensure_dir, resolve_path

app = typer.Typer(add_completion=False)


def _resolve_model_path(project_root: Path, model_path: Optional[Path]) -> Path:
    if model_path is not None:
        return model_path
    candidate = project_root / "models" / "model.pth"
    if candidate.exists():
        return candidate
    repo_root = Path(__file__).resolve().parents[2]
    repo_candidate = repo_root / "models" / "model.pth"
    if repo_candidate.exists():
        return repo_candidate
    raise typer.BadParameter(
        "Model not found. Provide --model-path or place it at models/model.pth."
    )


@app.command()
def init(project_dir: Path, force: bool = typer.Option(False, "--force")):
    """Initialize a new WSI project directory."""
    init_project(project_dir, overwrite=force)
    typer.echo(f"Initialized project at {project_dir}")


slides_app = typer.Typer()
app.add_typer(slides_app, name="slides")


@slides_app.command("scan")
def slides_scan(
    project: Path = typer.Option(..., "--project"),
    sha256: bool = typer.Option(False, "--sha256"),
):
    paths = ProjectPaths(project)
    cfg = load_project_config(project)
    repo_paths = [resolve_path(project, p) for p in cfg.get("slide_repos", [])]
    slide_paths = find_slides(repo_paths)
    scan_slides(
        slide_paths,
        paths.slides_manifest,
        paths.manifests / "slides_manifest.json",
        compute_sha256=sha256,
    )
    typer.echo(f"Scanned {len(slide_paths)} slides")


rois_app = typer.Typer()
app.add_typer(rois_app, name="rois")


@rois_app.command("build")
def rois_build(
    project: Path = typer.Option(..., "--project"),
    tool: str = typer.Option("generic_geojson_v1", "--tool"),
    input_dir: Optional[Path] = typer.Option(None, "--in"),
    output: Optional[Path] = typer.Option(None, "--out"),
    allow_unknown_labels: bool = typer.Option(False, "--allow-unknown-labels"),
):
    paths = ProjectPaths(project)
    cfg = load_project_config(project)
    taxonomy = load_taxonomy(paths.labels_yaml)

    if input_dir is None:
        tool_dir = "generic"
        if tool == "qupath_geojson":
            tool_dir = "qupath"
        elif tool == "sectra":
            tool_dir = "sectra"
        input_dir = paths.annotations_raw / tool_dir
    if output is None:
        output = paths.silver / "rois" / "rois.parquet"

    slides_manifest = load_slides_manifest(paths.slides_manifest)
    slide_ids = [row["slide_id"] for row in slides_manifest]

    build_rois(
        slide_ids=slide_ids,
        input_dir=input_dir,
        output_parquet=output,
        output_geojson=paths.annotations_canonical / "rois.geojson",
        tool=tool,
        taxonomy=taxonomy,
        allow_unknown_labels=allow_unknown_labels,
        manifest_path=paths.manifests / "rois_manifest.json",
    )
    typer.echo("Built canonical ROIs")


deepzoom_app = typer.Typer()
app.add_typer(deepzoom_app, name="deepzoom")


@deepzoom_app.command("build")
def deepzoom_build(
    project: Path = typer.Option(..., "--project"),
    tile_size: Optional[int] = typer.Option(None, "--tile-size"),
    overlap: Optional[int] = typer.Option(None, "--overlap"),
    limit_bounds: Optional[bool] = typer.Option(None, "--limit-bounds"),
    force: bool = typer.Option(False, "--force"),
):
    paths = ProjectPaths(project)
    cfg = load_project_config(project)
    tile_size = tile_size or cfg["tiling"]["tile_size"]
    overlap = overlap if overlap is not None else cfg["tiling"]["overlap"]
    limit_bounds = limit_bounds if limit_bounds is not None else cfg["tiling"]["limit_bounds"]

    slides = load_slides_manifest(paths.slides_manifest)
    for row in slides:
        slide_id = row["slide_id"]
        slide_path = Path(row["slide_path"])
        output_dir = paths.bronze / "deepzoom" / slide_id
        build_deepzoom(
            slide_id,
            slide_path,
            output_dir,
            tile_size,
            overlap,
            limit_bounds,
            paths.manifests / f"deepzoom_manifest_{slide_id}.json",
            force,
        )
    typer.echo("DeepZoom build complete")


tiles_app = typer.Typer()
app.add_typer(tiles_app, name="tiles")


@tiles_app.command("build")
def tiles_build(
    project: Path = typer.Option(..., "--project"),
    label_overlap_threshold: Optional[float] = typer.Option(None, "--label-overlap-threshold"),
    tissue_frac_threshold: Optional[float] = typer.Option(None, "--tissue-frac-threshold"),
    save_png: bool = typer.Option(True, "--save-png"),
    only_labeled: bool = typer.Option(True, "--only-labeled"),
    min_dzi_level: Optional[int] = typer.Option(None, "--min-dzi-level"),
    max_dzi_level: Optional[int] = typer.Option(None, "--max-dzi-level"),
    force: bool = typer.Option(False, "--force"),
):
    paths = ProjectPaths(project)
    cfg = load_project_config(project)
    taxonomy = load_taxonomy(paths.labels_yaml)
    label_overlap_threshold = (
        label_overlap_threshold if label_overlap_threshold is not None else cfg["tiles"]["label_overlap_threshold"]
    )
    tissue_frac_threshold = (
        tissue_frac_threshold if tissue_frac_threshold is not None else cfg["tiles"]["tissue_frac_threshold"]
    )

    slides = load_slides_manifest(paths.slides_manifest)
    splits = assign_splits(slides, cfg["splits"]["seed"], cfg["splits"])

    for row in slides:
        slide_id = row["slide_id"]
        slide_path = Path(row["slide_path"])
        build_tiles(
            slide_id=slide_id,
            slide_path=slide_path,
            rois_parquet=paths.silver / "rois" / "rois.parquet",
            output_parquet=paths.silver / "tiles" / f"tiles_{slide_id}.parquet",
            tile_png_dir=paths.bronze / "tile_pngs" / slide_id,
            tile_size=cfg["tiling"]["tile_size"],
            overlap=cfg["tiling"]["overlap"],
            limit_bounds=cfg["tiling"]["limit_bounds"],
            label_overlap_threshold=label_overlap_threshold,
            tissue_frac_threshold=tissue_frac_threshold,
            save_png=save_png,
            manifest_path=paths.manifests / f"tiles_manifest_{slide_id}.json",
            split=splits.get(slide_id),
            annotation_tool=cfg.get("annotation_tools"),
            label_version=taxonomy.get("version"),
            only_labeled=only_labeled,
            min_dzi_level=min_dzi_level,
            max_dzi_level=max_dzi_level,
            force=force,
        )
    typer.echo("Tiles build complete")


features_app = typer.Typer()
app.add_typer(features_app, name="features")


@features_app.command("extract")
def features_extract(
    project: Path = typer.Option(..., "--project"),
    model_path: Optional[Path] = typer.Option(None, "--model-path"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    device: Optional[str] = typer.Option(None, "--device"),
    only_labeled: bool = typer.Option(True, "--only-labeled"),
    min_dzi_level: Optional[int] = typer.Option(None, "--min-dzi-level"),
    max_dzi_level: Optional[int] = typer.Option(None, "--max-dzi-level"),
    write_csv: bool = typer.Option(False, "--write-csv"),
    force: bool = typer.Option(False, "--force"),
):
    paths = ProjectPaths(project)
    cfg = load_project_config(project)
    model_path = _resolve_model_path(paths.root, model_path)
    batch_size = batch_size or cfg["features"]["batch_size"]
    device = device or cfg["features"]["device"]
    image_size = cfg["features"]["image_size"]

    tiles_dir = paths.silver / "tiles"
    tiles_files = sorted(tiles_dir.glob("tiles_*.parquet"))
    if not tiles_files:
        raise typer.BadParameter("No tiles parquet found. Run `wsi tiles build` first.")

    # Concatenate tiles parquet into one for feature extraction
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.read_table(f) for f in tiles_files]
    tiles_table = pa.concat_tables(tables)
    combined_tiles = paths.silver / "tiles" / "tiles.parquet"
    pq.write_table(tiles_table, combined_tiles)

    extract_features(
        tiles_parquet=combined_tiles,
        output_dir=paths.gold / "features",
        model_path=model_path,
        batch_size=batch_size,
        device=device,
        image_size=image_size,
        manifest_path=paths.manifests / "features_manifest.json",
        only_labeled=only_labeled,
        min_dzi_level=min_dzi_level,
        max_dzi_level=max_dzi_level,
        write_csv=write_csv,
        force=force,
    )
    typer.echo("Feature extraction complete")


dataset_app = typer.Typer()
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("build")
def dataset_build(
    project: Path = typer.Option(..., "--project"),
    write_csv: bool = typer.Option(False, "--write-csv"),
    force: bool = typer.Option(False, "--force"),
):
    paths = ProjectPaths(project)

    # Merge per-slide tiles into a single parquet for dataset build
    tiles_dir = paths.silver / "tiles"
    tiles_files = sorted(tiles_dir.glob("tiles_*.parquet"))
    if not tiles_files:
        raise typer.BadParameter("No tiles parquet found. Run `wsi tiles build` first.")

    # Concatenate tiles parquet into one
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.read_table(f) for f in tiles_files]
    tiles_table = pa.concat_tables(tables)
    combined_tiles = paths.silver / "tiles" / "tiles.parquet"
    pq.write_table(tiles_table, combined_tiles)

    build_dataset(
        tiles_parquet=combined_tiles,
        features_dir=paths.gold / "features",
        output_dir=paths.gold / "dataset",
        manifest_path=paths.manifests / "dataset_manifest.json",
        write_csv=write_csv,
        force=force,
    )
    typer.echo("Dataset build complete")


slide_app = typer.Typer()
app.add_typer(slide_app, name="slide")


@slide_app.command("run")
def slide_run(
    slide_path: Path = typer.Option(..., "--slide-path"),
    roi_path: Path = typer.Option(..., "--roi-path"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    tool: str = typer.Option("generic_geojson_v1", "--tool"),
    model_path: Optional[Path] = typer.Option(None, "--model-path"),
    device: Optional[str] = typer.Option(None, "--device"),
    force: bool = typer.Option(False, "--force"),
):
    """Run a single-slide pipeline in a project-like directory."""
    if not out_dir.exists() or not any(out_dir.iterdir()):
        init_project(out_dir, overwrite=False)

    paths = ProjectPaths(out_dir)
    cfg = load_project_config(out_dir)
    taxonomy = load_taxonomy(paths.labels_yaml)

    # Scan the single slide
    scan_slides(
        [slide_path],
        paths.slides_manifest,
        paths.manifests / "slides_manifest.json",
        compute_sha256=False,
    )

    slide_id = slide_path.stem
    tool_dir = paths.annotations_raw / "generic"
    if tool == "qupath_geojson":
        tool_dir = paths.annotations_raw / "qupath"
    elif tool == "sectra":
        tool_dir = paths.annotations_raw / "sectra"
    ensure_dir(tool_dir)
    dest_roi = tool_dir / f"{slide_id}.geojson"
    if not dest_roi.exists() or force:
        dest_roi.write_text(roi_path.read_text(encoding="utf-8"), encoding="utf-8")

    build_rois(
        slide_ids=[slide_id],
        input_dir=tool_dir,
        output_parquet=paths.silver / "rois" / "rois.parquet",
        output_geojson=paths.annotations_canonical / "rois.geojson",
        tool=tool,
        taxonomy=taxonomy,
        allow_unknown_labels=False,
        manifest_path=paths.manifests / "rois_manifest.json",
    )

    build_deepzoom(
        slide_id=slide_id,
        slide_path=slide_path,
        output_dir=paths.bronze / "deepzoom" / slide_id,
        tile_size=cfg["tiling"]["tile_size"],
        overlap=cfg["tiling"]["overlap"],
        limit_bounds=cfg["tiling"]["limit_bounds"],
        manifest_path=paths.manifests / f"deepzoom_manifest_{slide_id}.json",
        force=force,
    )

    splits = {slide_id: "test"}
    build_tiles(
        slide_id=slide_id,
        slide_path=slide_path,
        rois_parquet=paths.silver / "rois" / "rois.parquet",
        output_parquet=paths.silver / "tiles" / f"tiles_{slide_id}.parquet",
        tile_png_dir=paths.bronze / "tile_pngs" / slide_id,
        tile_size=cfg["tiling"]["tile_size"],
        overlap=cfg["tiling"]["overlap"],
        limit_bounds=cfg["tiling"]["limit_bounds"],
        label_overlap_threshold=cfg["tiles"]["label_overlap_threshold"],
        tissue_frac_threshold=cfg["tiles"]["tissue_frac_threshold"],
        save_png=True,
        manifest_path=paths.manifests / f"tiles_manifest_{slide_id}.json",
        split=splits.get(slide_id),
        annotation_tool=tool,
        label_version=taxonomy.get("version"),
        only_labeled=True,
        min_dzi_level=None,
        max_dzi_level=None,
        force=force,
    )

    model_path = model_path or (paths.models_dir / "model.pth")
    device = device or cfg["features"]["device"]
    model_path = _resolve_model_path(paths.root, model_path)

    extract_features(
        tiles_parquet=paths.silver / "tiles" / f"tiles_{slide_id}.parquet",
        output_dir=paths.gold / "features",
        model_path=model_path,
        batch_size=cfg["features"]["batch_size"],
        device=device,
        image_size=cfg["features"]["image_size"],
        manifest_path=paths.manifests / "features_manifest.json",
        only_labeled=True,
        min_dzi_level=None,
        max_dzi_level=None,
        write_csv=False,
        force=force,
    )

    build_dataset(
        tiles_parquet=paths.silver / "tiles" / f"tiles_{slide_id}.parquet",
        features_dir=paths.gold / "features",
        output_dir=paths.gold / "dataset",
        manifest_path=paths.manifests / "dataset_manifest.json",
        write_csv=False,
        force=force,
    )

    typer.echo("Slide pipeline complete")
