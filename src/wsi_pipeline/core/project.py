from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wsi_pipeline.utils.common import ensure_dir, read_yaml, write_yaml


PROJECT_YAML = "project.yaml"
LABELS_YAML = "labels/labels.yaml"


@dataclass
class ProjectPaths:
    root: Path

    @property
    def project_yaml(self) -> Path:
        return self.root / PROJECT_YAML

    @property
    def labels_yaml(self) -> Path:
        return self.root / LABELS_YAML

    @property
    def slides_raw(self) -> Path:
        return self.root / "slides" / "raw"

    @property
    def slides_manifest(self) -> Path:
        return self.root / "slides" / "manifest.csv"

    @property
    def annotations_raw(self) -> Path:
        return self.root / "annotations" / "raw"

    @property
    def annotations_canonical(self) -> Path:
        return self.root / "annotations" / "canonical"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def bronze(self) -> Path:
        return self.root / "bronze"

    @property
    def silver(self) -> Path:
        return self.root / "silver"

    @property
    def gold(self) -> Path:
        return self.root / "gold"

    @property
    def manifests(self) -> Path:
        return self.root / "manifests"

    @property
    def logs(self) -> Path:
        return self.root / "logs"


DEFAULT_PROJECT = {
    "project_name": "wsi_project",
    "slide_repos": ["slides/raw"],
    "annotation_tools": [],
    "label_taxonomy": LABELS_YAML,
    "tiling": {
        "tile_size": 256,
        "overlap": 0,
        "limit_bounds": True,
    },
    "tiles": {
        "label_overlap_threshold": 0.2,
        "tissue_frac_threshold": 0.2,
    },
    "features": {
        "batch_size": 64,
        "device": "cpu",
        "image_size": 224,
    },
    "splits": {
        "seed": 1337,
        "train": 0.7,
        "val": 0.15,
        "test": 0.15,
    },
}


DEFAULT_LABELS = {
    "version": "1.0",
    "classes": ["tumor", "stroma", "necrosis", "background"],
}


def init_project(root: Path, overwrite: bool = False) -> ProjectPaths:
    paths = ProjectPaths(root=root)
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"Project directory is not empty: {root}")

    ensure_dir(paths.slides_raw)
    ensure_dir(paths.annotations_raw / "qupath")
    ensure_dir(paths.annotations_raw / "sectra")
    ensure_dir(paths.annotations_raw / "generic")
    ensure_dir(paths.annotations_canonical)
    ensure_dir(paths.labels_yaml.parent)
    ensure_dir(paths.models_dir)
    ensure_dir(paths.bronze / "deepzoom")
    ensure_dir(paths.bronze / "tile_pngs")
    ensure_dir(paths.silver / "rois")
    ensure_dir(paths.silver / "tiles")
    ensure_dir(paths.gold / "features")
    ensure_dir(paths.gold / "dataset")
    ensure_dir(paths.manifests)
    ensure_dir(paths.logs)

    write_yaml(paths.project_yaml, DEFAULT_PROJECT)
    write_yaml(paths.labels_yaml, DEFAULT_LABELS)

    readme = root / "README.md"
    if not readme.exists() or overwrite:
        readme.write_text(
            "# WSI Project\n\n"
            "This directory was initialized by `wsi init`.\n"
            "See the root repository README for the full end-to-end guide.\n",
            encoding="utf-8",
        )

    return paths


def load_project_config(root: Path) -> dict[str, Any]:
    cfg = read_yaml(root / PROJECT_YAML)
    if not cfg:
        raise FileNotFoundError(f"Missing {PROJECT_YAML} in {root}")
    return cfg
