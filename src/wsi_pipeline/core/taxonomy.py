from __future__ import annotations

from pathlib import Path
from typing import Iterable

from wsi_pipeline.utils.common import read_yaml


class LabelTaxonomyError(ValueError):
    pass


def load_taxonomy(path: Path) -> dict:
    data = read_yaml(path)
    if "classes" not in data:
        raise LabelTaxonomyError(f"labels.yaml missing 'classes': {path}")
    return data


def validate_labels(labels: Iterable[str], taxonomy: dict, allow_unknown: bool) -> None:
    classes = set(taxonomy.get("classes", []))
    unknown = sorted({label for label in labels if label not in classes})
    if unknown and not allow_unknown:
        raise LabelTaxonomyError(
            "Unknown labels: " + ", ".join(unknown)
        )
