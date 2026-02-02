from __future__ import annotations

from typing import Iterable

import numpy as np


def assign_splits(slides_manifest: list[dict[str, str]], seed: int, ratios: dict[str, float]) -> dict[str, str]:
    if not slides_manifest:
        return {}

    key = "patient_id" if "patient_id" in slides_manifest[0] else "slide_id"
    groups = {}
    for row in slides_manifest:
        group_id = row.get(key) or row.get("slide_id")
        groups.setdefault(group_id, []).append(row["slide_id"])

    rng = np.random.default_rng(seed)
    group_ids = sorted(groups.keys())
    rng.shuffle(group_ids)

    n = len(group_ids)
    n_train = int(n * ratios.get("train", 0.7))
    n_val = int(n * ratios.get("val", 0.15))
    n_test = n - n_train - n_val

    splits = {}
    for i, gid in enumerate(group_ids):
        if i < n_train:
            split = "train"
        elif i < n_train + n_val:
            split = "val"
        else:
            split = "test"
        for slide_id in groups[gid]:
            splits[slide_id] = split

    return splits
