from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image


def tissue_fraction(img: Image.Image) -> float:
    arr = np.asarray(img.convert("RGB"))
    non_white = np.any(arr < 220, axis=2)
    return float(non_white.mean())


def assign_label(tile_geom, rois, labels, ids, overlap_threshold: float):
    if not rois:
        return None, [], 0.0
    overlaps = []
    for geom, label, roi_id in zip(rois, labels, ids):
        inter = tile_geom.intersection(geom)
        if inter.is_empty:
            continue
        overlap = inter.area / tile_geom.area
        overlaps.append((overlap, label, roi_id))
    if not overlaps:
        return None, [], 0.0
    overlaps.sort(reverse=True, key=lambda x: x[0])
    best_overlap, best_label, best_id = overlaps[0]
    if best_overlap < overlap_threshold:
        return None, [oid for _, _, oid in overlaps], best_overlap
    return best_label, [oid for _, _, oid in overlaps], best_overlap
