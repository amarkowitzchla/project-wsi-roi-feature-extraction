from shapely.geometry import box

from wsi_pipeline.core.tiling_math import assign_label


def test_assign_label_overlap():
    tile = box(0, 0, 10, 10)
    roi1 = box(0, 0, 5, 5)
    roi2 = box(5, 5, 10, 10)
    label, roi_ids, overlap = assign_label(
        tile, [roi1, roi2], ["a", "b"], ["id1", "id2"], 0.2
    )
    assert label in {"a", "b"}
    assert len(roi_ids) == 2
    assert overlap > 0
