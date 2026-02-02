import pytest

from wsi_pipeline.core.taxonomy import validate_labels, LabelTaxonomyError


def test_validate_labels_unknown():
    taxonomy = {"classes": ["tumor", "stroma"]}
    with pytest.raises(LabelTaxonomyError):
        validate_labels(["tumor", "unknown"], taxonomy, allow_unknown=False)


def test_validate_labels_allow_unknown():
    taxonomy = {"classes": ["tumor", "stroma"]}
    validate_labels(["tumor", "unknown"], taxonomy, allow_unknown=True)
