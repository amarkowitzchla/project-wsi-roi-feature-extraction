from wsi_pipeline.utils.common import Manifest


def test_manifest_to_dict():
    m = Manifest(
        name="test",
        created_at="2024-01-01T00:00:00Z",
        version="1.0",
        params={"a": 1},
        inputs={"b": 2},
        outputs={"c": 3},
    )
    d = m.to_dict()
    assert d["name"] == "test"
    assert d["params"]["a"] == 1
