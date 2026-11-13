import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.conftest import png_bytes, scene, to_bytes
from vision.preprocessing.image import InvalidImageError


def test_image_pipeline_end_to_end(blob_pipeline, tmp_path):
    r = blob_pipeline.analyze_image(png_bytes(scene()), "scene.png", save_dir=tmp_path)
    by = {d.class_name: d for d in r.detections}
    assert set(r.counts_by_class) == {"staircase", "handrail", "obstacle", "walkway"}
    assert r.image.width == 640 and r.image.height == 480
    st = by["staircase"]
    assert st.bbox.xyxy() == [100, 100, 300, 300] and st.bbox.x1n == pytest.approx(100 / 640, abs=1e-4)
    assert st.confidence == 0.9 and st.model_name and st.model_version == "blob-1" and st.timestamp
    assert st.source.name == "scene.png"
    # accessibility attributes
    assert st.attributes["handrail_present"].value is True
    assert st.attributes["handrail_present"].status.value == "detected"
    assert st.attributes["riser_height"].status.value == "requires_measurement"
    ob = by["obstacle"]
    assert ob.attributes["obstacle_type"].value == "chair"
    assert ob.attributes["potential_obstruction"].value is True
    assert ob.attributes["relative_to_walkway"].value["lateral_position"] == "centre"
    assert {"NO_STEP_FREE_ALTERNATIVE_IN_VIEW", "POTENTIAL_PATH_OBSTRUCTION"} <= {f.code for f in r.findings}
    # artefacts
    assert Path(r.annotated_image).exists() and cv2.imread(r.annotated_image).shape == (480, 640, 3)
    saved = json.loads(Path(r.json_path).read_text())
    assert saved["detections"][0]["bbox"]["x1"] >= 0 and saved["schema_version"]
    # the 49px blob is below threshold and must not appear
    assert not any(d.bbox.x1 >= 590 and d.bbox.y2 <= 30 for d in r.detections)


def test_boxes_mapped_back_after_downscale(blob_pipeline, tmp_path):
    big = cv2.resize(scene(), (2560, 1920), interpolation=cv2.INTER_NEAREST)
    blob_pipeline.settings.max_infer_side = 640
    try:
        r = blob_pipeline.analyze_image(png_bytes(big), "big.png", save_dir=tmp_path)
    finally:
        blob_pipeline.settings.max_infer_side = 1920
    st = [d for d in r.detections if d.class_name == "staircase"][0]
    assert st.bbox.xyxy() == pytest.approx([400, 400, 1200, 1200], abs=4)
    assert any("downscaled" in w for w in r.warnings)


def test_class_filter_and_conf_override(blob_pipeline, tmp_path):
    r = blob_pipeline.analyze_image(png_bytes(scene()), "s.png", classes=["staircase"], save_dir=tmp_path)
    assert set(r.counts_by_class) == {"staircase"}
    r2 = blob_pipeline.analyze_image(png_bytes(scene()), "s.png", conf=0.95, save_dir=tmp_path)
    assert r2.detections == []
    with pytest.raises(ValueError, match="unknown classes"):
        blob_pipeline.analyze_image(png_bytes(scene()), "s.png", classes=["spaceship"], save_dir=tmp_path)


def test_batch_with_invalid_member(blob_pipeline, tmp_path):
    br = blob_pipeline.analyze_batch([(png_bytes(scene()), "ok.png"), (b"garbage", "bad.jpg")], save_dir=tmp_path)
    assert len(br.results) == 1 and br.errors[0]["name"] == "bad.jpg"
    assert (tmp_path / "batch.json").exists()


def test_invalid_image_raises(blob_pipeline):
    with pytest.raises(InvalidImageError):
        blob_pipeline.analyze_image(b"nope", "x.jpg")


def test_unknown_model(blob_pipeline):
    with pytest.raises(KeyError):
        blob_pipeline.analyze_image(png_bytes(scene()), "s.png", model="does_not_exist")


def test_missing_weights_graceful(tmp_path, monkeypatch):
    """A model whose weights are absent while offline must raise ModelUnavailableError, not crash."""
    from vision.config import Settings
    from vision.models.base import ModelUnavailableError
    from vision.models.registry import ModelRegistry
    monkeypatch.setenv("VISION_CACHE_DIR", str(tmp_path / "empty_cache"))
    monkeypatch.setenv("VISION_OFFLINE", "1")
    reg = ModelRegistry(Settings())
    with pytest.raises(ModelUnavailableError, match="not cached"):
        reg.get("yolo11_coco")
    assert "not cached" in [d for d in reg.describe() if d["key"] == "yolo11_coco"][0]["error"]
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
