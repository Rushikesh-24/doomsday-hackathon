"""Real-model tests on real photographs (samples/images, Wikimedia Commons, see SOURCES.json).
Skipped automatically when weights are not cached locally (no surprise downloads in CI).
Assertions are deliberately loose: they check the system produces genuine detections of the
facility the photo is known to contain, not a particular box or score."""
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "samples" / "images"
CACHE = Path(os.environ.get("VISION_CACHE_DIR", ROOT / "models_cache"))


def _cached(key):
    if key == "owlv2_base":
        return (CACHE / "hf/hub/models--google--owlv2-base-patch16-ensemble").exists()
    if key == "yolo11_coco":
        return (CACHE / "ultralytics/yolo11m.pt").exists()
    return False


pytestmark = pytest.mark.skipif(not (_cached("owlv2_base") and _cached("yolo11_coco")),
                                reason="model weights not cached; run scripts/download_models.py")


@pytest.fixture(scope="module")
def pipe():
    os.environ["VISION_OFFLINE"] = "1"  # never download inside tests
    from vision.config import get_settings, reset_settings
    from vision.inference.pipeline import VisionPipeline
    from vision.models.registry import ModelRegistry
    reset_settings()
    p = VisionPipeline(registry=ModelRegistry(get_settings()))
    yield p
    os.environ.pop("VISION_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    reset_settings()


@pytest.mark.parametrize("image,expected", [
    ("stairs_oau_campus.jpg", "staircase"),
    ("tactile_paving_singapore.jpg", "tactile_paving"),
    ("elevator_lobby_hotel.jpg", "elevator"),
    ("corridor_stirling_students.jpg", "person"),
])
def test_default_ensemble_finds_known_facility(pipe, image, expected, tmp_path):
    r = pipe.analyze_image(IMAGES / image, model="campus_ensemble", save_dir=tmp_path)
    assert expected in r.counts_by_class, (image, r.counts_by_class)
    d = max((d for d in r.detections if d.class_name == expected), key=lambda d: d.confidence)
    assert 0 < d.confidence <= 1 and d.bbox.width > 0 and d.model_version
    assert Path(r.annotated_image).exists()


def test_staircase_fragments_merged(pipe, tmp_path):
    r = pipe.analyze_image(IMAGES / "stairs_oau_campus.jpg", model="campus_ensemble", save_dir=tmp_path)
    stairs = [d for d in r.detections if d.class_name == "staircase"]
    big = max(stairs, key=lambda d: d.bbox.width * d.bbox.height)
    assert big.bbox.height > 0.3 * r.image.height  # one region, not 20 per-step slivers
    assert "visible_step_count" in big.attributes
