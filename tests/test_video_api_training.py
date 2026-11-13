from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from tests.conftest import png_bytes, scene
from vision.video.processor import InvalidVideoError, analyze_video


def make_video(path: Path, n=60, fps=10, still_from=40):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(n):
        img = np.full((240, 320, 3), 128, np.uint8)
        x = 20 + 3 * min(i, still_from)                      # stairs move, then the camera stops
        img[60:180, x:x + 100] = (0, 0, 255)                  # BGR red == RGB (255,0,0) -> staircase
        w.write(img)
    w.release()


def test_video_tracking_dedup_and_static_skip(blob_pipeline, tmp_path):
    p = tmp_path / "walk.mp4"
    make_video(p)
    r = analyze_video(p, sample_fps=5, pipeline=blob_pipeline, output_root=tmp_path / "out", conf=0.5)
    assert r.video.fps == 10 and r.video.frame_count == 60
    assert r.frames_processed + r.frames_skipped_static == 30      # every 2nd of 60 frames sampled
    assert r.frames_skipped_static >= 1                             # still segment skipped
    assert r.unique_counts_by_class == {"staircase": 1}             # one staircase, not one per frame
    t = r.unique_facilities[0]
    assert t.observations == r.frames_processed and t.first_seen_s == 0.0
    assert Path(r.annotated_video).exists() and Path(r.annotated_video).stat().st_size > 0
    assert len(list(Path(r.annotated_frames_dir).glob("*.jpg"))) == r.frames_processed
    assert Path(r.json_path).exists()


def test_invalid_video(tmp_path, blob_pipeline):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(InvalidVideoError):
        analyze_video(bad, pipeline=blob_pipeline)
    txt = tmp_path / "x.avi"
    txt.write_bytes(b"RIFF....AVI ")
    with pytest.raises(InvalidVideoError):
        analyze_video(txt, pipeline=blob_pipeline)


# ------------------------------------------------------------------- API
@pytest.fixture
def client(blob_pipeline, monkeypatch):
    monkeypatch.setenv("VISION_PRELOAD", "0")
    import vision.inference.pipeline as P
    monkeypatch.setattr(P, "_pipeline", blob_pipeline)
    from fastapi.testclient import TestClient
    from api.app import app
    with TestClient(app) as c:
        yield c


def test_api_health_models(client):
    h = client.get("/vision/health").json()
    assert h["status"] == "ok" and h["default_model"] == "blob"
    m = client.get("/vision/models").json()
    assert "staircase" in m["taxonomy"] and any(x["key"] == "campus_ensemble" for x in m["models"])


def test_api_analyze_image(client):
    r = client.post("/vision/analyze-image", files={"file": ("s.png", png_bytes(scene()), "image/png")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["counts_by_class"]["staircase"] == 1
    assert j["annotated_image"].startswith("/vision/outputs/")
    assert client.get(j["annotated_image"]).status_code == 200


def test_api_errors(client):
    assert client.post("/vision/analyze-image", files={"file": ("x.jpg", b"junk", "image/jpeg")}).status_code == 422
    r = client.post("/vision/analyze-image?model=nope", files={"file": ("s.png", png_bytes(scene()), "image/png")})
    assert r.status_code == 400
    r = client.post("/vision/analyze-image?classes=spaceship", files={"file": ("s.png", png_bytes(scene()), "image/png")})
    assert r.status_code == 400


def test_api_batch_and_video(client, tmp_path):
    files = [("files", ("a.png", png_bytes(scene()), "image/png")), ("files", ("b.jpg", b"junk", "image/jpeg"))]
    j = client.post("/vision/analyze-batch", files=files).json()
    assert len(j["results"]) == 1 and len(j["errors"]) == 1
    p = tmp_path / "v.mp4"
    make_video(p)
    r = client.post("/vision/analyze-video?sample_fps=5&conf=0.5", files={"file": ("v.mp4", p.read_bytes(), "video/mp4")})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["unique_counts_by_class"] == {"staircase": 1} and "frame_results" not in j
    assert client.get(j["annotated_video"]).status_code == 200


# ------------------------------------------------------------------- training data prep
def test_build_dataset_remaps_and_splits(tmp_path):
    from vision.config import get_settings
    from vision.taxonomy import load_taxonomy
    from vision.training.dataset import SourceSpec, build_dataset
    src = tmp_path / "rf"
    for split in ("train", "valid"):
        (src / split / "images").mkdir(parents=True)
        (src / split / "labels").mkdir(parents=True)
    (src / "data.yaml").write_text(yaml.safe_dump({"names": ["Stairs", "slope", "Mystery-Thing"]}))
    for i in range(20):
        sp = "train" if i < 15 else "valid"
        img = np.random.RandomState(i).randint(0, 255, (64, 64, 3), np.uint8)
        cv2.imwrite(str(src / sp / "images" / f"{i}.jpg"), img)
        (src / sp / "labels" / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n2 0.1 0.1 0.1 0.1\n")
    cv2.imwrite(str(src / "train" / "images" / "dup.jpg"), np.random.RandomState(0).randint(0, 255, (64, 64, 3), np.uint8))
    s = get_settings()
    out = tmp_path / "ds"
    man = build_dataset([SourceSpec(src, "yolo", "rf_test", license="CC BY 4.0")], out,
                        Path("configs/dataset_class_map.yaml").resolve(), load_taxonomy(str(s.taxonomy_config)).names())
    assert man["classes"] == ["ramp", "staircase"]
    assert man["sources"][0]["unmapped_labels"] == {"Mystery-Thing": 20}
    assert man["duplicates_removed"] == 1 and sum(man["images_per_split"].values()) == 20
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "ramp", 1: "staircase"}
    lbl = next((out / "labels").rglob("*.txt")).read_text().splitlines()
    assert len(lbl) == 2 and lbl[0].startswith("1 ")  # Stairs -> staircase (id 1); Mystery dropped
    man2 = build_dataset([SourceSpec(src, "yolo", "rf_test")], tmp_path / "ds2",
                         Path("configs/dataset_class_map.yaml").resolve(), load_taxonomy(str(s.taxonomy_config)).names())
    assert man2["images_per_split"] == man["images_per_split"]  # deterministic split
