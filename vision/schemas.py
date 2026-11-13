"""Structured output schema (pydantic). This is the public contract consumed by
future Flutter / Next.js clients. Bounding boxes are given in absolute pixels of the
ORIGINAL image (after EXIF orientation is applied) and normalised to [0, 1]."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"


class AttributeStatus(str, Enum):
    detected = "detected"                          # directly supported by model detections
    estimated = "estimated"                        # heuristic estimate from 2D pixels; approximate
    not_detected = "not_detected"                  # looked for it, nothing found (NOT proof of absence)
    requires_measurement = "requires_measurement"  # needs on-site measurement / depth / calibration
    not_determinable = "not_determinable"          # cannot be inferred from this input


class Attribute(BaseModel):
    value: Any = None
    status: AttributeStatus
    method: str | None = None    # how the value was obtained
    note: str | None = None
    confidence: float | None = None


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    x1n: float = Field(description="normalised x1 in [0,1]")
    y1n: float
    x2n: float
    y2n: float
    width: float
    height: float

    @classmethod
    def from_xyxy(cls, xyxy, img_w: int, img_h: int) -> "BBox":
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        x1, x2 = max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))
        y1, y2 = max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))
        r = lambda v: round(v, 2)
        n = lambda v: round(v, 5)
        return cls(x1=r(x1), y1=r(y1), x2=r(x2), y2=r(y2),
                   x1n=n(x1 / img_w), y1n=n(y1 / img_h), x2n=n(x2 / img_w), y2n=n(y2 / img_h),
                   width=r(x2 - x1), height=r(y2 - y1))

    def xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


class SourceRef(BaseModel):
    name: str
    kind: str = "image"           # image | video_frame
    frame_index: int | None = None
    timestamp_s: float | None = None


class ModelInfo(BaseModel):
    key: str
    name: str
    version: str
    source: str | None = None
    license: str | None = None
    device: str | None = None
    weights_sha256: str | None = None
    revision: str | None = None


class Detection(BaseModel):
    id: str
    class_name: str
    category: str
    confidence: float
    bbox: BBox
    source: SourceRef
    prompt: str | None = Field(None, description="text prompt / raw label that fired")
    detector: str | None = Field(None, description="member model that produced it (ensembles)")
    track_id: int | None = None
    attributes: dict[str, Attribute] = Field(default_factory=dict)
    model_name: str
    model_version: str
    timestamp: str


class ImageInfo(BaseModel):
    name: str
    width: int
    height: int
    format: str | None = None
    channels: int = 3
    exif_transposed: bool = False


class Timing(BaseModel):
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    attributes_ms: float
    total_ms: float


class SceneFinding(BaseModel):
    code: str
    message: str
    severity: str = "info"     # info | warning
    detection_ids: list[str] = Field(default_factory=list)


class ImageResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    image: ImageInfo
    model: ModelInfo
    thresholds: dict[str, float]
    detections: list[Detection]
    counts_by_class: dict[str, int]
    findings: list[SceneFinding] = Field(default_factory=list)
    timing: Timing
    processed_at: str
    annotated_image: str | None = Field(None, description="path or URL of annotated image")
    json_path: str | None = None
    warnings: list[str] = Field(default_factory=list)


class BatchResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    batch_id: str
    results: list[ImageResult]
    errors: list[dict[str, str]] = Field(default_factory=list)
    total_ms: float


class TrackSummary(BaseModel):
    track_id: int
    class_name: str
    category: str
    first_seen_frame: int
    last_seen_frame: int
    first_seen_s: float
    last_seen_s: float
    observations: int
    max_confidence: float
    mean_confidence: float
    best_frame_index: int
    best_bbox: BBox
    attributes: dict[str, Attribute] = Field(default_factory=dict)


class VideoInfo(BaseModel):
    name: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec: str | None = None


class VideoResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    video: VideoInfo
    model: ModelInfo
    sampling: dict[str, Any]
    frames_processed: int
    frames_skipped_static: int
    unique_facilities: list[TrackSummary]
    unique_counts_by_class: dict[str, int]
    frame_results: list[ImageResult]
    timing: dict[str, float]
    processed_at: str
    annotated_video: str | None = None
    annotated_frames_dir: str | None = None
    json_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
