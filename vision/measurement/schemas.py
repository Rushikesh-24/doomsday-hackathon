"""Measurement output contract."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from vision.schemas import ImageInfo

MEASUREMENT_SCHEMA_VERSION = "1.0.0"


class MeasurementStatus(str, Enum):
    VERIFIED = "VERIFIED"                                       # matches a supplied manual measurement
    ESTIMATED = "ESTIMATED"                                     # computed with a valid scale (or scale-free)
    REQUIRES_CALIBRATION = "REQUIRES_CALIBRATION"               # absolute size but no metric calibration
    REQUIRES_MANUAL_MEASUREMENT = "REQUIRES_MANUAL_MEASUREMENT" # not observable from imagery
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"                     # refused: not enough visual evidence


class ScaleStatus(str, Enum):
    calibrated_reference = "calibrated_reference"      # user-provided segment/object length
    calibrated_marker = "calibrated_marker"            # ArUco marker of known size
    calibrated_multi_image = "calibrated_multi_image"  # scale transferred from another calibrated image
    prior_reference_object = "prior_reference_object"  # known-object priors (door / person height)
    uncalibrated_monocular = "uncalibrated_monocular"  # metric-depth network scale only (not trusted)


class ModelRef(BaseModel):
    role: str
    key: str
    name: str
    version: str
    license: str | None = None
    device: str | None = None
    revision: str | None = None


class CameraInfo(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    hfov_deg: float
    source: str                      # user_intrinsics | user_fov | exif_focal_35mm | depth_model_fov | prior
    focal_rel_sigma: float
    distortion_corrected: bool = False


class ReferenceUse(BaseModel):
    kind: str                        # segment | object | marker | prior:door | prior:person | multi_image
    description: str
    detection_id: str | None = None
    true_length_m: float | None = None
    measured_unscaled: float | None = None
    implied_scale: float | None = None
    rel_sigma: float | None = None
    used: bool = True
    reason: str | None = None


class ScaleInfo(BaseModel):
    status: ScaleStatus
    scale_factor: float = 1.0        # multiplier applied to network depth
    rel_sigma: float                 # 1-sigma relative uncertainty of the scale
    reference_depth_m: float | None = None
    references: list[ReferenceUse] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def calibrated(self) -> bool:
        return self.status not in (ScaleStatus.uncalibrated_monocular,)


class Overlay(BaseModel):
    lines: list[list[float]] = Field(default_factory=list)      # [x1,y1,x2,y2] original-image px
    points: list[list[float]] = Field(default_factory=list)
    polygon: list[list[float]] | None = None
    label_at: list[float] | None = None


class GroundTruthComparison(BaseModel):
    manual_value: float
    engine_value: float | None
    error: float | None = None
    abs_error: float | None = None
    rel_error: float | None = None
    within_reported_uncertainty: bool | None = None


class Measurement(BaseModel):
    id: str
    detection_id: str
    facility_class: str
    name: str
    value: float | int | bool | str | None
    unit: str | None = None
    uncertainty: float | None = Field(None, description="1-sigma, same unit as value")
    interval_95: list[float] | None = None
    lower_bound: bool = False            # value is only a lower bound (object truncated / occluded)
    confidence: float
    status: MeasurementStatus
    method: str
    models: list[ModelRef] = Field(default_factory=list)
    scale_status: ScaleStatus | None = None
    scale_dependent: bool = True
    verification_required: bool = True
    verification_reason: str | None = None
    provisional_value: float | None = Field(None, description="uncalibrated monocular estimate; NOT a measurement")
    refusal_reason: str | None = None
    notes: list[str] = Field(default_factory=list)
    ground_truth: GroundTruthComparison | None = None
    overlay: Overlay | None = None
    timestamp: str


class FacilityResult(BaseModel):
    detection_id: str
    class_name: str
    detection_confidence: float
    bbox: list[float]
    mask_area_px: int | None = None
    mask_source: str | None = None
    truncated_sides: list[str] = Field(default_factory=list)
    mean_depth_m: float | None = None
    measurements: list[Measurement] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class AssessmentOutcome(str, Enum):
    POTENTIAL_ISSUE = "POTENTIAL_ISSUE"            # violates threshold even at the favourable end of uncertainty
    POSSIBLE_ISSUE = "POSSIBLE_ISSUE"              # threshold within the uncertainty band / estimate violates it
    NO_ISSUE_INDICATED = "NO_ISSUE_INDICATED"      # within threshold incl. uncertainty, but NOT verified
    MEETS_THRESHOLD_VERIFIED = "MEETS_THRESHOLD_VERIFIED"  # manual measurement satisfies this single rule
    CANNOT_ASSESS = "CANNOT_ASSESS"                # missing / uncalibrated / insufficient measurement
    NOT_CONFIGURED = "NOT_CONFIGURED"              # rule threshold not set in the profile


class AssessmentItem(BaseModel):
    rule_id: str
    profile: str
    clause: str | None = None
    detection_id: str
    facility_class: str
    measurement_id: str | None = None
    measurement_name: str
    outcome: AssessmentOutcome
    severity: str = "standard"
    threshold: Any = None
    measured_value: Any = None
    uncertainty: float | None = None
    message: str


class AssessmentSummary(BaseModel):
    profile: str
    profile_title: str
    counts: dict[str, int]
    statement: str = ("Screening only. No compliance determination is made; flagged items and all "
                      "unverified measurements require on-site verification by a qualified assessor.")


class MeasurementReport(BaseModel):
    schema_version: str = MEASUREMENT_SCHEMA_VERSION
    run_id: str
    image: ImageInfo
    vision_request_id: str | None = None
    detections_source: str               # provided | stored | computed
    camera: CameraInfo
    scale: ScaleInfo
    models: list[ModelRef]
    scene: dict[str, Any] = Field(default_factory=dict)
    facilities: list[FacilityResult]
    assessment: list[AssessmentItem]
    assessment_summary: AssessmentSummary
    timing_ms: dict[str, float]
    processed_at: str
    annotated_image: str | None = None
    json_path: str | None = None
    warnings: list[str] = Field(default_factory=list)


class MeasurementBatchResult(BaseModel):
    schema_version: str = MEASUREMENT_SCHEMA_VERSION
    batch_id: str
    reports: list[MeasurementReport]
    fused: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)
