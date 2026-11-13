"""Configurable accessibility screening rules (configs/accessibility_rules.yaml).

Outcome logic (k = 1.96, 95% band):
  max-threshold rules   value-k*sigma > T -> POTENTIAL_ISSUE ; value > T or value+k*sigma > T -> POSSIBLE_ISSUE ;
                        else NO_ISSUE_INDICATED (never "compliant")
  min-threshold rules   mirrored
  lower-bound values    can prove a min-rule is met only via the bound, and a max-rule violated only via the bound
  required / forbidden  booleans; "not detected" is POSSIBLE_ISSUE (absence of evidence != evidence of absence)
  VERIFIED measurements are the only ones that can yield MEETS_THRESHOLD_VERIFIED, per rule, never overall.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from vision.config import REPO_ROOT
from vision.measurement.schemas import (AssessmentItem, AssessmentOutcome as O, AssessmentSummary, FacilityResult,
                                        Measurement, MeasurementStatus as MS)

K = 1.96


@lru_cache(maxsize=2)
def load_rules(path: str | None = None) -> dict:
    return yaml.safe_load(Path(path or REPO_ROOT / "configs/accessibility_rules.yaml").read_text())


def _fmt(v, unit):
    if isinstance(v, float):
        return f"{v:.3f}{' ' + unit if unit else ''}"
    return f"{v}{' ' + unit if unit else ''}"


def evaluate_rule(rule: dict, profile: str, fac: FacilityResult, m: Measurement | None) -> AssessmentItem | None:
    base = dict(rule_id=rule["id"], profile=profile, clause=rule.get("clause"), detection_id=fac.detection_id,
                facility_class=fac.class_name, measurement_name=rule["measurement"],
                severity=rule.get("severity", "standard"), threshold=rule.get("threshold"))
    cmp_ = rule["comparison"]
    if cmp_ in ("max", "min") and rule.get("threshold") is None:
        return AssessmentItem(**base, outcome=O.NOT_CONFIGURED, measurement_id=m.id if m else None,
                              message="threshold not configured in this profile (verify from the official standard)")
    if m is None:
        if rule.get("optional"):
            return None
        return AssessmentItem(**base, outcome=O.CANNOT_ASSESS, message="measurement not produced for this facility")
    base.update(measurement_id=m.id, measured_value=m.value if m.value is not None else None, uncertainty=m.uncertainty)
    if m.status in (MS.INSUFFICIENT_DATA, MS.REQUIRES_MANUAL_MEASUREMENT):
        return AssessmentItem(**base, outcome=O.CANNOT_ASSESS, message=f"{m.status.value}: {m.refusal_reason}")
    if m.status == MS.REQUIRES_CALIBRATION:
        hint = ""
        if m.provisional_value is not None and rule.get("threshold") is not None:
            T = rule["threshold"]
            bad = m.provisional_value > T if cmp_ == "max" else m.provisional_value < T
            hint = (f" Provisional (uncalibrated) value {m.provisional_value:.2f} would "
                    f"{'violate' if bad else 'satisfy'} {T} - calibrate to assess.")
        return AssessmentItem(**base, outcome=O.CANNOT_ASSESS, message="needs metric calibration." + hint)
    v = m.value
    verified = m.status == MS.VERIFIED
    if cmp_ in ("required", "forbidden"):
        if v is None:
            return AssessmentItem(**base, outcome=O.CANNOT_ASSESS, message="no value")
        ok = bool(v) if cmp_ == "required" else not bool(v)
        if ok:
            return AssessmentItem(**base, outcome=O.MEETS_THRESHOLD_VERIFIED if verified else O.NO_ISSUE_INDICATED,
                                  message=("verified on site" if verified else
                                           f"{m.name} = {v} (image evidence, unverified)") +
                                  (f". {rule['note']}" if rule.get("note") else ""))
        out = O.POTENTIAL_ISSUE if verified else O.POSSIBLE_ISSUE
        msg = (f"{m.name} = {v}" + ("" if verified else " in image evidence - may be missed by the detector; verify"))
        return AssessmentItem(**base, outcome=out, message=msg + (f". {rule['note']}" if rule.get("note") else ""))
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return AssessmentItem(**base, outcome=O.CANNOT_ASSESS, message="non-numeric value")
    T = float(rule["threshold"])
    s = float(m.uncertainty or 0.0)
    unit = m.unit
    if verified:
        bad = v > T if cmp_ == "max" else v < T
        return AssessmentItem(**base, outcome=O.POTENTIAL_ISSUE if bad else O.MEETS_THRESHOLD_VERIFIED,
                              message=f"manual measurement {_fmt(v, unit)} vs {cmp_} {_fmt(T, unit)}")
    lo, hi = v - K * s, v + K * s
    if cmp_ == "max":
        if m.lower_bound:
            if lo > T:
                return AssessmentItem(**base, outcome=O.POTENTIAL_ISSUE,
                                      message=f"at least {_fmt(v, unit)} (lower bound) exceeds max {_fmt(T, unit)}")
            return AssessmentItem(**base, outcome=O.CANNOT_ASSESS,
                                  message=f"only a lower bound ({_fmt(v, unit)}) is visible; max rule undecidable")
        if lo > T:
            o, why = O.POTENTIAL_ISSUE, "exceeds the maximum even at the low end of the 95% band"
        elif v > T:
            o, why = O.POSSIBLE_ISSUE, "estimate exceeds the maximum (within uncertainty)"
        elif hi > T:
            o, why = O.POSSIBLE_ISSUE, "maximum lies inside the 95% uncertainty band"
        else:
            o, why = O.NO_ISSUE_INDICATED, "below the maximum including uncertainty (unverified)"
    else:
        if hi < T and not m.lower_bound:
            o, why = O.POTENTIAL_ISSUE, "below the minimum even at the high end of the 95% band"
        elif lo >= T:
            o, why = O.NO_ISSUE_INDICATED, "above the minimum including uncertainty (unverified)"
        elif m.lower_bound:
            o, why = O.CANNOT_ASSESS, "only a lower bound is visible and it does not clear the minimum"
        elif v < T:
            o, why = O.POSSIBLE_ISSUE, "estimate is below the minimum (within uncertainty)"
        else:
            o, why = O.POSSIBLE_ISSUE, "minimum lies inside the 95% uncertainty band"
    return AssessmentItem(**base, outcome=o,
                          message=f"{m.name} {_fmt(v, unit)} +- {_fmt(K * s, unit)} (95%) vs {cmp_} {_fmt(T, unit)}: {why}")


def assess(facilities: list[FacilityResult], profile: str | None = None,
           rules_cfg: dict | None = None) -> tuple[list[AssessmentItem], AssessmentSummary]:
    cfg = rules_cfg or load_rules()
    profile = profile or cfg["default_profile"]
    if profile not in cfg["profiles"]:
        raise ValueError(f"unknown rules profile '{profile}'; available: {list(cfg['profiles'])}")
    prof = cfg["profiles"][profile]
    items: list[AssessmentItem] = []
    for fac in facilities:
        for rule in prof["rules"]:
            fcls = rule["facility"] if isinstance(rule["facility"], list) else [rule["facility"]]
            if fac.class_name not in fcls and not (fac.class_name == "elevator" and "door" in fcls):
                continue
            m = next((x for x in fac.measurements if x.name == rule["measurement"]), None)
            it = evaluate_rule(rule, profile, fac, m)
            if it:
                items.append(it)
    counts: dict[str, int] = {}
    for it in items:
        counts[it.outcome.value] = counts.get(it.outcome.value, 0) + 1
    return items, AssessmentSummary(profile=profile, profile_title=prof.get("title", profile), counts=counts)
