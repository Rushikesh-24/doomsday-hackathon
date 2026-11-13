"""Ensemble detector: run several members and keep, per member, only the classes it is
trusted for. Overlaps between members are resolved later by class-wise NMS."""
from __future__ import annotations

import numpy as np

from vision.models.base import Detector, RawDetection


class EnsembleDetector(Detector):
    def __init__(self, *args, members: list[Detector], **kwargs):
        super().__init__(*args, **kwargs)
        self.members = members
        mc = self.spec.get("member_classes", {}) or {}
        self.member_classes = {k: set(v) for k, v in mc.items()}
        # members not listed in member_classes contribute all classes EXCEPT those claimed by others
        self.claimed = set().union(*self.member_classes.values()) if self.member_classes else set()
        self.shared = set(self.spec.get("shared_classes", []) or [])
        self.exclude_prompts = {k: {p.lower() for p in v} for k, v in (self.spec.get("exclude_prompts") or {}).items()}
        self.base_conf = 0.0  # members use their own thresholds

    def load(self) -> None:
        for m in self.members:
            if not hasattr(m, "model"):
                m.load()
        self.device = ",".join(sorted({m.device for m in self.members}))

    def supported_classes(self) -> list[str]:
        s = set()
        for m in self.members:
            s |= set(m.supported_classes())
        return sorted(s)

    def _allowed(self, member_key: str, cname: str, prompt: str = "") -> bool:
        if prompt.lower() in self.exclude_prompts.get(member_key, set()):
            return False
        if member_key in self.member_classes:
            return cname in self.member_classes[member_key]
        return cname not in self.claimed or cname in self.shared

    def _detect(self, image_rgb: np.ndarray, conf: float) -> list[RawDetection]:
        out: list[RawDetection] = []
        for m in self.members:
            # each member applies its own base threshold unless the caller overrides it
            for d in m.detect(image_rgb, conf=conf if conf and conf > 0 else None):
                if self._allowed(m.key, d.class_name, d.prompt):
                    out.append(d)
        return out

    def info(self):
        i = super().info()
        i.name = "+".join(m.info().name for m in self.members)
        i.weights_sha256 = None
        i.revision = ";".join(f"{m.key}:{(m.weights_sha256 or m.revision or '?')[:12]}" for m in self.members)
        return i
