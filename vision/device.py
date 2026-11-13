"""Compute device selection: CUDA > Apple MPS > CPU."""
from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def select_device(preference: str = "auto") -> str:
    pref = (preference or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if pref in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if pref not in ("auto", "cpu"):
        log.warning("Requested device '%s' not available, falling back to CPU", pref)
    return "cpu"


def device_report() -> dict:
    return {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
    }
