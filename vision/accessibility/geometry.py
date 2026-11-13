"""Pixel-level heuristics used by the attribute layer. All outputs are 2D image-plane
estimates; none of them are metric measurements."""
from __future__ import annotations

import math

import cv2
import numpy as np


def crop(rgb: np.ndarray, xyxy, pad: float = 0.0) -> np.ndarray:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = xyxy
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    x2, y2 = int(min(w, x2 + pw)), int(min(h, y2 + ph))
    return rgb[y1:y2, x1:x2]


def expand(xyxy, fx: float, fy: float | None = None):
    fy = fx if fy is None else fy
    x1, y1, x2, y2 = xyxy
    cw, ch = (x2 - x1) * fx / 2, (y2 - y1) * fy / 2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [cx - cw, cy - ch, cx + cw, cy + ch]


def dominant_line_angle(gray: np.ndarray, max_abs_deg: float = 60.0, min_len_frac: float = 0.25):
    """Length-weighted median angle (degrees, image plane, + = clockwise / descending to the right)
    of long straight segments within ±max_abs_deg of horizontal. Returns (angle, n_lines)."""
    h, w = gray.shape[:2]
    if min(h, w) < 16:
        return None, 0
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=max(12, int(min_len_frac * w)), maxLineGap=8)
    if lines is None:
        return None, 0
    angs, lens = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        if x2 == x1:
            continue
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if a > 90:
            a -= 180
        if a < -90:
            a += 180
        if abs(a) <= max_abs_deg:
            angs.append(a)
            lens.append(math.hypot(x2 - x1, y2 - y1))
    if not angs:
        return None, 0
    order = np.argsort(angs)
    angs, lens = np.array(angs)[order], np.array(lens)[order]
    cum = np.cumsum(lens)
    med = float(angs[np.searchsorted(cum, cum[-1] / 2)])
    return med, len(angs)


def _find_peaks(signal: np.ndarray, min_distance: int, prominence: float) -> list[int]:
    peaks = []
    n = len(signal)
    for i in range(1, n - 1):
        if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]:
            lo, hi = max(0, i - min_distance), min(n, i + min_distance + 1)
            if signal[i] < signal[lo:hi].max():
                continue
            # prominence relative to the lowest point within the window on both sides
            left = signal[max(0, i - 3 * min_distance):i + 1].min()
            right = signal[i:min(n, i + 3 * min_distance + 1)].min()
            if signal[i] - max(left, right) >= prominence:
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
    return peaks


def _longest_consistent_run(spacing: np.ndarray, lo: float = 0.6, hi: float = 1.65) -> tuple[int, int]:
    """Longest contiguous run of spacings whose consecutive ratios stay within [lo, hi]
    (perspective makes step spacing change smoothly; landings/ground/clutter break the run)."""
    best, start = (0, 0), 0
    for i in range(1, len(spacing) + 1):
        if i == len(spacing) or not (lo <= spacing[i] / max(spacing[i - 1], 1e-6) <= hi):
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = i
    return best


def estimate_step_edges(rgb_crop: np.ndarray) -> dict:
    """Estimate the number of visible step edges (nosings) in a staircase crop.

    Method: row profile of vertical-gradient energy (horizontal edges) over the central 60% of
    columns, detrended; peaks are found and the longest run of peaks with smoothly varying
    spacing (perspective-consistent) is taken as the flight of steps. Counts *visible* edges:
    occlusion, carpets, shadows and steep/side viewpoints bias the count.
    """
    if rgb_crop.size == 0 or rgb_crop.shape[0] < 150 or rgb_crop.shape[1] < 24:
        return {"count": None, "confidence": 0.0, "reason": "staircase region too small (<150px tall) to resolve steps"}
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
    H = 480
    gray = cv2.resize(gray, (max(16, int(gray.shape[1] * H / gray.shape[0])), H), interpolation=cv2.INTER_AREA)
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    w = gray.shape[1]
    prof = gy[:, int(0.2 * w):max(int(0.8 * w), int(0.2 * w) + 1)].mean(axis=1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 9), 2).ravel()
    sig = prof - cv2.blur(prof.reshape(-1, 1), (1, 61)).ravel()  # remove illumination trend
    if sig.std() < 1e-6:
        return {"count": 0, "confidence": 0.0, "reason": "no horizontal structure"}
    peaks = _find_peaks(sig, min_distance=6, prominence=0.35 * sig.std())
    if len(peaks) < 3:
        return {"count": len(peaks), "confidence": 0.1, "reason": "fewer than 3 edges"}
    # Each step may yield one edge (nosing) or two (nosing + riser base), giving alternating
    # short/long spacings. Evaluate all peaks and every-other peak; keep the best-supported run.
    best = None
    for cand in (peaks, peaks[0::2], peaks[1::2]):
        if len(cand) < 3:
            continue
        spacing = np.diff(cand).astype(float)
        a, b = _longest_consistent_run(spacing)
        run = spacing[a:b]
        edges = len(run) + 1
        coverage = (cand[b] - cand[a]) / H
        lr = np.log(run[1:] / run[:-1]) if len(run) > 1 else np.zeros(1)
        regularity = float(np.clip(1.0 - np.std(lr) / 0.35, 0.0, 1.0))
        conf = float(np.clip(edges / 5, 0, 1)) * float(np.clip(coverage / 0.5, 0, 1)) * regularity
        if best is None or conf > 1.3 * best[0] + 0.05:  # prefer all-peaks unless every-other is clearly better
            best = (conf, edges, coverage, regularity)
    if best is None:
        return {"count": len(peaks), "confidence": 0.1, "reason": "too few edges"}
    conf, edges, coverage, regularity = best
    return {"count": edges, "confidence": round(conf, 3), "run_coverage": round(float(coverage), 3),
            "spacing_regularity": round(regularity, 3), "peaks_total": len(peaks)}


def yellow_fraction(rgb_crop: np.ndarray) -> float:
    if rgb_crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2HSV)
    m = (hsv[..., 0] >= 18) & (hsv[..., 0] <= 38) & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 90)
    return float(m.mean())
