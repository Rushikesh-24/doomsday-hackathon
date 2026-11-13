"""Additional open-source AI capabilities (all native HF transformers, no remote code, licences verified):

  Florence-2-base   florence-community/Florence-2-base   MIT         captions, OCR (+regions) -> reads signboards
  Whisper-base      openai/whisper-base                  Apache-2.0  multilingual speech -> navigation intent
  DINOv2-small      facebook/dinov2-small                Apache-2.0  global image descriptors -> visual
                                                                     localisation against reconstruction keyframes
Each model is loaded once (lazily) and falls back to CPU if an accelerator op fails.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vision.config import get_settings
from vision.device import select_device
from vision.models.hf_models import _resolve_hf

MODELS = {
    "florence2": {"hf_id": "florence-community/Florence-2-base", "license": "MIT", "task": "vision-language"},
    "whisper": {"hf_id": "openai/whisper-base", "license": "Apache-2.0", "task": "speech recognition"},
    "dinov2": {"hf_id": "facebook/dinov2-small", "license": "Apache-2.0", "task": "image embedding"},
}
_cache: dict = {}
_lock = threading.RLock()


def _load(key: str):
    with _lock:
        if key in _cache:
            return _cache[key]
        s = get_settings()
        path, rev = _resolve_hf(MODELS[key]["hf_id"], s.offline, key)
        dev = select_device(s.device)
        t = time.time()
        if key == "florence2":
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
            proc, model = AutoProcessor.from_pretrained(path), Florence2ForConditionalGeneration.from_pretrained(path)
        elif key == "whisper":
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            proc, model = WhisperProcessor.from_pretrained(path), WhisperForConditionalGeneration.from_pretrained(path)
        else:
            from transformers import AutoImageProcessor, AutoModel
            proc, model = AutoImageProcessor.from_pretrained(path), AutoModel.from_pretrained(path)
        try:
            model.to(dev)
        except Exception:
            dev = "cpu"
        model.eval()
        _cache[key] = {"proc": proc, "model": model, "device": dev, "revision": rev,
                       "load_s": round(time.time() - t, 2), "lock": threading.Lock()}
        return _cache[key]


def status() -> dict:
    return {k: {**v, "loaded": k in _cache, "device": _cache[k]["device"] if k in _cache else None} for k, v in MODELS.items()}


def _run(entry, fn):
    with entry["lock"], torch.inference_mode():
        try:
            return fn(entry["device"])
        except (RuntimeError, NotImplementedError):
            if entry["device"] == "cpu":
                raise
            entry["device"] = "cpu"
            entry["model"].to("cpu")
            return fn("cpu")


# ------------------------------------------------------------------ Florence-2
TASKS = {"caption": "<CAPTION>", "detailed_caption": "<MORE_DETAILED_CAPTION>", "ocr": "<OCR>",
         "ocr_regions": "<OCR_WITH_REGION>"}


def florence(image: Image.Image, task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"task must be one of {list(TASKS)}")
    e = _load("florence2")
    tok = TASKS[task]
    t = time.perf_counter()

    def fn(dev):
        inp = e["proc"](text=tok, images=image, return_tensors="pt").to(dev, torch.float32)
        ids = e["model"].generate(**inp, max_new_tokens=256, num_beams=3)
        txt = e["proc"].batch_decode(ids, skip_special_tokens=False)[0]
        return e["proc"].post_process_generation(txt, task=tok, image_size=image.size)[tok]
    out = _run(e, fn)
    res = {"task": task, "model": MODELS["florence2"]["hf_id"], "ms": round((time.perf_counter() - t) * 1000)}
    if task == "ocr_regions":
        res["regions"] = [{"text": l.replace("</s>", "").strip(), "quad": [round(float(v), 1) for v in q]}
                          for l, q in zip(out.get("labels", []), out.get("quad_boxes", []))]
        res["text"] = " ".join(r["text"] for r in res["regions"])
    else:
        res["text"] = str(out).strip()
    return res


def read_signs(source: Image.Image, detections: list[dict], pad: float = 0.08) -> list[dict]:
    """OCR each signboard / accessibility-symbol detection crop (fills the 'text_content' attribute)."""
    out = []
    W, H = source.size
    for d in detections:
        if d["class_name"] not in ("signboard", "accessibility_symbol", "entrance_exit"):
            continue
        b = d["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
        crop = source.crop((max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph)))
        if min(crop.size) < 16:
            out.append({"detection_id": d["id"], "class_name": d["class_name"], "text": None,
                        "status": "not_determinable", "reason": "crop too small to read"})
            continue
        r = florence(crop, "ocr")
        txt = re.sub(r"\s+", " ", r["text"]).strip()
        out.append({"detection_id": d["id"], "class_name": d["class_name"], "text": txt or None,
                    "status": "detected" if txt else "not_detected", "method": "Florence-2 OCR on detection crop",
                    "ms": r["ms"]})
    return out


# ------------------------------------------------------------------ Whisper
INTENTS = {   # destination keywords (several languages) -> facility class
    "elevator": ["lift", "elevator", "لفٹ", "لِفٹ", "लिफ्ट", "एलिवेटर", "ascensor", "ascenseur", "aufzug", "elevador", "लिफ़्ट"],
    "staircase": ["stairs", "staircase", "steps", "سیڑھی", "سیڑھیاں", "सीढ़ी", "सीढ़ियाँ", "सीढ़ियां", "escaleras", "escalier", "treppe"],
    "ramp": ["ramp", "ریمپ", "रैंप", "rampa", "rampe"],
    "entrance_exit": ["exit", "entrance", "door out", "باہر", "निकास", "प्रवेश", "salida", "entrada", "sortie", "ausgang"],
    "door": ["door", "दरवाज़ा", "दरवाजा", "puerta", "porte", "tür"],
    "handrail": ["handrail", "railing", "रेलिंग", "barandilla"],
    "walkway": ["walkway", "path", "sidewalk", "रास्ता", "camino"],
}


def _decode_audio(data: bytes) -> np.ndarray:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("ffmpeg is required to decode audio")
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(data)
        f.flush()
        r = subprocess.run([ff, "-v", "error", "-i", f.name, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                           capture_output=True, timeout=120)
    if r.returncode != 0 or not r.stdout:
        raise ValueError("could not decode audio (send wav/mp3/m4a/webm/ogg)")
    a = np.frombuffer(r.stdout, dtype=np.float32)
    if len(a) < 1600:
        raise ValueError("audio shorter than 0.1 s")
    return a[: 16000 * 30]


def transcribe(data: bytes, language: str | None = None) -> dict:
    """language: ISO code hint (e.g. 'hi', 'en'); None = auto-detect (weak for Hindi with whisper-base)."""
    audio = _decode_audio(data)
    e = _load("whisper")
    t = time.perf_counter()

    def fn(dev):
        feats = e["proc"](audio, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
        lang = language
        if not lang and hasattr(e["model"], "detect_language"):
            tok = e["model"].detect_language(feats)
            lang = e["proc"].tokenizer.decode(tok[0]).strip("<|>") if tok is not None else None
        kw = {"task": "transcribe", "max_new_tokens": 120}
        if lang:
            kw["language"] = lang
        ids = e["model"].generate(feats, **kw)
        return lang, e["proc"].batch_decode(ids, skip_special_tokens=True)[0].strip()
    detected, text = _run(e, fn)
    lang = None
    low = text.lower()
    hits = [(cls, kw) for cls, kws in INTENTS.items() for kw in kws if kw in low]
    return {"text": text, "language": detected, "language_source": "user" if language else "auto",
            "seconds": round(len(audio) / 16000, 2),
            "intent": {"destination_class": hits[0][0], "keyword": hits[0][1]} if hits else None,
            "model": MODELS["whisper"]["hf_id"], "ms": round((time.perf_counter() - t) * 1000)}


# ------------------------------------------------------------------ DINOv2 localisation
def _embed(images: list[Image.Image]) -> np.ndarray:
    e = _load("dinov2")

    def fn(dev):
        inp = e["proc"](images=images, return_tensors="pt").to(dev)
        out = e["model"](**inp)
        v = torch.cat([out.last_hidden_state[:, 0], out.last_hidden_state[:, 1:].mean(1)], 1)   # CLS + mean patch
        return torch.nn.functional.normalize(v, dim=1).cpu().numpy()
    return _run(e, fn)


def keyframe_index(job_dir: Path) -> tuple[list[dict], np.ndarray]:
    cache = job_dir / "dinov2_index.npz"
    traj = json.loads((job_dir / "trajectory.json").read_text())["cameras"]
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        if list(d["names"]) == [c["frame"] for c in traj]:
            return traj, d["emb"]
    embs = []
    for i in range(0, len(traj), 16):
        ims = [Image.open(job_dir / "frames" / c["frame"]).convert("RGB") for c in traj[i:i + 16]]
        embs.append(_embed(ims))
    E = np.concatenate(embs) if embs else np.zeros((0, 768))
    np.savez(cache, names=np.array([c["frame"] for c in traj]), emb=E)
    return traj, E


def localize(image: Image.Image, job_dir: Path, k: int = 5) -> dict:
    t = time.perf_counter()
    traj, E = keyframe_index(job_dir)
    if not len(E):
        raise ValueError("reconstruction has no registered keyframes")
    q = _embed([image.convert("RGB")])[0]
    sim = E @ q
    order = np.argsort(-sim)[:k]
    top = [{"frame": traj[i]["frame"], "similarity": round(float(sim[i]), 4), "position_local": traj[i]["center_local"],
            "node": f"cam{i}"} for i in order]
    w = np.exp((sim[order] - sim[order[0]]) / 0.02)
    pos = (np.array([traj[i]["center_local"] for i in order]) * w[:, None]).sum(0) / w.sum()
    margin = float(sim[order[0]] - np.median(sim))
    # thresholds checked on the stairway job: a frame from a part of the walk that is NOT in the
    # reconstruction still scored 0.70 (look-alike stairs) -> only near-duplicate views count as reliable
    conf = "high" if sim[order[0]] > 0.9 else "medium" if sim[order[0]] > 0.8 else "low"
    return {"best_node": top[0]["node"], "estimated_position_local": pos.round(4).tolist(), "matches": top,
            "confidence": conf, "similarity_margin": round(margin, 4), "model": MODELS["dinov2"]["hf_id"],
            "method": "DINOv2 global descriptor retrieval over registered keyframes (keyframe-level, not 6-DoF)",
            "ms": round((time.perf_counter() - t) * 1000)}
