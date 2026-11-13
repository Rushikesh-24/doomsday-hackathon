"""Hugging Face transformers backends: Grounding DINO and OWLv2 (both Apache-2.0)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vision.models.base import Detector, ModelUnavailableError, RawDetection

log = logging.getLogger(__name__)


def _resolve_hf(repo_id: str, offline: bool, key: str) -> tuple[str, str | None]:
    """Download (or find cached) snapshot of the official HF repo. Returns (path, commit)."""
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(repo_id, local_files_only=offline,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    except Exception as e:
        raise ModelUnavailableError(key, f"cannot obtain {repo_id} ({'offline, not cached' if offline else e})") from e
    commit = Path(path).name if "snapshots" in path else None
    return path, commit


class _HFDetector(Detector):
    def _move(self) -> None:
        try:
            self.model.to(self.device)
        except Exception as e:  # some ops unsupported on MPS in older torch
            log.warning("%s: cannot use %s (%s); falling back to CPU", self.key, self.device, e)
            self.device = "cpu"
            self.model.to("cpu")
        self.model.eval()

    def _run(self, fn):
        """Run `fn()`; if it fails on an accelerator, retry permanently on CPU."""
        try:
            with torch.inference_mode():
                return fn()
        except (RuntimeError, NotImplementedError) as e:
            if self.device == "cpu":
                raise
            log.warning("%s: inference failed on %s (%s); switching to CPU", self.key, self.device, e)
            self.device = "cpu"
            self.model.to("cpu")
            with torch.inference_mode():
                return fn()


class GroundingDinoDetector(_HFDetector):
    def load(self) -> None:
        t = time.time()
        path, self.revision = _resolve_hf(self.spec["hf_id"], self.offline, self.key)
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(path)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path)
        except Exception as e:
            raise ModelUnavailableError(self.key, f"failed to load Grounding DINO: {e}") from e
        self._move()
        self.text_threshold = float(self.spec.get("text_threshold", 0.25))
        n = int(self.spec.get("prompts_per_pass", 12))
        prompts = self.taxonomy.all_prompts()
        self.chunks = [prompts[i:i + n] for i in range(0, len(prompts), n)]
        self.load_seconds = time.time() - t

    @staticmethod
    def _match(phrase: str, chunk: list[str]) -> str | None:
        """Grounding DINO returns the text span(s) that matched, which can be a fragment
        or a merge of adjacent prompts. Map it back to the best prompt by token overlap."""
        phrase = phrase.lower().strip()
        if phrase in chunk:
            return phrase
        ptoks = set(phrase.split())
        best, best_score = None, 0.0
        for p in chunk:
            toks = set(p.split())
            score = len(ptoks & toks) / max(len(toks), 1)
            if score > best_score:
                best, best_score = p, score
        return best if best_score >= 0.5 else None

    def _detect(self, image_rgb: np.ndarray, conf: float) -> list[RawDetection]:
        img = Image.fromarray(image_rgb)
        h, w = image_rgb.shape[:2]
        out: list[RawDetection] = []
        for chunk in self.chunks:
            chunk_l = [p.lower() for p in chunk]
            text = ". ".join(chunk_l) + "."

            def fn():
                inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                return self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=conf, text_threshold=self.text_threshold,
                    target_sizes=[(h, w)])[0]

            r = self._run(fn)
            labels = r.get("text_labels") or r.get("labels")
            for box, score, lab in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), labels):
                prompt = self._match(str(lab), chunk_l)
                cname = self.taxonomy.class_for_prompt(prompt) if prompt else None
                if cname:
                    out.append(RawDetection(box, float(score), cname, prompt, self.key))
        return out


class Owlv2Detector(_HFDetector):
    def load(self) -> None:
        t = time.time()
        path, self.revision = _resolve_hf(self.spec["hf_id"], self.offline, self.key)
        try:
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            self.processor = Owlv2Processor.from_pretrained(path)
            self.model = Owlv2ForObjectDetection.from_pretrained(path)
        except Exception as e:
            raise ModelUnavailableError(self.key, f"failed to load OWLv2: {e}") from e
        self._move()
        self.prompts = self.taxonomy.all_prompts()
        self.queries = [f"a photo of a {p}" for p in self.prompts]
        self.load_seconds = time.time() - t

    def _detect(self, image_rgb: np.ndarray, conf: float) -> list[RawDetection]:
        img = Image.fromarray(image_rgb)
        h, w = image_rgb.shape[:2]
        side = max(h, w)  # OWLv2 pads to a square (bottom/right) before resizing

        def fn():
            inputs = self.processor(text=[self.queries], images=img, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            return self.processor.post_process_grounded_object_detection(
                outputs=outputs, threshold=conf, target_sizes=[(side, side)])[0]

        r = self._run(fn)
        out = []
        for box, score, li in zip(r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), r["labels"].cpu().numpy()):
            prompt = self.prompts[int(li)]
            cname = self.taxonomy.class_for_prompt(prompt)
            box = np.clip(box, 0, [w, h, w, h])
            if cname and box[2] > box[0] and box[3] > box[1]:
                out.append(RawDetection(box, float(score), cname, prompt, self.key))
        return out
