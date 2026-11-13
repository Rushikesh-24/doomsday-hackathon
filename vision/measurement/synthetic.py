"""Analytic ray-cast scenes with exact metric ground truth (ramp, door, stairs, corridor).

Used to (1) unit-test the geometry/status/rules logic without network weights and (2) separate
geometric-algorithm error from depth-model error during validation. These are NOT real photos and
say nothing about monocular-depth accuracy on real campus images.
World frame: X right, Y up, Z forward (metres). Camera at (0, cam_h, 0), pitched down by pitch_deg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from vision.schemas import BBox, Detection, ImageInfo, ImageResult, ModelInfo, SourceRef, Timing


@dataclass
class Scene:
    w: int = 960
    h: int = 720
    hfov: float = 65.0
    cam_h: float = 1.5
    pitch_deg: float = 8.0
    prims: list = field(default_factory=list)   # (name, label, kind, params)

    @property
    def f(self) -> float:
        return (self.w / 2) / math.tan(math.radians(self.hfov) / 2)

    def rays_world(self):
        u, v = np.meshgrid(np.arange(self.w) + 0.5, np.arange(self.h) + 0.5)
        dc = np.stack([(u - self.w / 2) / self.f, (v - self.h / 2) / self.f, np.ones_like(u)], -1)  # camera, y down
        dw = dc * np.array([1, -1, 1])                                                              # y up
        p = math.radians(self.pitch_deg)
        R = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])     # pitch down
        return dw @ R.T, dc

    def render(self):
        dw, dc = self.rays_world()
        o = np.array([0, self.cam_h, 0.0])
        H, W = self.h, self.w
        tbest = np.full((H, W), np.inf)
        inst = np.full((H, W), -1)
        shade = np.zeros((H, W, 3))
        for i, (name, label, kind, P) in enumerate(self.prims):
            n = np.asarray(P["n"], float)
            n /= np.linalg.norm(n)
            den = dw @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (P["d"] - o @ n) / den
            X = o + dw * t[..., None]
            ok = (t > 0) & np.isfinite(t)
            for ax, (lo, hi) in P.get("bounds", {}).items():
                k = "xyz".index(ax)
                ok &= (X[..., k] >= lo - 1e-9) & (X[..., k] <= hi + 1e-9)
            if "fn" in P:
                ok &= P["fn"](X)
            better = ok & (t < tbest)
            tbest[better] = t[better]
            inst[better] = i
            shade[better] = np.asarray(P.get("color", (128, 128, 128)), float)
        # rays are the unnormalised camera rays (z-component 1), so the ray parameter t IS the z-depth
        z = np.where(np.isfinite(tbest), tbest, 1e3)
        rng = np.random.default_rng(0)
        rgb = np.clip(shade + rng.normal(0, 3, shade.shape), 0, 255).astype(np.uint8)
        labels = {i: p[1] for i, p in enumerate(self.prims)}
        return rgb, z.astype(np.float32), inst, labels

    def K(self) -> dict:
        return {"fx": self.f, "fy": self.f, "cx": self.w / 2, "cy": self.h / 2, "focal_rel_sigma": 0.005}

    def project(self, Xw: np.ndarray) -> np.ndarray:
        p = math.radians(self.pitch_deg)
        R = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        d = (np.atleast_2d(Xw) - np.array([0, self.cam_h, 0])) @ R      # world -> camera (y up)
        d = d * np.array([1, -1, 1])
        return np.stack([d[:, 0] / d[:, 2] * self.f + self.w / 2, d[:, 1] / d[:, 2] * self.f + self.h / 2], 1)


def plane(name, label, n, d, bounds, color, fn=None):
    P = {"n": n, "d": d, "bounds": bounds, "color": color}
    if fn:
        P["fn"] = fn
    return (name, label, "plane", P)


def ground(color=(120, 120, 120)):
    return plane("ground", "floor", (0, 1, 0), 0.0, {}, color)


def back_wall(z, color=(200, 190, 170)):
    return plane("wall", "wall", (0, 0, 1), z, {}, color)


def ramp_scene(slope_deg=4.76, length=4.0, width=1.5, z0=3.0, x0=-0.75) -> tuple[Scene, dict]:
    s = Scene(pitch_deg=12)
    t = math.tan(math.radians(slope_deg))
    # surface: Y = (Z - z0) * t  ->  n = (0, 1, -t), d = -z0*t
    run = length * math.cos(math.radians(slope_deg))
    s.prims += [ground(), back_wall(z0 + run + 4),
                plane("ramp", "path", (0, 1, -t), -z0 * t, {"x": (x0, x0 + width), "z": (z0, z0 + run)}, (170, 150, 120))]
    gt = {"running_slope_deg": slope_deg, "running_slope_percent": 100 * t, "ramp_length": length,
          "clear_width": width, "rise": length * math.sin(math.radians(slope_deg))}
    return s, gt


def door_scene(width=0.9, height=2.1, z=4.0, x0=-0.3) -> tuple[Scene, dict]:
    s = Scene(pitch_deg=3, cam_h=1.4)
    s.prims += [ground(), back_wall(z + 0.002),
                plane("door", "door", (0, 0, 1), z, {"x": (x0, x0 + width), "y": (0, height)}, (90, 60, 40))]
    return s, {"doorway_width": width, "door_height": height}


def stairs_scene(n=8, riser=0.16, tread=0.30, width=1.6, z0=3.0, x0=-0.8) -> tuple[Scene, dict]:
    s = Scene(pitch_deg=5, cam_h=1.55, w=960, h=1280, hfov=60)
    s.prims += [ground(), back_wall(z0 + n * tread + 3)]
    for k in range(n):
        zk = z0 + k * tread
        s.prims.append(plane(f"riser{k}", "stairs", (0, 0, 1), zk, {"x": (x0, x0 + width), "y": (k * riser, (k + 1) * riser)},
                             (70, 70, 80)))
        s.prims.append(plane(f"tread{k}", "stairs", (0, 1, 0), (k + 1) * riser,
                             {"x": (x0, x0 + width), "z": (zk, zk + tread)}, (210, 205, 195)))
    # top landing
    s.prims.append(plane("landing", "floor", (0, 1, 0), n * riser, {"x": (x0 - 1, x0 + width + 1), "z": (z0 + n * tread, z0 + n * tread + 3)},
                         (150, 150, 150)))
    return s, {"step_count": n, "riser_height": riser, "tread_depth": tread, "stair_width": width,
               "pitch_deg": math.degrees(math.atan(riser / tread))}


def corridor_scene(width=1.8, length=12.0, obstacle=None) -> tuple[Scene, dict]:
    s = Scene(pitch_deg=6, cam_h=1.5)
    hw = width / 2
    s.prims += [plane("floor", "floor", (0, 1, 0), 0.0, {"x": (-hw, hw), "z": (0, length)}, (150, 140, 120)),
                plane("wl", "wall", (1, 0, 0), -hw, {"y": (-0.01, 3)}, (220, 220, 210)),
                plane("wr", "wall", (1, 0, 0), hw, {"y": (-0.01, 3)}, (215, 215, 200)),
                back_wall(length)]
    gt = {"clear_width": width}
    if obstacle:
        x0, x1, z0, h = obstacle
        s.prims.append(plane("box", "box", (0, 0, 1), z0, {"x": (x0, x1), "y": (0, h)}, (40, 120, 40)))
        gt["min_available_clearance"] = max(x0 + hw, hw - x1)
    return s, gt


def detections_for(scene: Scene, inst: np.ndarray, groups: dict[str, list[str]], name: str = "synthetic.png") -> ImageResult:
    """Build an ImageResult with one detection per group (class -> primitive names)."""
    dets = []
    names = [p[0] for p in scene.prims]
    for i, (cls, prims) in enumerate(groups.items()):
        cls_name = cls.split("#")[0]
        idx = [names.index(p) for p in prims]
        m = np.isin(inst, idx)
        ys, xs = np.nonzero(m)
        box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        dets.append(Detection(id=f"syn-{i:03d}", class_name=cls_name, category="facility", confidence=0.9,
                              bbox=BBox.from_xyxy(box, scene.w, scene.h), source=SourceRef(name=name),
                              prompt=cls_name, model_name="synthetic", model_version="0", timestamp="t"))
    return ImageResult(request_id="syn", image=ImageInfo(name=name, width=scene.w, height=scene.h), thresholds={},
                       model=ModelInfo(key="synthetic", name="synthetic", version="0"), detections=dets,
                       counts_by_class={}, timing=Timing(preprocess_ms=0, inference_ms=0, postprocess_ms=0,
                                                          attributes_ms=0, total_ms=0), processed_at="t")


class FakeSemantic:
    """Semantic backend stand-in that returns the renderer's ground-truth labels."""
    def __init__(self, inst, labels):
        self.inst, self.labels = inst, labels
        names = sorted(set(labels.values()))
        self.id2label = {i: n for i, n in enumerate(names)}
        self._lut = {k: names.index(v) for k, v in labels.items()}

    def predict(self, rgb):
        seg = np.full(self.inst.shape, -1, np.int32)
        for k, v in self._lut.items():
            seg[self.inst == k] = v
        if seg.shape != rgb.shape[:2]:
            import cv2
            seg = cv2.resize(seg, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        return seg

    def mask_of(self, seg, names):
        ids = [i for i, n in self.id2label.items() if n in names]
        return np.isin(seg, ids)

    def ref(self):
        from vision.measurement.schemas import ModelRef
        return ModelRef(role="semantic_segmentation", key="gt_labels", name="synthetic ground truth", version="0")


class FakeBoxSegmenter:
    """SAM stand-in: returns the ground-truth instance mask that best fills each box."""
    def __init__(self, inst, groups_idx: list[list[int]]):
        self.inst, self.groups = inst, groups_idx

    def predict(self, rgb, boxes):
        out = []
        for b in boxes:
            best, bscore = np.zeros(rgb.shape[:2], bool), 0
            for g in self.groups:
                m = np.isin(self.inst, g)
                if m.shape != rgb.shape[:2]:
                    import cv2
                    m = cv2.resize(m.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                x1, y1, x2, y2 = (int(v) for v in b)
                sc = m[y1:y2, x1:x2].mean() if y2 > y1 and x2 > x1 else 0
                if sc > bscore:
                    best, bscore = m, sc
            out.append((best, 0.99))
        return out

    def ref(self):
        from vision.measurement.schemas import ModelRef
        return ModelRef(role="box_segmenter", key="gt_masks", name="synthetic ground truth", version="0")


class FakeDepth:
    """Depth stand-in returning the renderer's exact depth (optionally globally mis-scaled, like a
    monocular network whose metric scale is wrong)."""
    def __init__(self, z, scale_error=1.0):
        self.z, self.k = z, scale_error

    def predict(self, rgb):
        import cv2
        from vision.measurement.backends import DepthResult
        from vision.measurement.schemas import ModelRef
        z = self.z * self.k
        if z.shape != rgb.shape[:2]:
            z = cv2.resize(z, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        return DepthResult(z.astype(np.float32), None, ModelRef(role="depth", key="gt_depth",
                                                               name="synthetic ground truth", version="0"), 0.0)


class FakeModels:
    def __init__(self, inst, labels, groups_idx, z=None, scale_error=1.0):
        self._sem = FakeSemantic(inst, labels)
        self._sam = FakeBoxSegmenter(inst, groups_idx)
        self._depth = FakeDepth(z, scale_error) if z is not None else None

    def semantic(self):
        return self._sem

    def box_segmenter(self):
        return self._sam

    def depth(self, key):
        if self._depth is None:
            raise RuntimeError("synthetic scenes supply depth_override")
        return self._depth

    def describe(self):
        return {}


def build(scene_fn, groups: dict[str, list[str]], scale_error: float = 1.0, **kw):
    """Render a scene; return (png_bytes, depth, detections ImageResult, FakeModels, gt, scene)."""
    import io
    from PIL import Image
    scene, gt = scene_fn(**kw)
    rgb, z, inst, labels = scene.render()
    names = [p[0] for p in scene.prims]
    res = detections_for(scene, inst, groups)
    gidx = [[names.index(p) for p in prims] for prims in groups.values()]
    b = io.BytesIO()
    Image.fromarray(rgb).save(b, format="PNG")
    return b.getvalue(), z, res, FakeModels(inst, labels, gidx, z, scale_error), gt, scene
