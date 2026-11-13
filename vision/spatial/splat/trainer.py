"""Pure-PyTorch 3D Gaussian Splatting (Kerbl et al. 2023) trainer that runs on CUDA, Apple MPS or CPU.

Why in-house: the reference rasterizers (graphdeco diff-gaussian-rasterization, gsplat/Nerfstudio
Splatfacto) are CUDA-only; OpenSplat ships no binaries; the Brush binary (WebGPU/Metal) is supported as
an optional backend but is third-party native code that must be approved before it is executed.

Implemented faithfully but simplified:
  * anisotropic 3D Gaussians (mean, log-scale, quaternion, opacity logit, SH degree-0 colour)
  * EWA projection: Sigma2D = J W Sigma W^T J^T + 0.3 I
  * tile-based (16x16) rasterisation: Gaussians sorted front-to-back per tile; the first K per tile are
    alpha-composited with exact transmittance (K-truncation is the main approximation vs. the reference)
  * loss (1-l) * L1 + l * (1 - SSIM), Adam with the paper's per-parameter learning rates
  * adaptive density control: clone/split by view-space positional gradient, prune by opacity / size
Outputs a standard 3DGS PLY (x,y,z,nx,ny,nz,f_dc_0..2,opacity,scale_0..2,rot_0..3) readable by common
splat viewers (e.g. SuperSplat, antimatter15 splat, Nerfstudio/gsplat loaders).
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vision.spatial.splat.composite import Composite

log = logging.getLogger(__name__)
C0 = 0.28209479177387814  # SH degree-0 constant

PRESETS = {
    # measured on Apple M2 (MPS): ~0.5 s/iteration at 320 px with ~16k Gaussians; time grows with resolution
    # and Gaussian count. tile_k = compositing chunk (all overlapping Gaussians are always composited).
    "fast":     dict(iterations=1000, train_long_side=320, max_gaussians=60_000,  tile_k=64, densify_until=0.6),
    "balanced": dict(iterations=3000, train_long_side=480, max_gaussians=150_000, tile_k=64, densify_until=0.6),
    "high":     dict(iterations=7000, train_long_side=720, max_gaussians=400_000, tile_k=64, densify_until=0.6),
}


@dataclass
class View:
    name: str
    R: np.ndarray          # world -> camera rotation (3x3)
    t: np.ndarray          # world -> camera translation (3,)
    fx: float
    fy: float
    cx: float
    cy: float
    image: np.ndarray      # (H, W, 3) float32 in [0, 1], undistorted, at training resolution
    mask: np.ndarray | None = None   # (H, W) bool, valid pixels after undistortion (None = all valid)


@dataclass
class TrainResult:
    ply_path: str
    num_gaussians: int
    iterations: int
    seconds: float
    device: str
    train_psnr: float | None
    test_psnr: float | None
    test_ssim: float | None
    test_views: list[str]
    loss_curve: list[tuple[int, float]] = field(default_factory=list)
    preview_path: str | None = None


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


class GaussianModel(torch.nn.Module):
    def __init__(self, xyz: np.ndarray, rgb: np.ndarray, device: str):
        super().__init__()
        xyz_t = torch.tensor(xyz, dtype=torch.float32)
        # initial scale = mean distance to 3 nearest neighbours (computed in chunks on CPU)
        d = []
        for i in range(0, len(xyz_t), 4096):
            dd = torch.cdist(xyz_t[i:i + 4096], xyz_t)
            d.append(dd.topk(4, largest=False).values[:, 1:].mean(1))
        dist = torch.cat(d).clamp_min(1e-7)
        n = len(xyz_t)
        self.means = torch.nn.Parameter(xyz_t.to(device))
        self.log_scales = torch.nn.Parameter(torch.log(dist)[:, None].repeat(1, 3).to(device))
        q = torch.zeros(n, 4)
        q[:, 0] = 1
        self.quats = torch.nn.Parameter(q.to(device))
        self.opacity_logit = torch.nn.Parameter(torch.logit(torch.full((n, 1), 0.1)).to(device))
        rgb_t = torch.tensor(rgb, dtype=torch.float32).clamp(0, 1)
        self.f_dc = torch.nn.Parameter(((rgb_t - 0.5) / C0).to(device))

    @property
    def n(self) -> int:
        return self.means.shape[0]


class Rasterizer:
    """k = chunk size for compositing; every overlapping Gaussian is composited (up to max_per_tile)."""
    def __init__(self, tile: int = 16, k: int = 128, device: str = "cpu", max_per_tile: int = 4096):
        self.T, self.K, self.device, self.max_per_tile = tile, k, device, max_per_tile

    def __call__(self, g: GaussianModel, v: View, H: int, W: int, bg: torch.Tensor):
        dev = self.device
        R = torch.tensor(v.R, dtype=torch.float32, device=dev)
        t = torch.tensor(v.t, dtype=torch.float32, device=dev)
        pc = g.means @ R.T + t
        z = pc[:, 2]
        with torch.no_grad():
            vis = z > 0.05
        idx_vis = torch.nonzero(vis).squeeze(1)
        if idx_vis.numel() == 0:
            return bg.expand(H, W, 3).clone(), None, idx_vis
        pc = pc[idx_vis]
        z = pc[:, 2]
        x, y = pc[:, 0], pc[:, 1]
        u = v.fx * x / z + v.cx
        w_ = v.fy * y / z + v.cy
        means2d = torch.stack([u, w_], -1)
        means2d.retain_grad() if means2d.requires_grad else None
        # 3D covariance in world, then camera, then EWA 2D
        Rg = _quat_to_rot(g.quats[idx_vis])
        S = torch.exp(g.log_scales[idx_vis])
        M = Rg * S[:, None, :]
        cov3 = M @ M.transpose(1, 2)
        covc = R @ cov3 @ R.T
        zero = torch.zeros_like(z)
        J = torch.stack([torch.stack([v.fx / z, zero, -v.fx * x / z ** 2], -1),
                         torch.stack([zero, v.fy / z, -v.fy * y / z ** 2], -1)], 1)
        cov2 = J @ covc @ J.transpose(1, 2)
        a = cov2[:, 0, 0] + 0.3
        b = cov2[:, 0, 1]
        c = cov2[:, 1, 1] + 0.3
        det = (a * c - b * b).clamp_min(1e-10)
        conic = torch.stack([c / det, -b / det, a / det], -1)
        opac = torch.sigmoid(g.opacity_logit[idx_vis]).squeeze(1)
        col = (0.5 + C0 * g.f_dc[idx_vis]).clamp(0, 1)
        T = self.T
        ntx, nty = math.ceil(W / T), math.ceil(H / T)
        with torch.no_grad():
            # opacity-aware extent: a Gaussian is invisible where opacity * exp(-r^2/2) < 1/255,
            # and its axis-aligned extent is sqrt(cov_xx) * r, sqrt(cov_yy) * r (tighter than a 3-sigma circle)
            rk = torch.sqrt(2 * torch.log((255 * opac.detach()).clamp_min(1.0)))
            cap = 0.6 * max(W, H)
            ex = (torch.sqrt(a) * rk).clamp(max=cap).cpu()
            ey = (torch.sqrt(c) * rk).clamp(max=cap).cpu()
            m2 = means2d.detach().cpu()
            tx0 = torch.floor((m2[:, 0] - ex) / T).clamp(0, ntx - 1).long()
            tx1 = torch.floor((m2[:, 0] + ex) / T).clamp(0, ntx - 1).long()
            ty0 = torch.floor((m2[:, 1] - ey) / T).clamp(0, nty - 1).long()
            ty1 = torch.floor((m2[:, 1] + ey) / T).clamp(0, nty - 1).long()
            onscreen = ((rk > 0).cpu() & (m2[:, 0] + ex >= 0) & (m2[:, 0] - ex < W) & (m2[:, 1] + ey >= 0)
                        & (m2[:, 1] - ey < H))
            wcount = (tx1 - tx0 + 1) * onscreen
            hcount = (ty1 - ty0 + 1) * onscreen
            counts = wcount * hcount
            total = int(counts.sum())
            gi = torch.repeat_interleave(torch.arange(len(counts)), counts)
            starts = torch.cumsum(counts, 0) - counts
            off = torch.arange(total) - torch.repeat_interleave(starts, counts)
            ww = (tx1 - tx0 + 1)[gi]
            tile = (ty0[gi] + off // ww) * ntx + (tx0[gi] + off % ww)
            zr = torch.argsort(torch.argsort(z.detach().cpu()))           # depth rank
            key = tile * (len(z) + 1) + zr[gi]
            order = torch.argsort(key)
            tile_s, gi_s = tile[order], gi[order]
            first = torch.searchsorted(tile_s, torch.arange(ntx * nty))
            rank = torch.arange(total) - first[tile_s]
            kmax = int(min(self.max_per_tile, max(1, int(rank.max()) + 1 if total else 1)))
            kmax = int(math.ceil(kmax / self.K) * self.K)
            keep = rank < kmax
            pad = torch.full((ntx * nty, kmax), len(z), dtype=torch.long)
            pad[tile_s[keep], rank[keep]] = gi_s[keep]
            pad = pad.to(dev)
            ys, xs = torch.meshgrid(torch.arange(T, device=dev), torch.arange(T, device=dev), indexing="ij")
            tyy = torch.arange(nty, device=dev).repeat_interleave(ntx)
            txx = torch.arange(ntx, device=dev).repeat(nty)
            px = (txx[:, None] * T + xs.reshape(1, -1)).float() + 0.5          # (tiles, P)
            py = (tyy[:, None] * T + ys.reshape(1, -1)).float() + 0.5
        rgb = Composite.apply(means2d, conic, opac, col, pad, px, py, bg, self.K)
        img = rgb.reshape(nty, ntx, T, T, 3).permute(0, 2, 1, 3, 4).reshape(nty * T, ntx * T, 3)[:H, :W]
        return img, means2d, idx_vis


def _ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None]
    k = torch.tensor([math.exp(-(i - 5) ** 2 / (2 * 1.5 ** 2)) for i in range(11)], device=x.device)
    k = (k / k.sum())
    w = (k[:, None] @ k[None, :]).expand(3, 1, 11, 11)
    mu_x, mu_y = F.conv2d(x, w, padding=5, groups=3), F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))).mean()


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(1 / max(mse, 1e-10))


def save_ply(g: GaussianModel, path: Path) -> None:
    n = g.n
    xyz = g.means.detach().cpu().numpy()
    f_dc = g.f_dc.detach().cpu().numpy()
    op = g.opacity_logit.detach().cpu().numpy()
    sc = g.log_scales.detach().cpu().numpy()
    q = F.normalize(g.quats.detach(), dim=-1).cpu().numpy()
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate([xyz, np.zeros((n, 3)), f_dc, op, sc, q], 1).astype(np.float32)
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join(f"property float {p}\n" for p in props) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def load_ply(path: Path) -> dict:
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    head = raw[:end].decode().splitlines()
    n = int(next(l for l in head if l.startswith("element vertex")).split()[-1])
    props = [l.split()[-1] for l in head if l.startswith("property")]
    arr = np.frombuffer(raw[end:], dtype=np.float32).reshape(n, len(props))
    return {p: arr[:, i] for i, p in enumerate(props)}


def train(views: list[View], xyz: np.ndarray, rgb: np.ndarray, out_dir: Path, preset: str = "fast",
          device: str = "cpu", test_every: int = 8, progress=None, seed: int = 0,
          max_seconds: float | None = None) -> TrainResult:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset}; {list(PRESETS)}")
    P = PRESETS[preset]
    torch.manual_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(xyz) < 50:
        raise ValueError(f"only {len(xyz)} sparse points: not enough to initialise Gaussians")
    test_idx = set(range(0, len(views), test_every)) if len(views) >= 2 * test_every else set()
    train_views = [v for i, v in enumerate(views) if i not in test_idx] or views
    test_views = [v for i, v in enumerate(views) if i in test_idx]
    g = GaussianModel(xyz, rgb, device)
    extent = float(np.linalg.norm(xyz - xyz.mean(0), axis=1).max() if len(xyz) else 1.0)
    cams = np.stack([-v.R.T @ v.t for v in views])
    extent = float(max(np.linalg.norm(cams - cams.mean(0), axis=1).max() * 1.1, 1e-3))
    ras = Rasterizer(k=P["tile_k"], device=device)
    bg = torch.zeros(3, device=device)
    imgs = {v.name: torch.tensor(v.image, device=device) for v in views}
    masks = {v.name: (torch.tensor(v.mask, device=device, dtype=torch.float32)[..., None] if v.mask is not None
                      else torch.ones(*v.image.shape[:2], 1, device=device)) for v in views}

    def make_opt():
        return torch.optim.Adam([
            {"params": [g.means], "lr": 1.6e-4 * extent, "name": "means"},
            {"params": [g.log_scales], "lr": 5e-3, "name": "log_scales"},
            {"params": [g.quats], "lr": 1e-3, "name": "quats"},
            {"params": [g.opacity_logit], "lr": 5e-2, "name": "opacity_logit"},
            {"params": [g.f_dc], "lr": 2.5e-3, "name": "f_dc"}], eps=1e-15)
    opt = make_opt()
    grad_acc = torch.zeros(g.n, device=device)
    grad_cnt = torch.zeros(g.n, device=device)
    curve = []
    t0 = time.time()
    it_total = P["iterations"]
    densify_until = int(P["densify_until"] * it_total)
    rng = np.random.default_rng(seed)
    it = 0
    for it in range(1, it_total + 1):
        v = train_views[rng.integers(len(train_views))]
        H, W = v.image.shape[:2]
        img, m2d, idx = ras(g, v, H, W, bg)
        gt = imgs[v.name]
        mk = masks[v.name]
        l1 = ((img - gt).abs() * mk).sum() / (mk.sum() * 3)
        loss = 0.8 * l1 + 0.2 * (1 - _ssim(img * mk, gt * mk))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # means LR decay (exponential to 1% as in the paper)
        frac = it / it_total
        for pg in opt.param_groups:
            if pg["name"] == "means":
                pg["lr"] = 1.6e-4 * extent * (0.01 ** frac)
        opt.step()
        if m2d is not None and m2d.grad is not None and it < densify_until:
            gn = m2d.grad.norm(dim=-1) * 0.5 * max(H, W)  # pixel -> NDC gradient units (as in 3DGS)
            grad_acc[idx] += gn
            grad_cnt[idx] += 1
        if it % 25 == 0:
            curve.append((it, round(float(loss.detach()), 5)))
        if 100 <= it < densify_until and it % 100 == 0:
            g, opt, grad_acc, grad_cnt = _densify(g, opt, make_opt, grad_acc, grad_cnt, extent, P["max_gaussians"],
                                                  prune_only=False)
        if it % 50 == 0 and progress:
            progress(it / it_total, f"iter {it}/{it_total}, loss {float(loss.detach()):.4f}, {g.n} gaussians")
        if max_seconds and time.time() - t0 > max_seconds:
            log.warning("splat training stopped at time budget (%ss) after %d iterations", max_seconds, it)
            break
    # final prune of transparent gaussians
    with torch.no_grad():
        keep = torch.sigmoid(g.opacity_logit).squeeze(1) > 0.005
    g = _subset(g, keep)
    ply = out_dir / "splat.ply"
    save_ply(g, ply)
    # evaluation on held-out frames
    with torch.no_grad():
        def mpsnr(r, v):
            mk = masks[v.name]
            mse = float((((r - imgs[v.name]) ** 2) * mk).sum() / (mk.sum() * 3))
            return 10 * math.log10(1 / max(mse, 1e-10))
        tr = [mpsnr(ras(g, v, *v.image.shape[:2], bg)[0], v) for v in train_views[:: max(1, len(train_views) // 10)]]
        te, ts = [], []
        previews = []
        for v in test_views:
            r = ras(g, v, *v.image.shape[:2], bg)[0]
            te.append(mpsnr(r, v))
            ts.append(float(_ssim(r * masks[v.name], imgs[v.name] * masks[v.name])))
            if len(previews) < 4:
                previews.append(np.concatenate([imgs[v.name].cpu().numpy(), r.clamp(0, 1).cpu().numpy()], 1))
        if not previews:
            v = train_views[0]
            previews.append(np.concatenate([imgs[v.name].cpu().numpy(),
                                            ras(g, v, *v.image.shape[:2], bg)[0].clamp(0, 1).cpu().numpy()], 1))
    prev_path = out_dir / "preview_gt_vs_render.jpg"
    import cv2
    H0 = min(p.shape[0] for p in previews)
    sheet = np.concatenate([p[:H0] for p in previews], 0)
    cv2.imwrite(str(prev_path), cv2.cvtColor((sheet * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    res = TrainResult(ply_path=str(ply), num_gaussians=g.n, iterations=it, seconds=round(time.time() - t0, 1),
                      device=device, train_psnr=round(float(np.mean(tr)), 2) if tr else None,
                      test_psnr=round(float(np.mean(te)), 2) if te else None,
                      test_ssim=round(float(np.mean(ts)), 3) if ts else None,
                      test_views=[v.name for v in test_views], loss_curve=curve, preview_path=str(prev_path))
    (out_dir / "train_metrics.json").write_text(json.dumps(res.__dict__, indent=2))
    return res


def _subset(g: GaussianModel, keep: torch.Tensor) -> GaussianModel:
    for name in ("means", "log_scales", "quats", "opacity_logit", "f_dc"):
        setattr(g, name, torch.nn.Parameter(getattr(g, name).detach()[keep]))
    return g


def _densify(g, opt, make_opt, grad_acc, grad_cnt, extent, max_n, prune_only=False, thresh=0.0002):
    with torch.no_grad():
        avg = grad_acc / grad_cnt.clamp_min(1)
        scales_max = torch.exp(g.log_scales).max(1).values
        big = scales_max > 0.01 * extent
        hot = avg > thresh               # 3DGS default view-space gradient threshold (NDC units)
        room = max(0, max_n - g.n)
        clone = hot & ~big
        split = hot & big
        if clone.sum() + 2 * split.sum() > room:   # keep the hottest within budget
            order = torch.argsort(avg, descending=True)
            allow = torch.zeros_like(hot)
            allow[order[: max(0, room // 2)]] = True
            clone &= allow
            split &= allow
        new = {}
        for name in ("means", "log_scales", "quats", "opacity_logit", "f_dc"):
            new[name] = [getattr(g, name).detach()]
        # clone
        for name in new:
            new[name].append(getattr(g, name).detach()[clone])
        # split into 2 samples drawn from the gaussian, scale / 1.6
        if split.any():
            Rg = _quat_to_rot(g.quats.detach()[split])
            S = torch.exp(g.log_scales.detach()[split])
            for _ in range(2):
                smp = (Rg @ (torch.randn_like(S) * S)[..., None]).squeeze(-1) + g.means.detach()[split]
                new["means"].append(smp)
                new["log_scales"].append(g.log_scales.detach()[split] - math.log(1.6))
                new["quats"].append(g.quats.detach()[split])
                new["opacity_logit"].append(g.opacity_logit.detach()[split])
                new["f_dc"].append(g.f_dc.detach()[split])
        for name in new:
            setattr(g, name, torch.nn.Parameter(torch.cat(new[name])))
        n_old = len(split)
        keep = torch.ones(g.n, dtype=torch.bool, device=g.means.device)
        keep[:n_old][split] = False                                       # originals of split gaussians
        keep &= torch.sigmoid(g.opacity_logit).squeeze(1) > 0.005
        keep &= torch.exp(g.log_scales).max(1).values < 0.5 * extent
    g = _subset(g, keep)
    return g, make_opt(), torch.zeros(g.n, device=g.means.device), torch.zeros(g.n, device=g.means.device)
