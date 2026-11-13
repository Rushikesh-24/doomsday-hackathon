"""Backend-independent evaluation of any 3DGS PLY (SH degree 0-3) on held-out views, using the same
rasterizer, views and validity masks for every backend so PSNR/SSIM are directly comparable."""
from __future__ import annotations

import math
import re

import numpy as np
import torch

from vision.spatial.splat.trainer import C0, GaussianModel, Rasterizer, _ssim, load_ply

C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
      1.445305721320277, -0.5900435899266435]


def sh_colour(dc: np.ndarray, rest: np.ndarray, deg: int, dirs: np.ndarray) -> np.ndarray:
    """Standard 3DGS real SH evaluation. dc (N,3), rest (N,K,3), dirs (N,3) unit view directions."""
    x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
    r = C0 * dc
    if deg >= 1:
        r = r - C1 * y * rest[:, 0] + C1 * z * rest[:, 1] - C1 * x * rest[:, 2]
    if deg >= 2:
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        r = (r + C2[0] * xy * rest[:, 3] + C2[1] * yz * rest[:, 4] + C2[2] * (2 * zz - xx - yy) * rest[:, 5]
             + C2[3] * xz * rest[:, 6] + C2[4] * (xx - yy) * rest[:, 7])
    if deg >= 3:
        r = (r + C3[0] * y * (3 * xx - yy) * rest[:, 8] + C3[1] * xy * z * rest[:, 9]
             + C3[2] * y * (4 * zz - xx - yy) * rest[:, 10] + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * rest[:, 11]
             + C3[4] * x * (4 * zz - xx - yy) * rest[:, 12] + C3[5] * z * (xx - yy) * rest[:, 13]
             + C3[6] * x * (xx - 3 * yy) * rest[:, 14])
    return r + 0.5


def load_any(path) -> dict:
    p = load_ply(path)
    n_rest = len([k for k in p if k.startswith("f_rest_")])
    deg = {0: 0, 9: 1, 24: 2, 45: 3}.get(n_rest)
    if deg is None:
        raise ValueError(f"unsupported f_rest count {n_rest}")
    rest = np.zeros((len(p["x"]), max(n_rest // 3, 1), 3), np.float32)
    if n_rest:
        # 3DGS layout: f_rest_{c * K + k}, channel-major
        K = n_rest // 3
        for c in range(3):
            for k in range(K):
                rest[:, k, c] = p[f"f_rest_{c * K + k}"]
    return {"xyz": np.stack([p["x"], p["y"], p["z"]], 1), "dc": np.stack([p["f_dc_0"], p["f_dc_1"], p["f_dc_2"]], 1),
            "rest": rest, "deg": deg, "opacity": p["opacity"],
            "scale": np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], 1),
            "rot": np.stack([p["rot_0"], p["rot_1"], p["rot_2"], p["rot_3"]], 1), "n": len(p["x"])}


def evaluate(ply_path, views, device: str = "cpu", previews: int = 0) -> dict:
    d = load_any(ply_path)
    g = GaussianModel.__new__(GaussianModel)
    torch.nn.Module.__init__(g)
    t = lambda a: torch.nn.Parameter(torch.tensor(a, dtype=torch.float32, device=device), requires_grad=False)
    g.means, g.log_scales, g.quats = t(d["xyz"]), t(d["scale"]), t(d["rot"])
    g.opacity_logit = t(d["opacity"][:, None])
    ras = Rasterizer(k=64, device=device)
    bg = torch.zeros(3, device=device)
    ps, ss, imgs = [], [], []
    with torch.no_grad():
        for v in views:
            cam = -v.R.T @ v.t
            dirs = d["xyz"] - cam
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
            col = sh_colour(d["dc"], d["rest"], d["deg"], dirs)
            g.f_dc = t((col - 0.5) / C0)            # trainer renders 0.5 + C0 * f_dc (clamped)
            H, W = v.image.shape[:2]
            r = ras(g, v, H, W, bg)[0].clamp(0, 1)
            gt = torch.tensor(v.image, device=device)
            mk = (torch.tensor(v.mask, device=device, dtype=torch.float32)[..., None] if v.mask is not None
                  else torch.ones(H, W, 1, device=device))
            mse = float((((r - gt) ** 2) * mk).sum() / (mk.sum() * 3))
            ps.append(10 * math.log10(1 / max(mse, 1e-10)))
            ss.append(float(_ssim(r * mk, gt * mk)))
            if len(imgs) < previews:
                imgs.append(np.concatenate([v.image, r.cpu().numpy()], 1))
    return {"num_gaussians": d["n"], "sh_degree": d["deg"], "views": [v.name for v in views],
            "psnr": round(float(np.mean(ps)), 2), "ssim": round(float(np.mean(ss)), 3),
            "psnr_per_view": [round(x, 2) for x in ps], "previews": imgs}
