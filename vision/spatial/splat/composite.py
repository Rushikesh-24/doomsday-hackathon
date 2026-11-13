"""Memory-bounded alpha compositing of depth-sorted Gaussians per tile, with an analytic backward pass
(the same recurrence as the reference 3DGS CUDA kernel: back-to-front accumulation of the colour
behind each Gaussian). Every overlapping Gaussian is composited (no K truncation) in chunks; forward
stops early when all pixels of all tiles are saturated (transmittance < 1e-4)."""
from __future__ import annotations

import torch

ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_EPS = 1e-4


def _alpha(m2d, con, op, px, py):
    """m2d (t,k,2) con (t,k,3) op (t,k) px/py (t,P) -> alpha (t,k,P), power, dx, dy, raw mask."""
    dx = px[:, None, :] - m2d[..., 0:1]
    dy = py[:, None, :] - m2d[..., 1:2]
    power = -0.5 * (con[..., 0:1] * dx * dx + con[..., 2:3] * dy * dy) - con[..., 1:2] * dx * dy
    g = torch.exp(power.clamp(max=0))
    a_raw = op[..., None] * g
    valid = (power <= 0) & (a_raw >= ALPHA_MIN)
    alpha = torch.where(valid, a_raw.clamp(max=ALPHA_MAX), torch.zeros_like(a_raw))
    unclamped = valid & (a_raw < ALPHA_MAX)
    return alpha, g, dx, dy, unclamped


def _chunk_weights(alpha: torch.Tensor, Tin: torch.Tensor):
    """Per-pixel early termination as in the reference kernel: a Gaussian only contributes while the
    transmittance before it is >= T_EPS. Returns (masked alpha, live mask, transmittance before each)."""
    om = 1 - alpha
    excl = torch.cumprod(torch.cat([torch.ones_like(om[:, :1]), om[:, :-1]], 1), 1)
    live = (Tin[:, None, :] * excl) >= T_EPS
    a = alpha * live
    om2 = 1 - a
    excl2 = torch.cumprod(torch.cat([torch.ones_like(om2[:, :1]), om2[:, :-1]], 1), 1)
    return a, live, Tin[:, None, :] * excl2


class Composite(torch.autograd.Function):
    """Per chunk, only tiles that still have Gaussians in that chunk and are not saturated are processed.
    The transmittance entering each processed chunk is stored, so the backward pass recomputes T_i with
    forward products (numerically stable; no division by tiny transmittances)."""

    @staticmethod
    def forward(ctx, means2d, conic, opac, color, pad, px, py, bg, chunk: int):
        N = means2d.shape[0]
        m2 = torch.cat([means2d, means2d.new_zeros(1, 2)])
        cn = torch.cat([conic, conic.new_ones(1, 3)])
        opv = torch.cat([opac, opac.new_zeros(1)])
        cl = torch.cat([color, color.new_zeros(1, 3)])
        tiles, kmax = pad.shape
        P = px.shape[1]
        counts = (pad < N).sum(1)
        Tr = means2d.new_ones(tiles, P)
        C = means2d.new_zeros(tiles, P, 3)
        schedule = []
        for s in range(0, kmax, chunk):
            rows = torch.nonzero((counts > s) & (Tr.max(1).values > T_EPS)).squeeze(1)
            if rows.numel() == 0:
                break
            idx = pad[rows, s:s + chunk]
            alpha, *_ = _alpha(m2[idx], cn[idx], opv[idx], px[rows], py[rows])
            Tin = Tr[rows]
            a, live, Ti = _chunk_weights(alpha, Tin)
            w = a * Ti
            C[rows] = C[rows] + torch.einsum("tkp,tkc->tpc", w, cl[idx])
            Tr[rows] = Tin * (1 - a).prod(1)
            schedule.append((s, rows, Tin))
        out = C + Tr[..., None] * bg
        ctx.save_for_backward(means2d, conic, opac, color, pad, px, py, bg, Tr)
        ctx.chunk, ctx.schedule = chunk, schedule
        return out

    @staticmethod
    def backward(ctx, gout):
        means2d, conic, opac, color, pad, px, py, bg, T_final = ctx.saved_tensors
        N = means2d.shape[0]
        m2 = torch.cat([means2d, means2d.new_zeros(1, 2)])
        cn = torch.cat([conic, conic.new_ones(1, 3)])
        opv = torch.cat([opac, opac.new_zeros(1)])
        cl = torch.cat([color, color.new_zeros(1, 3)])
        g_m2 = torch.zeros_like(m2)
        g_cn = torch.zeros_like(cn)
        g_op = torch.zeros_like(opv)
        g_cl = torch.zeros_like(cl)
        g_bg = (gout * T_final[..., None]).sum((0, 1))
        Sg = (gout * bg).sum(-1) * T_final     # (colour behind current position) . dL/dC   (t,P)
        chunk = ctx.chunk
        for s, rows, Tin in reversed(ctx.schedule):
            idx = pad[rows, s:s + chunk]
            go = gout[rows]
            alpha, g, dx, dy, unc = _alpha(m2[idx], cn[idx], opv[idx], px[rows], py[rows])
            a, live, Ti = _chunk_weights(alpha, Tin)
            w = a * Ti
            c = cl[idx]
            cg = torch.einsum("tkc,tpc->tkp", c, go)                  # c_i . dL/dC (early contraction)
            wcg = w * cg
            behind_g = Sg[rows][:, None, :] + torch.flip(torch.cumsum(torch.flip(wcg, [1]), 1), [1]) - wcg
            g_alpha = Ti * cg - behind_g / (1 - a)                     # 1 - a >= 0.01 (alpha clamp)
            g_c = torch.einsum("tkp,tpc->tkc", w, go)
            g_alpha = torch.where(unc & live, g_alpha, torch.zeros_like(g_alpha))
            op_k = opv[idx][..., None]
            g_power = g_alpha * op_k * g
            g_opk = (g_alpha * g).sum(-1)
            con = cn[idx]
            g_mx = (g_power * (con[..., 0:1] * dx + con[..., 1:2] * dy)).sum(-1)
            g_my = (g_power * (con[..., 2:3] * dy + con[..., 1:2] * dx)).sum(-1)
            g_a = (g_power * (-0.5 * dx * dx)).sum(-1)
            g_b = (g_power * (-dx * dy)).sum(-1)
            g_cc = (g_power * (-0.5 * dy * dy)).sum(-1)
            flat = idx.reshape(-1)
            g_m2.index_add_(0, flat, torch.stack([g_mx, g_my], -1).reshape(-1, 2))
            g_cn.index_add_(0, flat, torch.stack([g_a, g_b, g_cc], -1).reshape(-1, 3))
            g_op.index_add_(0, flat, g_opk.reshape(-1))
            g_cl.index_add_(0, flat, g_c.reshape(-1, 3))
            Sg[rows] = Sg[rows] + wcg.sum(1)
        return g_m2[:N], g_cn[:N], g_op[:N], g_cl[:N], None, None, None, g_bg, None


def composite_reference(means2d, conic, opac, color, pad, px, py, bg):
    """Plain autograd version (all Gaussians at once) used to verify the custom backward."""
    N = means2d.shape[0]
    m2 = torch.cat([means2d, means2d.new_zeros(1, 2)])[pad]
    cn = torch.cat([conic, conic.new_ones(1, 3)])[pad]
    opv = torch.cat([opac, opac.new_zeros(1)])[pad]
    cl = torch.cat([color, color.new_zeros(1, 3)])[pad]
    alpha, *_ = _alpha(m2, cn, opv, px, py)
    a, _, Ti = _chunk_weights(alpha, torch.ones_like(px))
    w = a * Ti
    return torch.einsum("tkp,tkc->tpc", w, cl) + (1 - w.sum(1))[..., None] * bg
