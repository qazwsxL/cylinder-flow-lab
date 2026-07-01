#!/usr/bin/env python
# coding: utf-8
"""
cfp_pt.py  –  Cylinder-flow PINN with Pseudo-Time Stepping
============================================================
Implements the adaptive pseudo-time stepping (PT) method from

    Wang, Koohy, Lu & Perdikaris (2026)
    "When PINNs Go Wrong: Pseudo-Time Stepping Against Spurious Solutions"
    arXiv:2604.23528

The core idea:  at each "pseudo-time step" τ, the PDE residual is modified to

    R_PT = R(u) + w * (u_curr − u_prev)

where
  • u_prev   = network output at the PREVIOUS pseudo-time step (stop-gradient)
  • w = 1/Δτ  is chosen adaptively as  ‖ΔR‖ / ‖Δu‖,
    a finite-difference estimate of the spectral radius of ∂R/∂u.

This is combined with collocation-point resampling to avoid spurious solutions.

Usage (same CLI as cfp.py, plus --use-pt / --pt-inner-steps):
    python cfp_pt.py \\
        --vtk-dir  Re60/Re60_vtks \\
        --vtk40    Re40.vtk \\
        --save-dir runs/soap_pt/checkpoints \\
        --device   cuda \\
        --bfgs-method SSBroyden2 \\
        --use-pt \\
        --pt-inner-steps 200
"""

import math, os, sys, glob
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm

# ── _optimize path fix ────────────────────────────────────────────────────────
_opt_dir = os.path.dirname(os.path.abspath(__file__))
if _opt_dir not in sys.path:
    sys.path.insert(0, _opt_dir)

try:
    from _optimize import fmin_bfgs as _custom_fmin_bfgs
    _CUSTOM_BFGS_AVAILABLE = True
    print("[OK] _optimize loaded – custom BFGS available")
except ImportError as _e:
    _CUSTOM_BFGS_AVAILABLE = False
    import warnings
    warnings.warn(f"_optimize not found ({_e}): falling back to scipy BFGS",
                  RuntimeWarning)
    from scipy.optimize import fmin_bfgs as _scipy_fmin_bfgs

    def _custom_fmin_bfgs(f, x0, fprime=None, gtol=1e-5, maxiter=None,
                           disp=False, full_output=False,
                           method_bfgs="BFGS", initial_scale=False, **kw):
        return _scipy_fmin_bfgs(f, x0, fprime=fprime, gtol=gtol,
                                 maxiter=maxiter, disp=disp,
                                 full_output=full_output, **kw)
    _CUSTOM_BFGS_AVAILABLE = True


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Stream-function + pressure MLP: (x, y, t) → (ψ, p)."""

    def __init__(self, in_dim=3, out_dim=2, width=32, depth=3, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xyt):
        return self.net(xyt)


# Modified MLP: multiplicative (U, V) feature encoding – better for multiscale
class ModifiedMLP(nn.Module):
    """
    ModifiedMLP from Wang et al. 2022 / 2026.
    Uses two encoder branches whose element-wise product is added at each layer,
    giving better spectral learning across scales.
    """

    def __init__(self, in_dim=3, out_dim=2, width=64, depth=4, act=nn.Tanh):
        super().__init__()
        self.act = act()
        self.enc_U = nn.Linear(in_dim, width)
        self.enc_V = nn.Linear(in_dim, width)
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, out_dim))
        self.layers = nn.ModuleList(layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xyt):
        U = self.act(self.enc_U(xyt))
        V = self.act(self.enc_V(xyt))
        h = self.act(self.layers[0](xyt))
        for layer in self.layers[1:-1]:
            h = self.act(layer(h)) * U + (1 - self.act(layer(h))) * V
        return self.layers[-1](h)


# ─────────────────────────────────────────────────────────────────────────────
# AD helpers
# ─────────────────────────────────────────────────────────────────────────────
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]


def second_grad(outputs, inputs, idx):
    g = grad(outputs, inputs)[:, idx:idx+1]
    return grad(g, inputs)[:, idx:idx+1]


# ─────────────────────────────────────────────────────────────────────────────
# Physics
# ─────────────────────────────────────────────────────────────────────────────
def uvp_from_psip(model, x, y, t):
    xyt = torch.cat([x, y, t], dim=1).requires_grad_(True)
    psi_p = model(xyt)
    psi, p = psi_p[:, 0:1], psi_p[:, 1:2]
    dpsi = grad(psi, xyt)
    u =  dpsi[:, 1:2]
    v = -dpsi[:, 0:1]
    return xyt, psi, p, u, v


def pde_residual(model, x, y, t, Re):
    Re = float(Re)
    xyt, _, p, u, v = uvp_from_psip(model, x, y, t)
    u_t = grad(u, xyt)[:, 2:3]; u_x = grad(u, xyt)[:, 0:1]; u_y = grad(u, xyt)[:, 1:2]
    v_t = grad(v, xyt)[:, 2:3]; v_x = grad(v, xyt)[:, 0:1]; v_y = grad(v, xyt)[:, 1:2]
    p_x = grad(p, xyt)[:, 0:1]; p_y = grad(p, xyt)[:, 1:2]
    u_xx = second_grad(u, xyt, 0); u_yy = second_grad(u, xyt, 1)
    v_xx = second_grad(v, xyt, 0); v_yy = second_grad(v, xyt, 1)
    Ru = u_t + u*u_x + v*u_y + p_x - (1.0/Re)*(u_xx + u_yy)
    Rv = v_t + u*v_x + v*v_y + p_y - (1.0/Re)*(v_xx + v_yy)
    return Ru, Rv


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-Time Stepper
# ─────────────────────────────────────────────────────────────────────────────
class PseudoTimeStepper:
    """
    Adaptive pseudo-time stepping (Wang et al. 2026).

    The PDE residual is modified:
        R_PT(u) = R(u) + w * (u − u_prev)

    Adaptive weight update (FD spectral radius estimate):
        w_new = ‖R(u_curr) − R(u_prev)‖ / ‖u_curr − u_prev‖

    where u_curr / u_prev are network outputs (u, v) evaluated at a fixed set
    of reference collocation points.
    """

    def __init__(self,
                 w_init: float = 1.0,
                 w_min:  float = 1e-2,
                 w_max:  float = 1e2,
                 ema:    float = 0.9,
                 n_ref:  int   = 4000):
        self.w      = w_init
        self.w_min  = w_min
        self.w_max  = w_max
        self.ema    = ema
        self.n_ref  = n_ref

        # Reference collocation points (fixed between resamplings)
        self.x_ref: Optional[torch.Tensor] = None
        self.y_ref: Optional[torch.Tensor] = None
        self.t_ref: Optional[torch.Tensor] = None

        # Cached previous solution & residuals at reference points
        self.u_prev:  Optional[torch.Tensor] = None
        self.v_prev:  Optional[torch.Tensor] = None
        self.Ru_prev: Optional[torch.Tensor] = None
        self.Rv_prev: Optional[torch.Tensor] = None

    # ── Reference point management ───────────────────────────────────────────
    def resample_reference(self, x_bounds, y_bounds, T, R_cyl, device):
        """Draw new reference collocation points (call each outer resample)."""
        x, y = _reject_sample_rect_outside_cylinder(
            self.n_ref,
            x_bounds[0], x_bounds[1],
            y_bounds[0], y_bounds[1],
            R_cyl, device)
        t = _rand_uniform(self.n_ref, 0.0, float(T), device)
        self.x_ref = x.detach()
        self.y_ref = y.detach()
        self.t_ref = t.detach()

    # ── Cache previous state ─────────────────────────────────────────────────
    @torch.no_grad()
    def cache_prev(self, model: nn.Module, Re: float):
        """
        Store u_prev and R_prev at reference points.
        Must be called at the BEGINNING of each pseudo-time step (before the
        inner optimisation loop).
        """
        if self.x_ref is None:
            return
        model.eval()
        x = self.x_ref.clone().requires_grad_(True)
        y = self.y_ref.clone().requires_grad_(True)
        t = self.t_ref.clone().requires_grad_(True)
        with torch.enable_grad():
            Ru, Rv = pde_residual(model, x, y, t, Re)
            _, _, _, u, v = uvp_from_psip(model, x, y, t)
        self.u_prev  = u.detach()
        self.v_prev  = v.detach()
        self.Ru_prev = Ru.detach()
        self.Rv_prev = Rv.detach()

    # ── Adaptive weight update ────────────────────────────────────────────────
    @torch.no_grad()
    def update_weight(self, model: nn.Module, Re: float):
        """
        Recompute adaptive weight w = ‖ΔR‖ / ‖Δu‖.
        Call AFTER the inner optimisation loop (end of each pseudo-time step).
        """
        if self.u_prev is None or self.Ru_prev is None:
            return

        model.eval()
        x = self.x_ref.clone().requires_grad_(True)
        y = self.y_ref.clone().requires_grad_(True)
        t = self.t_ref.clone().requires_grad_(True)
        with torch.enable_grad():
            Ru_c, Rv_c = pde_residual(model, x, y, t, Re)
            _, _, _, u_c, v_c = uvp_from_psip(model, x, y, t)

        dR_norm = (torch.cat([Ru_c.detach() - self.Ru_prev,
                               Rv_c.detach() - self.Rv_prev]).norm()).item()
        du_norm = (torch.cat([u_c.detach()  - self.u_prev,
                               v_c.detach()  - self.v_prev]).norm()).item()

        w_new = float(np.clip(dR_norm / (du_norm + 1e-8),
                               self.w_min, self.w_max))
        self.w = self.ema * self.w + (1.0 - self.ema) * w_new
        return w_new

    # ── PT-modified residuals ─────────────────────────────────────────────────
    def pt_residuals(self, model: nn.Module, xf, yf, tf, Re: float):
        """
        Return PT-modified residuals evaluated at collocation batch (xf,yf,tf):

            R_PT = R(u) + w * (u_curr − u_prev_interp)

        Because u_prev lives on self.{x,y,t}_ref and (xf,yf,tf) may differ,
        we evaluate the PT correction at the REFERENCE points and combine both
        contributions.  The returned (Ru, Rv) are at the reference points.

        For simplicity this function evaluates at reference points only and the
        caller is expected to also pass the standard collocation points for the
        boundary/IC losses.
        """
        x = self.x_ref.clone().requires_grad_(True)
        y = self.y_ref.clone().requires_grad_(True)
        t = self.t_ref.clone().requires_grad_(True)

        Ru, Rv = pde_residual(model, x, y, t, Re)

        if self.u_prev is None:
            return Ru, Rv

        _, _, _, u_c, v_c = uvp_from_psip(model, x, y, t)
        Ru = Ru + self.w * (u_c - self.u_prev)
        Rv = Rv + self.w * (v_c - self.v_prev)
        return Ru, Rv


# ─────────────────────────────────────────────────────────────────────────────
# VTK parser  (verbatim from cfp.py)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_vtk_snapshot(raw: bytes):
    def read_line(pos):
        end = raw.find(b'\n', pos)
        if end == -1: return '', len(raw)
        return raw[pos:end].decode('latin-1').strip(), end + 1

    pos = 0
    for _ in range(5):
        line, pos = read_line(pos)
    n_pts = int(line.split()[1])
    pts3  = np.frombuffer(raw[pos:pos+n_pts*3*4], dtype='>f4').reshape(n_pts, 3)
    pos  += n_pts * 3 * 4

    while True:
        line, newpos = read_line(pos)
        if line.strip(): break
        pos = newpos
    pos = newpos

    n_cells = int(line.split()[1]); n_cv = int(line.split()[2])
    cells_raw = np.frombuffer(raw[pos:pos+n_cv*4], dtype='>i4')
    pos += n_cv * 4

    while True:
        line, newpos = read_line(pos)
        if line.strip(): break
        pos = newpos
    pos = newpos
    pos += n_cells * 4

    while True:
        line, newpos = read_line(pos)
        if line.strip(): break
        pos = newpos
    pos = newpos

    while True:
        line, newpos = read_line(pos)
        if line.strip(): break
        pos = newpos
    pos = newpos
    n_fields = int(line.split()[2])

    fields = {}
    for _ in range(n_fields):
        while True:
            line, newpos = read_line(pos)
            if line.strip(): break
            pos = newpos
        pos = newpos
        parts = line.split()
        fname, ncomp, ntuples = parts[0], int(parts[1]), int(parts[2])
        arr = np.frombuffer(raw[pos:pos+ncomp*ntuples*4], dtype='>f4'
                            ).reshape(ntuples, ncomp).astype(np.float32, copy=True)
        pos += ncomp * ntuples * 4
        fields[fname] = arr

    centroids = np.zeros((n_cells, 2), dtype=np.float32)
    cidx = 0
    for i in range(n_cells):
        nv = cells_raw[cidx]
        verts = cells_raw[cidx+1:cidx+1+nv]
        centroids[i] = pts3[verts, :2].mean(axis=0)
        cidx += nv + 1

    U_key = 'U' if 'U' in fields else 'UMean'
    p_key = 'p' if 'p' in fields else 'pMean'
    return (centroids.astype(np.float32, copy=False),
            fields[U_key][:, :2].astype(np.float32, copy=False),
            fields[p_key][:, 0].astype(np.float32, copy=False))


def load_vtk_steady(vtk_path):
    with open(vtk_path, 'rb') as f:
        raw = f.read()
    centroids, U_xy, p_arr = _parse_vtk_snapshot(raw)
    print(f"[VTK steady] {vtk_path}  cells={centroids.shape[0]}")
    return centroids, U_xy, p_arr


def load_vtk_series(vtk_dir, prefix="Re60_", t_start=80.0,
                    dt_per_step=0.2, index_step=7, max_snapshots=None):
    pattern = os.path.join(vtk_dir, f"{prefix}*.vtk")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    if max_snapshots is not None:
        files = files[:max_snapshots]

    def _get_index(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return int(stem.replace(prefix, ""))

    indices = [_get_index(f) for f in files]
    idx0    = min(indices)
    snapshots = []
    for fpath, idx in zip(files, indices):
        t_phys = t_start + (idx - idx0) / index_step * dt_per_step
        with open(fpath, 'rb') as fh:
            raw = fh.read()
        centroids, U_xy, p_arr = _parse_vtk_snapshot(raw)
        snapshots.append(dict(t=t_phys, xy=centroids, U=U_xy, p=p_arr))
        print(f"  Loaded {os.path.basename(fpath)}  t={t_phys:.2f}s  cells={centroids.shape[0]}")

    print(f"[Series] {len(snapshots)} snapshots, "
          f"t in [{snapshots[0]['t']:.2f}, {snapshots[-1]['t']:.2f}] s")
    return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# Data-driven batch (from cfp.py)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DataDrivenBatch:
    x: torch.Tensor; y: torch.Tensor; t: torch.Tensor
    u: torch.Tensor; v: torch.Tensor
    weight: float = 1.0


def build_data_batches(snapshots, device, n_per_snapshot=2000, seed=42,
                       emphasize_late_time=True):
    rng = np.random.default_rng(seed)
    t_vals = np.array([float(s['t']) for s in snapshots], dtype=np.float32)
    t_min, t_max = float(t_vals.min()), float(t_vals.max())
    t_span = max(t_max - t_min, 1e-8)
    batches = []
    for snap in snapshots:
        N   = snap['xy'].shape[0]
        idx = rng.choice(N, size=min(n_per_snapshot, N), replace=False)
        xy  = snap['xy'][idx].astype(np.float32)
        U   = snap['U'][idx].astype(np.float32)
        t_v = float(snap['t'])
        tau = (t_v - t_min) / t_span
        w   = float(0.5 + 1.5 * tau) if emphasize_late_time else 1.0
        db  = DataDrivenBatch(
            x=torch.as_tensor(xy[:, 0:1], device=device, dtype=torch.float32),
            y=torch.as_tensor(xy[:, 1:2], device=device, dtype=torch.float32),
            t=torch.full((idx.size, 1), t_v, device=device, dtype=torch.float32),
            u=torch.as_tensor(U[:, 0:1], device=device, dtype=torch.float32),
            v=torch.as_tensor(U[:, 1:2], device=device, dtype=torch.float32),
            weight=w,
        )
        batches.append(db)
    return batches


def data_loss_fn(model, data_batches, n_random=16, v_weight=8.0):
    if not data_batches:
        return torch.tensor(0.0)
    idx = np.random.choice(len(data_batches),
                           size=min(n_random, len(data_batches)), replace=False)
    losses, weights = [], []
    for i in idx:
        db = data_batches[i]
        _, _, _, u_pred, v_pred = uvp_from_psip(model, db.x, db.y, db.t)
        Lu = ((u_pred - db.u) ** 2).mean()
        Lv = ((v_pred - db.v) ** 2).mean()
        losses.append(Lu + v_weight * Lv)
        weights.append(float(db.weight))
    wt = torch.as_tensor(weights, device=losses[0].device, dtype=torch.float32)
    lt = torch.stack(losses)
    return (wt * lt).sum() / wt.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Geometry / sampling helpers (from cfp.py)
# ─────────────────────────────────────────────────────────────────────────────
def _rand_uniform(n, lo, hi, device):
    return lo + (hi - lo) * torch.rand((n, 1), device=device, dtype=torch.float32)

def _sample_time(n, T, device):
    return _rand_uniform(n, 0.0, float(T), device)

def _outside_cylinder_mask(x, y, R, eps=0.0):
    return (x * x + y * y) >= (R + eps) ** 2

def _reject_sample_rect_outside_cylinder(n, xlo, xhi, ylo, yhi, R, device,
                                          max_rounds=50):
    xs, ys, need = [], [], n
    for _ in range(max_rounds):
        m = int(need * 1.4) + 32
        x = _rand_uniform(m, xlo, xhi, device)
        y = _rand_uniform(m, ylo, yhi, device)
        mask = _outside_cylinder_mask(x, y, R).squeeze(1)
        x, y = x[mask], y[mask]
        if x.numel() == 0: continue
        take = min(need, x.shape[0])
        xs.append(x[:take]); ys.append(y[:take]); need -= take
        if need <= 0: break
    if need > 0: raise RuntimeError("Rejection sampler failed.")
    return torch.cat(xs, 0), torch.cat(ys, 0)

def _sample_near_wall_band(n, R, band, x_bounds, y_bounds, device, max_rounds=80):
    xlo, xhi = x_bounds; ylo, yhi = y_bounds
    xs, ys, need = [], [], n
    for _ in range(max_rounds):
        m = int(need * 1.6) + 64
        theta = 2.0*math.pi*torch.rand((m,1), device=device, dtype=torch.float32)
        r0, r1 = R, R + band
        u = torch.rand((m,1), device=device, dtype=torch.float32)
        r = torch.sqrt((r1*r1 - r0*r0)*u + r0*r0)
        x, y = r*torch.cos(theta), r*torch.sin(theta)
        mask = ((x>=xlo)&(x<=xhi)&(y>=ylo)&(y<=yhi)).squeeze(1)
        x, y = x[mask], y[mask]
        if x.numel() == 0: continue
        take = min(need, x.shape[0])
        xs.append(x[:take]); ys.append(y[:take]); need -= take
        if need <= 0: break
    if need > 0:
        x2, y2 = _reject_sample_rect_outside_cylinder(
            need, xlo, xhi, ylo, yhi, R, device)
        xs.append(x2); ys.append(y2)
    return torch.cat(xs, 0), torch.cat(ys, 0)

def _sample_wake_box(n, box, R, device):
    xlo, xhi, ylo, yhi = box
    return _reject_sample_rect_outside_cylinder(
        n, xlo, xhi, ylo, yhi, R, device)


# ─────────────────────────────────────────────────────────────────────────────
# Residual resampler (from cfp.py)
# ─────────────────────────────────────────────────────────────────────────────
class ResidualResampler:
    def __init__(self, pool_size=80_000, Nf=8_000,
                 x_in=-5.0, x_out=25.0, y_min=-10.0, y_max=10.0,
                 R=0.5, near_wall_band=0.6, wake_box=(0.0,15.0,-3.0,3.0),
                 frac_uniform=0.50, frac_near_wall=0.30, frac_wake=0.20, T=80.0):
        self.pool_size=pool_size; self.Nf=Nf
        self.x_in=x_in; self.x_out=x_out; self.y_min=y_min
        self.y_max=y_max; self.R=R
        self.near_wall_band=near_wall_band; self.wake_box=wake_box
        self.frac_uniform=frac_uniform; self.frac_near_wall=frac_near_wall
        self.frac_wake=frac_wake; self.T=T
        self._pool_x=self._pool_y=self._pool_t=None

    def _build_pool(self, device):
        n=self.pool_size
        n_u=int(n*self.frac_uniform); n_w=int(n*self.frac_near_wall); n_k=n-n_u-n_w
        x1,y1=_reject_sample_rect_outside_cylinder(
            n_u,self.x_in,self.x_out,self.y_min,self.y_max,self.R,device)
        x2,y2=_sample_near_wall_band(
            n_w,self.R,self.near_wall_band,
            (self.x_in,self.x_out),(self.y_min,self.y_max),device)
        x3,y3=_sample_wake_box(n_k,self.wake_box,self.R,device)
        self._pool_x=torch.cat([x1,x2,x3],0)
        self._pool_y=torch.cat([y1,y2,y3],0)
        self._pool_t=_sample_time(n,self.T,device)

    @torch.no_grad()
    def _compute_weights(self, model, Re, device):
        x=self._pool_x.clone(); y=self._pool_y.clone(); t=self._pool_t.clone()
        residuals=[]
        for i in range(0, x.shape[0], 4096):
            xi=x[i:i+4096].detach().requires_grad_(True)
            yi=y[i:i+4096].detach().requires_grad_(True)
            ti=t[i:i+4096].detach().requires_grad_(True)
            with torch.enable_grad():
                Ru, Rv = pde_residual(model, xi, yi, ti, Re)
            residuals.append((Ru.detach()**2+Rv.detach()**2).squeeze(1))
        return torch.cat(residuals, 0)

    def sample(self, model, Re, device):
        if self._pool_x is None: self._build_pool(device)
        else: self._pool_t = _sample_time(self.pool_size, self.T, device)
        w = self._compute_weights(model, Re, device) + 1e-8
        probs = w / w.sum()
        idx = torch.multinomial(probs, self.Nf, replacement=True)
        return self._pool_x[idx], self._pool_y[idx], self._pool_t[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Batch / sample
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SampleBatch:
    xf: torch.Tensor;  yf: torch.Tensor;  tf: torch.Tensor
    xin: torch.Tensor; yin: torch.Tensor; tin: torch.Tensor
    uin: torch.Tensor; vin: torch.Tensor
    xwall: torch.Tensor; ywall: torch.Tensor; twall: torch.Tensor
    xtb: torch.Tensor;  ytb: torch.Tensor;  ttb: torch.Tensor
    utb: torch.Tensor;  vtb: torch.Tensor
    xout: torch.Tensor; yout: torch.Tensor; tout: torch.Tensor
    x0: torch.Tensor;  y0: torch.Tensor;  t0: torch.Tensor
    u0: torch.Tensor;  v0: torch.Tensor


def sample_batch(device, Nf=8_000, Nb=4000, N0=8000, T=80.0,
                 x_in=-5.0, x_out=25.0, y_min=-10.0, y_max=10.0, R=0.5,
                 frac_uniform=0.50, frac_near_wall=0.30, frac_wake=0.20,
                 near_wall_band=0.6, wake_box=(0.0,15.0,-3.0,3.0),
                 ic_mode="uniform", steady_fn=None):
    n_uni=int(Nf*frac_uniform); n_wall=int(Nf*frac_near_wall); n_wake=Nf-n_uni-n_wall
    xf1,yf1=_reject_sample_rect_outside_cylinder(
        n_uni,x_in,x_out,y_min,y_max,R,device)
    xf2,yf2=_sample_near_wall_band(
        n_wall,R,near_wall_band,(x_in,x_out),(y_min,y_max),device)
    xf3,yf3=_sample_wake_box(n_wake,wake_box,R,device)
    xf=torch.cat([xf1,xf2,xf3],0); yf=torch.cat([yf1,yf2,yf3],0)
    tf=torch.cat([_sample_time(n_uni,T,device),
                  _sample_time(n_wall,T,device),
                  _sample_time(n_wake,T,device)],0)

    Nb_in=int(Nb*0.25); Nb_wall=int(Nb*0.35)
    Nb_tb=int(Nb*0.25); Nb_out=Nb-Nb_in-Nb_wall-Nb_tb
    xin=torch.full((Nb_in,1),float(x_in),device=device,dtype=torch.float32)
    yin=_rand_uniform(Nb_in,y_min,y_max,device); tin=_sample_time(Nb_in,T,device)
    uin=torch.ones((Nb_in,1),device=device,dtype=torch.float32)
    vin=torch.zeros((Nb_in,1),device=device,dtype=torch.float32)

    theta=2.0*math.pi*torch.rand((Nb_wall,1),device=device,dtype=torch.float32)
    xwall=R*torch.cos(theta); ywall=R*torch.sin(theta)
    twall=_sample_time(Nb_wall,T,device)

    n_top=Nb_tb//2; n_bot=Nb_tb-n_top
    xtb=torch.cat([_rand_uniform(n_top,x_in,x_out,device),
                   _rand_uniform(n_bot,x_in,x_out,device)],0)
    ytb=torch.cat([
        torch.full((n_top,1),float(y_max),device=device,dtype=torch.float32),
        torch.full((n_bot,1),float(y_min),device=device,dtype=torch.float32)],0)
    ttb=_sample_time(Nb_tb,T,device)
    utb=torch.ones((Nb_tb,1),device=device,dtype=torch.float32)
    vtb=torch.zeros((Nb_tb,1),device=device,dtype=torch.float32)
    xout_t=torch.full((Nb_out,1),float(x_out),device=device,dtype=torch.float32)
    yout_t=_rand_uniform(Nb_out,y_min,y_max,device)
    tout_t=_sample_time(Nb_out,T,device)

    x0,y0=_reject_sample_rect_outside_cylinder(N0,x_in,x_out,y_min,y_max,R,device)
    t0=torch.zeros((N0,1),device=device,dtype=torch.float32)
    if ic_mode=="uniform":
        u0=torch.ones((N0,1),device=device,dtype=torch.float32)
        v0=0.01*torch.sin(0.5*math.pi*y0)*torch.exp(-0.02*(x0-x_in)**2)
    elif ic_mode=="steady_fn":
        if steady_fn is None: raise ValueError("steady_fn required")
        u0, v0 = steady_fn(x0, y0)
    else:
        raise ValueError("ic_mode must be 'uniform' or 'steady_fn'")

    return SampleBatch(xf=xf,yf=yf,tf=tf,xin=xin,yin=yin,tin=tin,uin=uin,vin=vin,
                       xwall=xwall,ywall=ywall,twall=twall,xtb=xtb,ytb=ytb,ttb=ttb,
                       utb=utb,vtb=vtb,xout=xout_t,yout=yout_t,tout=tout_t,
                       x0=x0,y0=y0,t0=t0,u0=u0,v0=v0)


# ─────────────────────────────────────────────────────────────────────────────
# Loss function  (extended with PT)
# ─────────────────────────────────────────────────────────────────────────────
def mse(a): return (a**2).mean()


def loss_fn(model, batch: SampleBatch, Re: float, w,
            stepper: Optional[PseudoTimeStepper] = None,
            data_batches=None, n_data_snapshots=16, data_v_weight=8.0):
    Re = float(Re)

    # ── PDE residual (with optional PT correction) ───────────────────────────
    if stepper is not None and stepper.u_prev is not None:
        # Evaluate at reference points with PT modification
        Ru, Rv = stepper.pt_residuals(model, batch.xf, batch.yf, batch.tf, Re)
    else:
        Ru, Rv = pde_residual(model, batch.xf, batch.yf, batch.tf, Re)
    L_pde = mse(Ru) + mse(Rv)

    # ── Boundary conditions ──────────────────────────────────────────────────
    _,_,_,u_in,v_in   = uvp_from_psip(model,batch.xin,batch.yin,batch.tin)
    L_in  = mse(u_in-batch.uin)+mse(v_in-batch.vin)
    _,_,_,u_wal,v_wal = uvp_from_psip(model,batch.xwall,batch.ywall,batch.twall)
    L_wall= mse(u_wal)+mse(v_wal)
    _,_,_,u_tb,v_tb   = uvp_from_psip(model,batch.xtb,batch.ytb,batch.ttb)
    L_tb  = mse(u_tb-batch.utb)+mse(v_tb-batch.vtb)
    xyt_out,_,p_out,u_out,v_out = uvp_from_psip(model,batch.xout,batch.yout,batch.tout)
    L_out = (mse(p_out)+mse(grad(u_out,xyt_out)[:,0:1])
             +mse(grad(v_out,xyt_out)[:,0:1]))
    _,_,_,u0_p,v0_p   = uvp_from_psip(model,batch.x0,batch.y0,batch.t0)
    L_ic  = mse(u0_p-batch.u0)+mse(v0_p-batch.v0)
    L_bc  = L_in+L_wall+L_tb

    L = w["pde"]*L_pde + w["bc"]*L_bc + w["ic"]*L_ic + w["out"]*L_out

    # ── Data loss ────────────────────────────────────────────────────────────
    L_data = torch.tensor(0.0, device=batch.xf.device)
    if data_batches:
        L_data = data_loss_fn(model, data_batches,
                              n_random=n_data_snapshots,
                              v_weight=data_v_weight)
        L += w.get("data", 50.0) * L_data

    logs = dict(L=L.item(), L_pde=L_pde.item(), L_bc=L_bc.item(),
                L_ic=L_ic.item(), L_out=L_out.item(), L_data=L_data.item(),
                pt_w=stepper.w if stepper else 0.0)
    return L, logs


# ─────────────────────────────────────────────────────────────────────────────
# Param helpers
# ─────────────────────────────────────────────────────────────────────────────
def _params_to_vec(model):
    return np.concatenate(
        [p.detach().cpu().numpy().ravel() for p in model.parameters()])

def _vec_to_params(model, vec):
    vec = torch.as_tensor(vec, dtype=torch.float32)
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[offset:offset+n].view_as(p)); offset += n


# ─────────────────────────────────────────────────────────────────────────────
# Training  (with pseudo-time stepping)
# ─────────────────────────────────────────────────────────────────────────────
def train_one_Re(
    model, Re, device,
    # ── Adam warm-up ──────────────────────────────────────────────────────────
    warmup_adam_steps=500,
    lr_warmup=1e-3,
    # ── BFGS ─────────────────────────────────────────────────────────────────
    epochs_bfgs=2000,
    bfgs_method="BFGS",
    bfgs_gtol=1e-6,
    bfgs_resample_every=100,
    # ── Pseudo-time stepping ──────────────────────────────────────────────────
    use_pt=True,
    pt_w_init=1.0,
    pt_w_min=1e-2,
    pt_w_max=1e2,
    pt_ema=0.9,
    pt_n_ref=4000,
    pt_update_every=100,   # update pt weight every N BFGS calls
    # ── Weights & sampling ────────────────────────────────────────────────────
    w=None,
    Nf=8_000, Nb=4000, N0=8000, T=80.0,
    pool_size=80_000,
    ic_mode="uniform", steady_fn=None,
    # ── Data supervision ──────────────────────────────────────────────────────
    data_batches=None,
    n_data_snapshots=16,
    data_v_weight=8.0,
    # ── Domain ───────────────────────────────────────────────────────────────
    x_in=-5.0, x_out=25.0, y_min=-10.0, y_max=10.0, R=0.5,
):
    if w is None:
        if data_batches:
            w = {"pde": 1.0, "bc": 10.0, "ic": 2.0, "out": 1.0, "data": 50.0}
        else:
            w = {"pde": 1.0, "bc": 10.0, "ic": 10.0, "out": 1.0, "data": 0.0}

    # ── Stepper ───────────────────────────────────────────────────────────────
    stepper: Optional[PseudoTimeStepper] = None
    if use_pt:
        stepper = PseudoTimeStepper(
            w_init=pt_w_init, w_min=pt_w_min, w_max=pt_w_max,
            ema=pt_ema, n_ref=pt_n_ref)
        stepper.resample_reference(
            (x_in, x_out), (y_min, y_max), T, R, device)
        stepper.cache_prev(model, Re)
        print(f"  [PT] enabled  w_init={pt_w_init}")

    resampler = ResidualResampler(pool_size=pool_size, Nf=Nf, T=T,
                                   x_in=x_in, x_out=x_out,
                                   y_min=y_min, y_max=y_max, R=R)
    batch = sample_batch(device, Nf=Nf, Nb=Nb, N0=N0, T=T,
                         x_in=x_in, x_out=x_out, y_min=y_min, y_max=y_max, R=R,
                         ic_mode=ic_mode, steady_fn=steady_fn)

    def patch_interior(b, xf_new, yf_new, tf_new):
        return SampleBatch(xf=xf_new,yf=yf_new,tf=tf_new,
                           xin=b.xin,yin=b.yin,tin=b.tin,uin=b.uin,vin=b.vin,
                           xwall=b.xwall,ywall=b.ywall,twall=b.twall,
                           xtb=b.xtb,ytb=b.ytb,ttb=b.ttb,utb=b.utb,vtb=b.vtb,
                           xout=b.xout,yout=b.yout,tout=b.tout,
                           x0=b.x0,y0=b.y0,t0=b.t0,u0=b.u0,v0=b.v0)

    # ── Adam warm-up ──────────────────────────────────────────────────────────
    if warmup_adam_steps > 0:
        print(f"  [Warm-up] Adam {warmup_adam_steps} steps")
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr_warmup)
        for ep in range(warmup_adam_steps):
            opt.zero_grad()
            L, logs = loss_fn(model, batch, Re, w,
                              stepper=None,   # no PT during warm-up
                              data_batches=data_batches,
                              n_data_snapshots=n_data_snapshots,
                              data_v_weight=data_v_weight)
            L.backward(); opt.step()
            if ep % 100 == 0:
                dstr = f"  DATA={logs['L_data']:.3e}" if data_batches else ""
                print(f"  [Warm-up Re={Re:5.1f}] ep={ep:4d}"
                      f"  L={logs['L']:.3e}  PDE={logs['L_pde']:.3e}{dstr}")
        # Initialise PT after warm-up
        if stepper is not None:
            stepper.cache_prev(model, Re)

    # ── BFGS ──────────────────────────────────────────────────────────────────
    pt_str = f"  PT=on(w={stepper.w:.2f})" if stepper else ""
    print(f"  [BFGS] method={bfgs_method}  max_iter={epochs_bfgs}{pt_str}")

    call_counter  = [0]
    current_batch = [batch]

    def f_and_g(x):
        call_counter[0] += 1
        c = call_counter[0]

        # ── Resample collocation points ───────────────────────────────────
        if c % bfgs_resample_every == 0:
            model.eval()
            xf_r, yf_r, tf_r = resampler.sample(model, Re, device)
            current_batch[0] = patch_interior(current_batch[0], xf_r, yf_r, tf_r)

            # PT: update prev solution & adaptive weight at each resample
            if stepper is not None:
                w_raw = stepper.update_weight(model, Re)
                stepper.resample_reference((x_in, x_out), (y_min, y_max), T, R, device)
                stepper.cache_prev(model, Re)
                print(f"  [PT] resample  w={stepper.w:.4f}  (raw={w_raw:.4f})")

        _vec_to_params(model, x)
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()
        model.train()
        L, logs = loss_fn(model, current_batch[0], Re, w,
                          stepper=stepper,
                          data_batches=data_batches,
                          n_data_snapshots=n_data_snapshots,
                          data_v_weight=data_v_weight)
        L.backward()
        g = np.concatenate([
            (p.grad.detach().cpu().numpy().ravel() if p.grad is not None
             else np.zeros(p.numel(), dtype=np.float32))
            for p in model.parameters()])

        if c % 50 == 0:
            dstr = f"  DATA={logs['L_data']:.3e}" if data_batches else ""
            pstr = f"  PT_w={logs['pt_w']:.3f}" if stepper else ""
            print(f"  [BFGS Re={Re:5.1f}] call={c:5d}"
                  f"  L={logs['L']:.3e}  PDE={logs['L_pde']:.3e}"
                  f"  BC={logs['L_bc']:.3e}{dstr}{pstr}")

        return float(L.item()), g

    x0_vec = _params_to_vec(model)
    try:
        x_opt = _custom_fmin_bfgs(
            f=f_and_g,
            x0=x0_vec,
            fprime=None,
            gtol=bfgs_gtol,
            maxiter=epochs_bfgs,
            disp=False,
            full_output=False,
            method_bfgs=bfgs_method,
            initial_scale=True,
        )
        _vec_to_params(model, x_opt)
        print(f"  [BFGS] converged – {call_counter[0]} total calls")
    except Exception as exc:
        print(f"  [BFGS] stopped: {exc}  ({call_counter[0]} calls)")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation (from cfp.py)
# ─────────────────────────────────────────────────────────────────────────────
def _vorticity_from_cfd(snap, x_bounds=(-5,25), y_bounds=(-10,10), R=0.5,
                         grid_nx=300, grid_ny=150):
    xy=snap['xy']; U=snap['U']
    outside=(xy[:,0]**2+xy[:,1]**2)>=R**2
    xy,U=xy[outside],U[outside]
    xi=np.linspace(x_bounds[0],x_bounds[1],grid_nx)
    yi=np.linspace(y_bounds[0],y_bounds[1],grid_ny)
    Xi,Yi=np.meshgrid(xi,yi)
    tri=mtri.Triangulation(xy[:,0],xy[:,1])
    Ui=np.array(mtri.LinearTriInterpolator(tri,U[:,0])(Xi,Yi))
    Vi=np.array(mtri.LinearTriInterpolator(tri,U[:,1])(Xi,Yi))
    dx=xi[1]-xi[0]; dy=yi[1]-yi[0]
    omega=np.gradient(Vi,dx,axis=1)-np.gradient(Ui,dy,axis=0)
    mask=(Xi**2+Yi**2)<R**2
    omega[mask]=np.nan
    return Xi,Yi,Ui,Vi,omega


def _vorticity_from_pinn(model, t, device,
                          x_bounds=(-5,25), y_bounds=(-10,10),
                          R=0.5, grid_nx=300, grid_ny=150):
    model.eval()
    x=np.linspace(x_bounds[0],x_bounds[1],grid_nx,dtype=np.float32)
    y=np.linspace(y_bounds[0],y_bounds[1],grid_ny,dtype=np.float32)
    Xi,Yi=np.meshgrid(x,y)
    x_t=torch.as_tensor(Xi.reshape(-1,1),device=device,dtype=torch.float32)
    y_t=torch.as_tensor(Yi.reshape(-1,1),device=device,dtype=torch.float32)
    t_t=torch.full_like(x_t,float(t))
    xyt,_,_,u,v=uvp_from_psip(model,x_t,y_t,t_t)
    du=torch.autograd.grad(u.sum(),xyt,create_graph=False,retain_graph=True)[0]
    dv=torch.autograd.grad(v.sum(),xyt,create_graph=False,retain_graph=False)[0]
    omega=(dv[:,0:1]-du[:,1:2]).detach().cpu().numpy().reshape(grid_ny,grid_nx)
    U=u.detach().cpu().numpy().reshape(grid_ny,grid_nx)
    V=v.detach().cpu().numpy().reshape(grid_ny,grid_nx)
    mask=(Xi**2+Yi**2)<R**2
    return Xi,Yi,np.where(mask,np.nan,U),np.where(mask,np.nan,V),np.where(mask,np.nan,omega)


# ─────────────────────────────────────────────────────────────────────────────
# Reynolds schedule + continuation scan
# ─────────────────────────────────────────────────────────────────────────────
def reynolds_schedule():
    Re_list  = list(range(5,45,5))
    Re_list += list(range(41,61,1))
    Re_list += list(range(65,101,5))
    return Re_list


def continuation_scan(
    device="cuda", save_dir=".", resume=True,
    vtk_path=None,
    re60_vtk_dir=None, re60_t_start=80.0, re60_dt=0.2, re60_index_step=7,
    re60_n_per_snap=2000, re60_max_snapshots=None,
    warmup_adam_steps=500,
    epochs_bfgs=2000,
    bfgs_method="BFGS",
    bfgs_gtol=1e-6,
    # PT options
    use_pt=True,
    pt_w_init=1.0,
    pt_n_ref=4000,
    # Architecture
    arch="mlp",
    width=32, depth=3,
    viz_after_re60=True, viz_n_snapshots=8,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    if arch.lower() == "modifiedmlp":
        model = ModifiedMLP(width=width, depth=depth).to(device)
    else:
        model = MLP(width=width, depth=depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {arch}  width={width}  depth={depth}  params={n_params}")

    # Re40 steady IC
    vtk_steady_fn = None
    if vtk_path is not None and os.path.exists(vtk_path):
        centroids, U_xy, _ = load_vtk_steady(vtk_path)
        from scipy.interpolate import LinearNDInterpolator
        iu=LinearNDInterpolator(centroids,U_xy[:,0],fill_value=1.0)
        iv=LinearNDInterpolator(centroids,U_xy[:,1],fill_value=0.0)
        def vtk_steady_fn(x0_t,y0_t):
            xy=np.column_stack([x0_t.cpu().numpy().ravel(),
                                y0_t.cpu().numpy().ravel()])
            u0=torch.as_tensor(iu(xy).reshape(-1,1),device=device,dtype=torch.float32)
            v0=torch.as_tensor(iv(xy).reshape(-1,1),device=device,dtype=torch.float32)
            return u0,v0
        print("VTK steady IC interpolator ready.")

    # Re60 time-series data
    re60_data_batches = None; re60_snapshots = None
    if re60_vtk_dir is not None and os.path.exists(re60_vtk_dir):
        print(f"\nLoading Re60 time series from {re60_vtk_dir} ...")
        re60_snapshots = load_vtk_series(
            re60_vtk_dir, prefix="Re60_",
            t_start=re60_t_start, dt_per_step=re60_dt, index_step=re60_index_step,
            max_snapshots=re60_max_snapshots)
        re60_data_batches = build_data_batches(re60_snapshots, device,
                                               n_per_snapshot=re60_n_per_snap)
        print(f"  -> {len(re60_data_batches)} data batches ready.")

    # Resume from last checkpoint
    schedule = reynolds_schedule()
    if resume:
        existing=[Re for Re in schedule
                  if os.path.exists(os.path.join(save_dir,f"pinn_Re{Re}.pt"))]
        if existing:
            last_re=max(existing)
            ckpt=os.path.join(save_dir,f"pinn_Re{last_re}.pt")
            print(f"Resuming from {ckpt}")
            model.load_state_dict(torch.load(ckpt,map_location=device))
        else:
            print("Starting from scratch.")

    results={}
    for Re in schedule:
        ckpt=os.path.join(save_dir,f"pinn_Re{Re}.pt")
        if os.path.exists(ckpt):
            print(f"\nSkipping Re={Re} (checkpoint exists)")
            results[Re]={"path":ckpt}; continue

        print("\n"+"="*70+f"\nTraining Re={Re}  [use_pt={use_pt}]\n"+"="*70)

        ic_mode = "steady_fn" if (vtk_steady_fn is not None and 35<=Re<=45) else "uniform"
        sfn     = vtk_steady_fn if ic_mode=="steady_fn" else None
        data_b  = re60_data_batches if (Re==60 and re60_data_batches is not None) else None

        model = train_one_Re(
            model, Re, device,
            warmup_adam_steps=warmup_adam_steps,
            epochs_bfgs=epochs_bfgs,
            bfgs_method=bfgs_method,
            bfgs_gtol=bfgs_gtol,
            bfgs_resample_every=100,
            use_pt=use_pt,
            pt_w_init=pt_w_init,
            pt_n_ref=pt_n_ref,
            w={"pde":1.0,"bc":10.0,"ic":2.0,"out":1.0,
               "data":50.0 if data_b else 0.0},
            Nf=8_000, Nb=4000, N0=8000,
            T=100.0 if Re==60 else 80.0,
            pool_size=80_000,
            ic_mode=ic_mode, steady_fn=sfn,
            data_batches=data_b,
            n_data_snapshots=16,
            data_v_weight=8.0,
        )

        torch.save(model.state_dict(),ckpt)
        torch.save(model.state_dict(),os.path.join(save_dir,"pinn_latest.pt"))
        results[Re]={"path":ckpt}
        print(f"Saved: {ckpt}")

        if Re==60 and viz_after_re60 and re60_snapshots is not None:
            viz_dir=os.path.join(save_dir,"vortex_viz")
            _viz_vortex(re60_snapshots, model, device, viz_dir, viz_n_snapshots)

    return results


def _viz_vortex(snapshots, model, device, out_dir, n_snapshots=8):
    os.makedirs(out_dir, exist_ok=True)
    step=max(1,len(snapshots)//n_snapshots)
    chosen=snapshots[::step][:n_snapshots]
    norm=TwoSlopeNorm(vmin=-2,vcenter=0,vmax=2); cmap="RdBu_r"
    theta_cyl=np.linspace(0,2*np.pi,300)
    cx=0.5*np.cos(theta_cyl); cy=0.5*np.sin(theta_cyl)
    ncols=len(chosen)
    fig,axes=plt.subplots(2,ncols,figsize=(3*ncols,5),sharex=True,sharey=True,
                           constrained_layout=True)
    for col,snap in enumerate(chosen):
        t_val=snap['t']
        Xi,Yi,_,_,Om_cfd=_vorticity_from_cfd(snap,x_bounds=(-2,15),y_bounds=(-4,4))
        Xi,Yi,_,_,Om_pin=_vorticity_from_pinn(model,t_val,device,
                                               x_bounds=(-2,15),y_bounds=(-4,4))
        axes[0,col].pcolormesh(Xi,Yi,Om_cfd,cmap=cmap,norm=norm,shading='auto')
        axes[0,col].fill(cx,cy,color='gray',zorder=3)
        axes[0,col].set_title(f"t={t_val:.1f}",fontsize=8)
        if col==0: axes[0,col].set_ylabel("CFD")
        axes[1,col].pcolormesh(Xi,Yi,Om_pin,cmap=cmap,norm=norm,shading='auto')
        axes[1,col].fill(cx,cy,color='gray',zorder=3)
        if col==0: axes[1,col].set_ylabel("PINN+PT")
    fig.suptitle(f"Vorticity Re=60 (use_pt=True)",fontsize=12,fontweight='bold')
    fig.savefig(os.path.join(out_dir,"vortex_pt.png"),dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Cylinder-flow PINN with pseudo-time stepping (Wang et al. 2026)")
    parser.add_argument("--vtk-dir",    default=None,
                        help="Directory with Re60_XXXX.vtk files")
    parser.add_argument("--vtk40",      default="Re40.vtk")
    parser.add_argument("--save-dir",   default="runs/pinn_pt/checkpoints")
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--bfgs-method", default="SSBroyden2",
                        choices=["BFGS","BFGS_scipy","SSBFGS_OL",
                                 "SSBFGS_AB","SSBroyden1","SSBroyden2"])
    parser.add_argument("--warmup",     type=int, default=500)
    parser.add_argument("--bfgs-iter",  type=int, default=2000)
    # PT flags
    parser.add_argument("--use-pt",     action="store_true", default=True,
                        help="Enable pseudo-time stepping (default: on)")
    parser.add_argument("--no-pt",      dest="use_pt", action="store_false",
                        help="Disable pseudo-time stepping")
    parser.add_argument("--pt-w-init",  type=float, default=1.0,
                        help="Initial PT weight (1/Δτ)")
    parser.add_argument("--pt-n-ref",   type=int,   default=4000,
                        help="Number of reference points for PT")
    # Architecture
    parser.add_argument("--arch",       default="mlp",
                        choices=["mlp","modifiedmlp"])
    parser.add_argument("--width",      type=int, default=32)
    parser.add_argument("--depth",      type=int, default=3)
    args = parser.parse_args()

    continuation_scan(
        device=args.device,
        save_dir=args.save_dir,
        resume=True,
        vtk_path=args.vtk40,
        re60_vtk_dir=args.vtk_dir,
        warmup_adam_steps=args.warmup,
        epochs_bfgs=args.bfgs_iter,
        bfgs_method=args.bfgs_method,
        bfgs_gtol=1e-6,
        use_pt=args.use_pt,
        pt_w_init=args.pt_w_init,
        pt_n_ref=args.pt_n_ref,
        arch=args.arch,
        width=args.width,
        depth=args.depth,
        viz_after_re60=True,
    )
