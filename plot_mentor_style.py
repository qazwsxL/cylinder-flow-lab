#!/usr/bin/env python
# coding: utf-8
"""
plot_mentor_style.py
====================
Reproduce the mentor's figure layout (PINN for vortex shedding: Cd/Cl data only).

Figure structure
----------------
Left column  : FEM vs PINN field comparison for u  (3 rows × N_snap columns)
Right column : FEM vs PINN field comparison for v  (same)
Bottom-left  : Cd and Cl time series (FEM vs PINN)
Bottom-mid   : Convergence (total loss vs step, log scale)
Bottom-mid-R : Ablation bar chart  (mean field rel-L2 for each method)
Bottom-right : Error over the period (field rel-L2 vs t/T per method)

Typical call (after a run with cfp_pt.py):
    python plot_mentor_style.py \\
        --vtk-dir   Re60/Re60_vtks \\
        --ckpt      runs/pinn_pt/checkpoints/pinn_Re60.pt \\
        --run-label "SOAP+PT+ModifiedMLP" \\
        --loss-csv  runs/pinn_pt/loss_log.csv \\
        --out       figures/mentor_style.png

You can also import individual functions and call them from a notebook:
    from plot_mentor_style import make_full_figure
    make_full_figure(...)
"""

import os, glob, math, argparse
from typing import Optional, List, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm, Normalize
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib style – close to the mentor's aesthetic
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    "font.size"        : 7,
    "axes.titlesize"   : 7,
    "axes.labelsize"   : 7,
    "xtick.labelsize"  : 6,
    "ytick.labelsize"  : 6,
    "legend.fontsize"  : 6,
    "lines.linewidth"  : 1.0,
    "axes.linewidth"   : 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
}
plt.rcParams.update(STYLE)

# Colormaps
CMAP_FIELD  = "RdBu_r"          # velocity fields (FEM / PINN rows)
CMAP_ERR    = "hot"             # error row  (dark = small, bright = large)

# Domain bounds for plotting panels
XBOUNDS = (-2.0, 15.0)
YBOUNDS = (-4.0,  4.0)
R_CYL   = 0.5
GRID_NX = 350
GRID_NY = 120


# ─────────────────────────────────────────────────────────────────────────────
# VTK loader (minimal; same parse logic as cfp.py)
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


def load_vtk_series(vtk_dir, prefix="Re60_", t_start=80.0,
                    dt_per_step=0.2, index_step=7, max_snapshots=None):
    pattern = os.path.join(vtk_dir, f"{prefix}*.vtk")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No VTK files matching {pattern}")
    if max_snapshots:
        files = files[:max_snapshots]

    def _idx(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return int(stem.replace(prefix, ""))

    indices = [_idx(f) for f in files]
    idx0    = min(indices)
    snaps   = []
    for fpath, idx in zip(files, indices):
        t_phys = t_start + (idx - idx0) / index_step * dt_per_step
        with open(fpath, 'rb') as fh:
            raw = fh.read()
        xy, U, p = _parse_vtk_snapshot(raw)
        snaps.append(dict(t=t_phys, xy=xy, U=U, p=p))
    print(f"[VTK] {len(snaps)} snaps  t=[{snaps[0]['t']:.2f}, {snaps[-1]['t']:.2f}]")
    return snaps


# ─────────────────────────────────────────────────────────────────────────────
# PINN model helpers
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, width=32, depth=3, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ModifiedMLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, width=64, depth=4, act=nn.Tanh):
        super().__init__()
        self.act = act()
        self.enc_U = nn.Linear(in_dim, width)
        self.enc_V = nn.Linear(in_dim, width)
        layers = [nn.Linear(width, width) for _ in range(depth)]
        layers.append(nn.Linear(width, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, xyt):
        U = self.act(self.enc_U(xyt))
        V = self.act(self.enc_V(xyt))
        h = self.act(self.layers[0](xyt))
        for layer in self.layers[1:-1]:
            h = self.act(layer(h)) * U + (1 - self.act(layer(h))) * V
        return self.layers[-1](h)


def _grad(out, inp):
    return torch.autograd.grad(
        out, inp,
        grad_outputs=torch.ones_like(out),
        create_graph=True, retain_graph=True, only_inputs=True)[0]


def eval_uv_on_grid(model, t_val: float, device,
                     x_bounds=XBOUNDS, y_bounds=YBOUNDS,
                     grid_nx=GRID_NX, grid_ny=GRID_NY, R=R_CYL):
    """Evaluate PINN (u, v) on a regular grid at time t_val."""
    model.eval()
    xs = np.linspace(x_bounds[0], x_bounds[1], grid_nx, dtype=np.float32)
    ys = np.linspace(y_bounds[0], y_bounds[1], grid_ny, dtype=np.float32)
    Xi, Yi = np.meshgrid(xs, ys)

    x_t = torch.as_tensor(Xi.reshape(-1, 1), device=device)
    y_t = torch.as_tensor(Yi.reshape(-1, 1), device=device)
    t_t = torch.full_like(x_t, float(t_val))

    xyt = torch.cat([x_t, y_t, t_t], 1).requires_grad_(True)
    psi_p = model(xyt)
    psi   = psi_p[:, 0:1]
    dpsi  = _grad(psi, xyt)
    u = dpsi[:, 1:2].detach().cpu().numpy().reshape(grid_ny, grid_nx)
    v = (-dpsi[:, 0:1]).detach().cpu().numpy().reshape(grid_ny, grid_nx)

    mask = (Xi ** 2 + Yi ** 2) < R ** 2
    u[mask] = np.nan
    v[mask] = np.nan
    return Xi, Yi, u, v


def interpolate_fem_to_grid(snap: dict, x_bounds=XBOUNDS, y_bounds=YBOUNDS,
                             grid_nx=GRID_NX, grid_ny=GRID_NY, R=R_CYL):
    """Triangulate FEM snapshot and interpolate to regular grid."""
    xy = snap['xy']; U = snap['U']
    outside = (xy[:, 0] ** 2 + xy[:, 1] ** 2) >= R ** 2
    xy, U = xy[outside], U[outside]

    xs = np.linspace(x_bounds[0], x_bounds[1], grid_nx)
    ys = np.linspace(y_bounds[0], y_bounds[1], grid_ny)
    Xi, Yi = np.meshgrid(xs, ys)

    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    Ui  = np.array(mtri.LinearTriInterpolator(tri, U[:, 0])(Xi, Yi))
    Vi  = np.array(mtri.LinearTriInterpolator(tri, U[:, 1])(Xi, Yi))

    mask = (Xi ** 2 + Yi ** 2) < R ** 2
    Ui[mask] = np.nan
    Vi[mask] = np.nan
    return Xi, Yi, Ui, Vi


def field_rel_l2(fem, pinn):
    """Relative L2 error between two 2-D arrays (NaN-safe)."""
    valid = ~np.isnan(fem) & ~np.isnan(pinn)
    err   = pinn[valid] - fem[valid]
    return float(np.linalg.norm(err) / (np.linalg.norm(fem[valid]) + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Cd / Cl computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_cdcl_from_vtk(snap: dict, Re: float = 60.0,
                           R: float = R_CYL, n_theta: int = 512):
    """
    Approximate Cd and Cl from a VTK snapshot by interpolating p and ∂u/∂n
    onto the cylinder surface and integrating.

    Returns (Cd, Cl).
    """
    xy = snap['xy']; U = snap['U']; p = snap['p']

    # Build triangulation for pressure and velocity
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    p_interp = mtri.LinearTriInterpolator(tri, p)
    u_interp = mtri.LinearTriInterpolator(tri, U[:, 0])
    v_interp = mtri.LinearTriInterpolator(tri, U[:, 1])

    theta = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
    eps   = 1e-3         # small offset outside cylinder
    xc    = (R + eps) * np.cos(theta)
    yc    = (R + eps) * np.sin(theta)
    nx    = np.cos(theta)   # outward normal (from fluid to solid → inward)
    ny    = np.sin(theta)

    p_surf  = np.array(p_interp(xc, yc), dtype=np.float64)
    nan_m   = np.isnan(p_surf)
    if nan_m.all():
        return np.nan, np.nan

    # Pressure contribution: F = -∮ p n dA  (dA = R dθ)
    dtheta  = 2 * math.pi / n_theta
    Fx_pres = -np.nansum(p_surf * nx) * R * dtheta
    Fy_pres = -np.nansum(p_surf * ny) * R * dtheta

    # Viscous: τ = (1/Re)(∇u + (∇u)^T) · n  — approximate via FD in normal dir
    xc2 = (R + 2 * eps) * np.cos(theta)
    yc2 = (R + 2 * eps) * np.sin(theta)
    u1  = np.array(u_interp(xc, yc), dtype=np.float64)
    u2  = np.array(u_interp(xc2, yc2), dtype=np.float64)
    v1  = np.array(v_interp(xc, yc), dtype=np.float64)
    v2  = np.array(v_interp(xc2, yc2), dtype=np.float64)
    dudn = (u2 - u1) / eps
    dvdn = (v2 - v1) / eps

    nu = 1.0 / Re
    tau_x = nu * dudn
    tau_y = nu * dvdn
    Fx_visc = np.nansum(tau_x) * R * dtheta
    Fy_visc = np.nansum(tau_y) * R * dtheta

    Fx = Fx_pres + Fx_visc
    Fy = Fy_pres + Fy_visc

    # Non-dimensionalise by (½ ρ U² D): ρ=1, U=1, D=2R
    D  = 2 * R
    Cd = Fx / (0.5 * D)
    Cl = Fy / (0.5 * D)
    return float(Cd), float(Cl)


def compute_cdcl_from_pinn(model, t_val: float, device, Re: float = 60.0,
                            R: float = R_CYL, n_theta: int = 512):
    """
    Compute Cd and Cl from the PINN by integrating the surface traction.
    Uses automatic differentiation to get pressure and velocity derivatives.
    """
    model.eval()
    theta_np = np.linspace(0, 2 * math.pi, n_theta, endpoint=False, dtype=np.float32)
    xc = (R * np.cos(theta_np)).reshape(-1, 1)
    yc = (R * np.sin(theta_np)).reshape(-1, 1)
    tc = np.full_like(xc, float(t_val))

    x_t = torch.as_tensor(xc, device=device).requires_grad_(True)
    y_t = torch.as_tensor(yc, device=device).requires_grad_(True)
    t_t = torch.as_tensor(tc, device=device)

    xyt = torch.cat([x_t, y_t, t_t], 1).requires_grad_(True)
    out = model(xyt)
    psi = out[:, 0:1]
    p   = out[:, 1:2]

    dpsi = torch.autograd.grad(psi.sum(), xyt, create_graph=True)[0]
    u = dpsi[:, 1:2]
    v = -dpsi[:, 0:1]

    p_surf = p.detach().cpu().numpy().ravel()
    u_surf = u.detach().cpu().numpy().ravel()
    v_surf = v.detach().cpu().numpy().ravel()

    # Compute ∂u/∂n via AD
    # n̂ = (cos θ, sin θ)  (outward from cylinder into fluid)
    nx_t = torch.as_tensor(np.cos(theta_np).reshape(-1, 1), device=device)
    ny_t = torch.as_tensor(np.sin(theta_np).reshape(-1, 1), device=device)

    du = torch.autograd.grad(u.sum(), xyt, create_graph=False, retain_graph=True)[0]
    dv = torch.autograd.grad(v.sum(), xyt, create_graph=False, retain_graph=False)[0]
    du_dn = (du[:, 0:1] * nx_t + du[:, 1:2] * ny_t).detach().cpu().numpy().ravel()
    dv_dn = (dv[:, 0:1] * nx_t + dv[:, 1:2] * ny_t).detach().cpu().numpy().ravel()

    nx_np = np.cos(theta_np)
    ny_np = np.sin(theta_np)
    dtheta = 2 * math.pi / n_theta
    nu = 1.0 / Re

    Fx_pres = -np.sum(p_surf * nx_np) * R * dtheta
    Fy_pres = -np.sum(p_surf * ny_np) * R * dtheta
    Fx_visc =  np.sum(nu * du_dn)    * R * dtheta
    Fy_visc =  np.sum(nu * dv_dn)    * R * dtheta

    D  = 2 * R
    Cd = (Fx_pres + Fx_visc) / (0.5 * D)
    Cl = (Fy_pres + Fy_visc) / (0.5 * D)
    return float(Cd), float(Cl)


# ─────────────────────────────────────────────────────────────────────────────
# Period detection
# ─────────────────────────────────────────────────────────────────────────────
def estimate_period(snaps: List[dict], Re=60.0) -> float:
    """
    Estimate vortex-shedding period from the v_t.csv if present,
    otherwise from peak-to-peak of Cl computed from VTK snapshots.
    """
    # Try v_t.csv first
    csv_candidates = [
        os.path.join(os.path.dirname(snaps[0].get('_path', '')), '..', 'v_t.csv'),
        "Re60/v_t.csv",
    ]
    for csv_path in csv_candidates:
        if os.path.isfile(csv_path):
            data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
            t_arr = data[:, 0]; v_arr = data[:, 1]
            # find peaks via sign changes of derivative
            dv = np.diff(v_arr)
            peaks = np.where((dv[:-1] > 0) & (dv[1:] <= 0))[0] + 1
            if len(peaks) >= 2:
                T = float(np.mean(np.diff(t_arr[peaks])))
                print(f"[period] from v_t.csv: T={T:.4f}")
                return T

    # Fallback: compute Cl and find period
    print("[period] computing Cl from VTK snapshots …")
    t_arr = np.array([s['t'] for s in snaps])
    cl_arr = np.array([compute_cdcl_from_vtk(s, Re=Re)[1] for s in snaps])
    dv = np.diff(cl_arr)
    peaks = np.where((dv[:-1] > 0) & (dv[1:] <= 0))[0] + 1
    if len(peaks) >= 2:
        T = float(np.mean(np.diff(t_arr[peaks])))
    else:
        T = t_arr[-1] - t_arr[0]
    print(f"[period] estimated T={T:.4f}")
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Field-comparison panel (one velocity component)
# ─────────────────────────────────────────────────────────────────────────────
def _cylinder_patch():
    theta = np.linspace(0, 2 * np.pi, 200)
    return R_CYL * np.cos(theta), R_CYL * np.sin(theta)


def draw_field_panel(axes_row_fem, axes_row_pinn, axes_row_err,
                     snap_list, model, device, component: str,
                     t_period: float, t0_snap: float,
                     clim: float = 1.5):
    """
    Fill 3 × N_snap axes with FEM / PINN / |error| for one velocity component.

    Parameters
    ----------
    axes_row_fem, axes_row_pinn, axes_row_err : array of Axes, shape (N_snap,)
    snap_list  : list of VTK snapshot dicts, length N_snap
    component  : "u" or "v"
    t_period   : vortex-shedding period T
    t0_snap    : physical time of the first snapshot  (used for phase label)
    clim       : colour limit for velocity field (symmetric)
    """
    c_idx = 0 if component == "u" else 1
    cx, cy = _cylinder_patch()

    norm_vel = TwoSlopeNorm(vmin=-clim, vcenter=0, vmax=clim)
    norm_err = Normalize(vmin=0, vmax=clim * 0.3)

    imgs = []   # collect for shared colorbar

    rel_l2_mean = 0.0

    for col, snap in enumerate(snap_list):
        t_val   = snap['t']
        phase   = (t_val - t0_snap) / t_period  # fractional phase
        t_label = f"t={t_val - t0_snap:.2f} ({phase:.3f})"

        # FEM on grid
        Xi, Yi, Uf, Vf = interpolate_fem_to_grid(snap)
        fem_field = Uf if component == "u" else Vf

        # PINN on grid
        Xi, Yi, Up, Vp = eval_uv_on_grid(model, t_val, device)
        pinn_field = Up if component == "u" else Vp

        # Rel-L2 at this snapshot
        rl2 = field_rel_l2(fem_field, pinn_field)
        rel_l2_mean += rl2 / len(snap_list)

        # ── FEM row ──────────────────────────────────────────────────────────
        ax = axes_row_fem[col]
        im = ax.pcolormesh(Xi, Yi, fem_field, cmap=CMAP_FIELD, norm=norm_vel,
                           shading='auto', rasterized=True)
        ax.fill(cx, cy, color='gray', zorder=4)
        ax.set_aspect('equal')
        ax.set_xlim(XBOUNDS); ax.set_ylim(YBOUNDS)
        ax.set_title(t_label, fontsize=6, pad=1)
        ax.set_xticks([]); ax.set_yticks([])
        imgs.append(im)

        # ── PINN row ─────────────────────────────────────────────────────────
        ax2 = axes_row_pinn[col]
        ax2.pcolormesh(Xi, Yi, pinn_field, cmap=CMAP_FIELD, norm=norm_vel,
                       shading='auto', rasterized=True)
        ax2.fill(cx, cy, color='gray', zorder=4)
        ax2.set_aspect('equal')
        ax2.set_xlim(XBOUNDS); ax2.set_ylim(YBOUNDS)
        ax2.set_title(f"rel-L2={rl2:.4f}", fontsize=5, pad=1)
        ax2.set_xticks([]); ax2.set_yticks([])

        # ── Error row ─────────────────────────────────────────────────────────
        ax3 = axes_row_err[col]
        err = np.abs(pinn_field - fem_field)
        ax3.pcolormesh(Xi, Yi, err, cmap=CMAP_ERR, norm=norm_err,
                       shading='auto', rasterized=True)
        ax3.fill(cx, cy, color='gray', zorder=4)
        ax3.set_aspect('equal')
        ax3.set_xlim(XBOUNDS); ax3.set_ylim(YBOUNDS)
        ax3.set_xticks([]); ax3.set_yticks([])

    return imgs, rel_l2_mean


# ─────────────────────────────────────────────────────────────────────────────
# Cd/Cl panel
# ─────────────────────────────────────────────────────────────────────────────
def draw_cdcl_panel(ax_cd, ax_cl, snaps_period, model, device,
                    Re=60.0, t0=None, t_period=None):
    """
    Plot Cd and Cl time series: FEM (from VTKs) and PINN (via AD).
    """
    t_vals = np.array([s['t'] for s in snaps_period])
    if t0 is None: t0 = t_vals[0]

    print("  Computing FEM Cd/Cl …")
    cdcl_fem = [compute_cdcl_from_vtk(s, Re=Re) for s in snaps_period]
    cd_fem, cl_fem = zip(*cdcl_fem)

    print("  Computing PINN Cd/Cl …")
    cdcl_pin = [compute_cdcl_from_pinn(model, s['t'], device, Re=Re)
                for s in snaps_period]
    cd_pin, cl_pin = zip(*cdcl_pin)

    t_rel = t_vals - t0

    # Closure error = max abs diff at endpoints
    def _closure_err(arr_a, arr_b):
        a, b = np.array(arr_a), np.array(arr_b)
        return 0.5 * (abs(a[0] - b[0]) + abs(a[-1] - b[-1]))

    cd_ferr = _closure_err(cd_fem, cd_pin)
    cl_ferr = _closure_err(cl_fem, cl_pin)

    ax_cd.plot(t_rel, cd_fem, 'k-',  label='FEM', lw=1.0)
    ax_cd.plot(t_rel, cd_pin, 'r--', label='PINN', lw=1.0)
    ax_cd.set_ylabel("$C_d$", fontsize=7)
    ax_cd.legend(loc='upper right')
    ax_cd.set_title(f"FEM vs PINN over one period  (closure err={cd_ferr:.4f})",
                    fontsize=7)

    ax_cl.plot(t_rel, cl_fem, 'k-',  lw=1.0)
    ax_cl.plot(t_rel, cl_pin, 'r--', lw=1.0)
    ax_cl.set_ylabel("$C_l$", fontsize=7)
    ax_cl.set_xlabel("time", fontsize=7)

    if t_period is not None:
        for ax in [ax_cd, ax_cl]:
            ax.axvline(0, color='r', ls=':', lw=0.6)
            ax.axvline(t_period, color='r', ls=':', lw=0.6)


# ─────────────────────────────────────────────────────────────────────────────
# Convergence panel
# ─────────────────────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "SOAP+PT+ModifiedMLP (full)" : "#2ca02c",
    "SOAP, no pseudo-time"        : "#ff7f0e",
    "Adam + pseudo-time"          : "#d62728",
    "Shampoo + pseudo-time"       : "#9467bd",
    "plain MLP (SOAP+PT)"         : "#8c564b",
}
METHOD_LS = {
    "SOAP+PT+ModifiedMLP (full)" : "-",
    "SOAP, no pseudo-time"        : "--",
    "Adam + pseudo-time"          : "-.",
    "Shampoo + pseudo-time"       : ":",
    "plain MLP (SOAP+PT)"         : (0, (3, 1, 1, 1)),
}


def draw_convergence_panel(ax, loss_data: Dict[str, np.ndarray]):
    """
    loss_data : {method_name: 1-D array of total loss per step}
    """
    for name, losses in loss_data.items():
        c  = METHOD_COLORS.get(name, None)
        ls = METHOD_LS.get(name, "-")
        steps = np.arange(1, len(losses) + 1) * 100  # assume logged every 100 steps
        ax.semilogy(steps, losses, label=name, color=c, ls=ls, lw=0.9)

    ax.set_xlabel("training step", fontsize=7)
    ax.set_ylabel("total loss (MSE)", fontsize=7)
    ax.set_title("Convergence", fontsize=7)
    ax.legend(fontsize=5, loc='upper right')


# ─────────────────────────────────────────────────────────────────────────────
# Ablation bar chart
# ─────────────────────────────────────────────────────────────────────────────
def draw_ablation_panel(ax, ablation_data: Dict[str, float],
                        title="Ablation: mean field rel-L2 (over period)"):
    """
    ablation_data : {method_name: mean_rel_l2}
    """
    methods = list(ablation_data.keys())
    values  = [ablation_data[m] for m in methods]
    colors  = [METHOD_COLORS.get(m, "#aec7e8") for m in methods]
    short   = [m.replace("SOAP+PT+ModifiedMLP (full)", "SOAP+PT\nModMLP(full)")
                .replace("SOAP, no pseudo-time", "SOAP\nno PT")
                .replace("Adam + pseudo-time", "Adam\n+PT")
                .replace("Shampoo + pseudo-time", "Shampoo\n+PT")
                .replace("plain MLP (SOAP+PT)", "plain MLP\n(SOAP+PT)")
               for m in methods]

    bars = ax.bar(range(len(methods)), values, color=colors, width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(short, fontsize=5)
    ax.set_ylabel("mean field rel-L2", fontsize=7)
    ax.set_title(title, fontsize=6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.002,
                f"{val:.4f}", ha='center', va='bottom', fontsize=5)


# ─────────────────────────────────────────────────────────────────────────────
# Error-over-period panel
# ─────────────────────────────────────────────────────────────────────────────
def draw_error_period_panel(ax, error_data: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """
    error_data : {method_name: (t_over_T, rel_l2_arr)}
    """
    for name, (tT, err) in error_data.items():
        c  = METHOD_COLORS.get(name, None)
        ls = METHOD_LS.get(name, "-")
        mean_err = np.mean(err)
        ax.plot(tT, err, label=f"{name} (mean={mean_err:.4f})",
                color=c, ls=ls, lw=0.9)
    ax.set_xlabel("t / T", fontsize=7)
    ax.set_ylabel("field rel-L2", fontsize=7)
    ax.set_title("Error over the period", fontsize=7)
    ax.legend(fontsize=4, loc='upper left')
    ax.set_xlim(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Full figure
# ─────────────────────────────────────────────────────────────────────────────
def make_full_figure(
    snaps_period: List[dict],
    model: nn.Module,
    device: torch.device,
    n_snap: int = 5,
    Re: float = 60.0,
    t_period: float = None,
    run_label: str = "SOAP+PT+ModifiedMLP (full)",
    # Optional comparison data from other runs (for convergence / ablation)
    loss_data: Optional[Dict[str, np.ndarray]] = None,
    ablation_data: Optional[Dict[str, float]] = None,
    error_period_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    out_path: str = "mentor_style_figure.png",
    dpi: int = 200,
):
    """
    Create the full mentor-style figure and save to out_path.

    Parameters
    ----------
    snaps_period : list of VTK dicts covering (at least) one full shedding period
    model        : trained PINN
    n_snap       : number of time snapshots to display (≤ len(snaps_period))
    Re           : Reynolds number
    t_period     : shedding period T (estimated automatically if None)
    run_label    : name for the main run (used in titles and as ablation key)
    loss_data    : {method: loss_array}  — for convergence panel
    ablation_data: {method: mean_rel_l2} — for bar chart
    error_period_data : {method: (tT, err)} — for error-over-period panel
    """
    if t_period is None:
        t_period = estimate_period(snaps_period, Re=Re)

    # Select n_snap evenly spaced across one period
    step   = max(1, len(snaps_period) // n_snap)
    chosen = snaps_period[::step][:n_snap]
    t0     = chosen[0]['t']

    # ── Layout ────────────────────────────────────────────────────────────────
    # Top section:  2 panels (u-field, v-field), each 3×N_snap
    # Bottom section: 4 panels side by side
    fig = plt.figure(figsize=(14, 7.5), dpi=dpi)
    outer = gridspec.GridSpec(2, 1, figure=fig,
                               height_ratios=[1.45, 1.0],
                               hspace=0.35)

    # ── Top: field comparison panels ─────────────────────────────────────────
    top_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], wspace=0.06)

    def make_field_gs(parent):
        return gridspec.GridSpecFromSubplotSpec(
            3, n_snap, subplot_spec=parent,
            hspace=0.04, wspace=0.04,
            height_ratios=[1, 1, 0.6])

    gs_u = make_field_gs(top_gs[0])
    gs_v = make_field_gs(top_gs[1])

    def get_row_axes(gs, row):
        return [fig.add_subplot(gs[row, c]) for c in range(n_snap)]

    # u-field axes
    ax_u_fem  = get_row_axes(gs_u, 0)
    ax_u_pinn = get_row_axes(gs_u, 1)
    ax_u_err  = get_row_axes(gs_u, 2)
    # v-field axes
    ax_v_fem  = get_row_axes(gs_v, 0)
    ax_v_pinn = get_row_axes(gs_v, 1)
    ax_v_err  = get_row_axes(gs_v, 2)

    # Row labels
    ax_u_fem[0].set_ylabel("FEM", fontsize=6)
    ax_u_pinn[0].set_ylabel("PINN", fontsize=6)
    ax_u_err[0].set_ylabel("|error|", fontsize=6)
    ax_v_fem[0].set_ylabel("FEM", fontsize=6)
    ax_v_pinn[0].set_ylabel("PINN", fontsize=6)
    ax_v_err[0].set_ylabel("|error|", fontsize=6)

    print("[plot] Drawing u-field panels …")
    imgs_u, rl2_u = draw_field_panel(
        ax_u_fem, ax_u_pinn, ax_u_err,
        chosen, model, device, "u",
        t_period=t_period, t0_snap=t0)

    print("[plot] Drawing v-field panels …")
    imgs_v, rl2_v = draw_field_panel(
        ax_v_fem, ax_v_pinn, ax_v_err,
        chosen, model, device, "v",
        t_period=t_period, t0_snap=t0)

    mean_rl2 = 0.5 * (rl2_u + rl2_v)

    # Panel titles
    title_kw = dict(fontsize=7, fontweight='normal', loc='left', pad=3)
    ax_u_fem[0].get_figure().text(
        ax_u_fem[0].get_position().x0, 1.01,
        f"FEM vs PINN ({run_label}) – u over one period; "
        f"mean field rel-L2 = {rl2_u:.4f} (T={t_period:.3f})",
        transform=fig.transFigure, fontsize=6.5, va='bottom')
    ax_v_fem[0].get_figure().text(
        ax_v_fem[0].get_position().x0, 1.01,
        f"FEM vs PINN ({run_label}) – v over one period; "
        f"mean field rel-L2 = {rl2_v:.4f} (T={t_period:.3f})",
        transform=fig.transFigure, fontsize=6.5, va='bottom')

    # Shared colorbars (one per field panel)
    cb_u = fig.colorbar(imgs_u[0], ax=ax_u_err, shrink=0.8,
                         pad=0.01, location='right')
    cb_u.ax.tick_params(labelsize=5)
    cb_v = fig.colorbar(imgs_v[0], ax=ax_v_err, shrink=0.8,
                         pad=0.01, location='right')
    cb_v.ax.tick_params(labelsize=5)

    # ── Bottom: 4 sub-panels ─────────────────────────────────────────────────
    bot_gs = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1], wspace=0.40)

    # Cd/Cl (2 sub-rows)
    cdcl_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=bot_gs[0], hspace=0.15)
    ax_cd = fig.add_subplot(cdcl_gs[0])
    ax_cl = fig.add_subplot(cdcl_gs[1])

    ax_conv   = fig.add_subplot(bot_gs[1])
    ax_ablat  = fig.add_subplot(bot_gs[2])
    ax_errper = fig.add_subplot(bot_gs[3])

    # ── Cd/Cl panel ──────────────────────────────────────────────────────────
    print("[plot] Computing Cd/Cl …")
    draw_cdcl_panel(ax_cd, ax_cl, chosen, model, device,
                    Re=Re, t0=t0, t_period=t_period)

    # ── Convergence panel ────────────────────────────────────────────────────
    if loss_data is not None:
        draw_convergence_panel(ax_conv, loss_data)
    else:
        ax_conv.text(0.5, 0.5, "No loss data\n(pass --loss-csv)",
                     ha='center', va='center', transform=ax_conv.transAxes,
                     fontsize=7, color='gray')
        ax_conv.set_title("Convergence", fontsize=7)

    # ── Ablation panel ───────────────────────────────────────────────────────
    if ablation_data is None:
        ablation_data = {run_label: mean_rl2}
    # Make sure the current run is in the dict
    ablation_data.setdefault(run_label, mean_rl2)
    draw_ablation_panel(ax_ablat, ablation_data)

    # ── Error-over-period panel ───────────────────────────────────────────────
    if error_period_data is None:
        # compute for the current run from the chosen snapshots
        tT_arr  = np.array([(s['t'] - t0) / t_period for s in chosen])
        rl2_arr = []
        print("[plot] Computing per-snapshot errors for error-over-period …")
        for snap in chosen:
            Xi, Yi, Uf, Vf = interpolate_fem_to_grid(snap)
            Xi, Yi, Up, Vp = eval_uv_on_grid(model, snap['t'], device)
            e = 0.5 * (field_rel_l2(Uf, Up) + field_rel_l2(Vf, Vp))
            rl2_arr.append(e)
        error_period_data = {run_label: (tT_arr, np.array(rl2_arr))}

    draw_error_period_panel(ax_errper, error_period_data)

    # ── Super-title ───────────────────────────────────────────────────────────
    fig.suptitle(f"PINN for vortex shedding (Re={Re:.0f}) – {run_label}",
                 fontsize=9, fontweight='bold', y=1.005)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Load helper
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device,
               arch: str = "mlp", width: int = 32, depth: int = 3) -> nn.Module:
    if arch.lower() == "modifiedmlp":
        model = ModifiedMLP(width=width, depth=depth).to(device)
    else:
        model = MLP(width=width, depth=depth).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    print(f"[load] {ckpt_path}  arch={arch}  w={width}  d={depth}")
    return model


def load_loss_csv(csv_path: str) -> Dict[str, np.ndarray]:
    """
    Load a loss CSV with columns: step, method, loss.
    Returns {method: loss_array}.
    """
    import csv
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    data: Dict[str, Dict[int, float]] = {}
    for row in rows:
        m = row.get('method', 'run')
        s = int(row.get('step', 0))
        l = float(row.get('loss', 0))
        data.setdefault(m, {})[s] = l
    out = {}
    for m, d in data.items():
        steps = sorted(d.keys())
        out[m] = np.array([d[s] for s in steps])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mentor-style PINN figure")
    parser.add_argument("--vtk-dir",  required=True,
                        help="Directory with Re60_XXXX.vtk files")
    parser.add_argument("--ckpt",     required=True,
                        help="PINN checkpoint (.pt)")
    parser.add_argument("--arch",     default="mlp",
                        choices=["mlp", "modifiedmlp"])
    parser.add_argument("--width",    type=int, default=32)
    parser.add_argument("--depth",    type=int, default=3)
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--Re",       type=float, default=60.0)
    parser.add_argument("--n-snap",   type=int, default=5,
                        help="Number of time snapshots to display")
    parser.add_argument("--run-label", default="SOAP+PT+ModifiedMLP (full)")
    parser.add_argument("--loss-csv", default=None,
                        help="CSV with columns step,method,loss for convergence panel")
    parser.add_argument("--out",      default="figures/mentor_style.png")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    snaps = load_vtk_series(args.vtk_dir)
    model = load_model(args.ckpt, device,
                       arch=args.arch, width=args.width, depth=args.depth)

    loss_data = None
    if args.loss_csv and os.path.isfile(args.loss_csv):
        loss_data = load_loss_csv(args.loss_csv)

    make_full_figure(
        snaps_period=snaps,
        model=model,
        device=device,
        n_snap=args.n_snap,
        Re=args.Re,
        run_label=args.run_label,
        loss_data=loss_data,
        out_path=args.out,
    )
