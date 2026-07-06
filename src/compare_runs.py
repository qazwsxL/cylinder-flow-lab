"""
Compare two cfp40.py checkpoints side by side.

Typical usage (run from src/):
    python compare_runs.py \
        --vtk-path ../Re40.vtk \
        --ckpt-base ../runs/phase2/checkpoints_phase2/pinn_Re40_single.pt \
        --ckpt-pt   ../checkpoints_pt/pinn_Re40_single.pt \
        --label-base "No PT (phase2)" \
        --label-pt   "With PT" \
        --width 96 --depth 5 \
        --out-dir ../runs/compare_pt_vs_base
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── import from cfp40 in the same directory ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from cfp40 import (
    MLPStreamPressure,
    load_single_vtk,
    compute_pde_residuals,
    uvp_from_psip,
    safe_load_checkpoint,
)


# =============================================================================
# Helpers
# =============================================================================

def load_model(ckpt_path, width, depth, device):
    model = MLPStreamPressure(width=width, depth=depth).to(device)
    state = safe_load_checkpoint(ckpt_path, device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def eval_on_grid(model, xs, ys, t_val, device):
    """Evaluate model on meshgrid (xs, ys). Returns u, v, p, omega (vorticity)."""
    X, Y = np.meshgrid(xs, ys)
    xf = torch.tensor(X.ravel(), dtype=torch.float32, device=device)
    yf = torch.tensor(Y.ravel(), dtype=torch.float32, device=device)
    tf = torch.full_like(xf, t_val)

    xf.requires_grad_(True)
    yf.requires_grad_(True)
    u, v, p = uvp_from_psip(model, xf, yf, tf)

    # vorticity = dv/dx - du/dy
    du_dy = torch.autograd.grad(u.sum(), yf, create_graph=False)[0]
    dv_dx = torch.autograd.grad(v.sum(), xf, create_graph=False)[0]
    omega = (dv_dx - du_dy).cpu().numpy().reshape(X.shape)

    u = u.detach().cpu().numpy().reshape(X.shape)
    v = v.detach().cpu().numpy().reshape(X.shape)
    p = p.detach().cpu().numpy().reshape(X.shape)
    return X, Y, u, v, p, omega


@torch.no_grad()
def pde_residual_on_grid(model, xs, ys, t_val, Re, device):
    """Mean |f_u|, |f_v|, |div| on a grid (inside domain, outside cylinder)."""
    X, Y = np.meshgrid(xs, ys)
    mask = X**2 + Y**2 > 0.5**2          # outside cylinder
    xf = torch.tensor(X[mask], dtype=torch.float32, device=device).requires_grad_(True)
    yf = torch.tensor(Y[mask], dtype=torch.float32, device=device).requires_grad_(True)
    tf = torch.full_like(xf, t_val)

    f_u, f_v, div = compute_pde_residuals(model, xf, yf, tf, Re=Re)
    f_u = f_u.detach().cpu().numpy()
    f_v = f_v.detach().cpu().numpy()
    div = div.detach().cpu().numpy()

    # Map back to full grid
    fu_grid = np.full(X.shape, np.nan)
    fv_grid = np.full(X.shape, np.nan)
    div_grid = np.full(X.shape, np.nan)
    fu_grid[mask]  = f_u
    fv_grid[mask]  = f_v
    div_grid[mask] = div
    return fu_grid, fv_grid, div_grid


def mask_cylinder(arr, X, Y, r=0.5):
    inside = X**2 + Y**2 <= r**2
    out = arr.copy()
    out[inside] = np.nan
    return out


def cfd_on_grid(snapshot, xs, ys):
    """Bilinear interpolation of CFD data onto the eval grid."""
    from scipy.interpolate import griddata
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([snapshot.x.numpy(), snapshot.y.numpy()], axis=1)
    u_cfd = griddata(pts, snapshot.u.numpy(), (X, Y), method="linear")
    v_cfd = griddata(pts, snapshot.v.numpy(), (X, Y), method="linear")
    return X, Y, u_cfd, v_cfd


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(model, snapshot, xs, ys, t_val, Re, device):
    X, Y, u, v, p, omega = eval_on_grid(model, xs, ys, t_val, device)
    _, _, u_cfd, v_cfd = cfd_on_grid(snapshot, xs, ys)
    fu, fv, div = pde_residual_on_grid(model, xs, ys, t_val, Re, device)

    # Only compare where CFD data is available and outside cylinder
    outside = (X**2 + Y**2 > 0.5**2)
    valid = outside & ~np.isnan(u_cfd) & ~np.isnan(u)

    rmse_u  = np.sqrt(np.nanmean((u[valid] - u_cfd[valid])**2))
    rmse_v  = np.sqrt(np.nanmean((v[valid] - v_cfd[valid])**2))
    mae_fu  = np.nanmean(np.abs(fu))
    mae_fv  = np.nanmean(np.abs(fv))
    mae_div = np.nanmean(np.abs(div))
    pde_total = mae_fu + mae_fv + mae_div

    return dict(
        rmse_u=rmse_u, rmse_v=rmse_v,
        mae_fu=mae_fu, mae_fv=mae_fv, mae_div=mae_div,
        pde_total=pde_total,
        X=X, Y=Y, u=u, v=v, omega=omega,
        u_cfd=u_cfd, v_cfd=v_cfd,
        fu=fu, fv=fv, div=div,
    )


# =============================================================================
# Plot
# =============================================================================

def make_comparison_figure(m_base, m_pt, label_base, label_pt, out_path):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f"Comparison: {label_base}  vs  {label_pt}", fontsize=13, y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

    X, Y = m_base["X"], m_base["Y"]

    def _im(ax, data, title, cmap="RdBu_r", vabs=None, label=""):
        data_m = mask_cylinder(data, X, Y)
        if vabs is None:
            vmax = np.nanpercentile(np.abs(data_m), 98)
        else:
            vmax = vabs
        im = ax.pcolormesh(X, Y, data_m, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
        cyl = plt.Circle((0, 0), 0.5, color="white", zorder=5)
        ax.add_patch(cyl)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        if label:
            ax.text(0.02, 0.95, label, transform=ax.transAxes,
                    fontsize=8, va="top", color="k",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    # Row 0: vorticity
    ax = fig.add_subplot(gs[0, 0])
    _im(ax, m_base["omega"], f"Vorticity — {label_base}", cmap="RdBu_r")
    ax = fig.add_subplot(gs[0, 1])
    _im(ax, m_pt["omega"],   f"Vorticity — {label_pt}",   cmap="RdBu_r")
    ax = fig.add_subplot(gs[0, 2])
    _im(ax, m_base["u_cfd"] - m_base["u"], f"u error — {label_base}", cmap="RdBu_r")
    ax = fig.add_subplot(gs[0, 3])
    _im(ax, m_pt["u_cfd"]   - m_pt["u"],   f"u error — {label_pt}",   cmap="RdBu_r")

    # Row 1: v error + PDE f_u residual
    ax = fig.add_subplot(gs[1, 0])
    _im(ax, m_base["v_cfd"] - m_base["v"], f"v error — {label_base}", cmap="RdBu_r")
    ax = fig.add_subplot(gs[1, 1])
    _im(ax, m_pt["v_cfd"]   - m_pt["v"],   f"v error — {label_pt}",   cmap="RdBu_r")
    ax = fig.add_subplot(gs[1, 2])
    _im(ax, m_base["fu"], f"|f_u| — {label_base}", cmap="hot")
    ax = fig.add_subplot(gs[1, 3])
    _im(ax, m_pt["fu"],   f"|f_u| — {label_pt}",   cmap="hot")

    # Row 2: div residual + metrics text
    ax = fig.add_subplot(gs[2, 0])
    _im(ax, m_base["div"], f"|div| — {label_base}", cmap="hot")
    ax = fig.add_subplot(gs[2, 1])
    _im(ax, m_pt["div"],   f"|div| — {label_pt}",   cmap="hot")

    # Metrics summary
    ax_txt = fig.add_subplot(gs[2, 2:])
    ax_txt.axis("off")
    rows = [
        ["Metric", label_base, label_pt, "Δ (PT − base)"],
        ["RMSE u",       f"{m_base['rmse_u']:.4f}",    f"{m_pt['rmse_u']:.4f}",
         f"{m_pt['rmse_u']-m_base['rmse_u']:+.4f}"],
        ["RMSE v",       f"{m_base['rmse_v']:.4f}",    f"{m_pt['rmse_v']:.4f}",
         f"{m_pt['rmse_v']-m_base['rmse_v']:+.4f}"],
        ["MAE f_u",      f"{m_base['mae_fu']:.4f}",    f"{m_pt['mae_fu']:.4f}",
         f"{m_pt['mae_fu']-m_base['mae_fu']:+.4f}"],
        ["MAE f_v",      f"{m_base['mae_fv']:.4f}",    f"{m_pt['mae_fv']:.4f}",
         f"{m_pt['mae_fv']-m_base['mae_fv']:+.4f}"],
        ["MAE div",      f"{m_base['mae_div']:.4f}",   f"{m_pt['mae_div']:.4f}",
         f"{m_pt['mae_div']-m_base['mae_div']:+.4f}"],
        ["PDE total",    f"{m_base['pde_total']:.4f}", f"{m_pt['pde_total']:.4f}",
         f"{m_pt['pde_total']-m_base['pde_total']:+.4f}"],
    ]
    tbl = ax_txt.table(cellText=rows[1:], colLabels=rows[0],
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    # Colour the Δ column: green = better (negative), red = worse
    for i in range(1, len(rows)):
        try:
            val = float(rows[i][3])
            colour = "#d4f1c4" if val < 0 else "#f7c6c6"
            tbl[i, 3].set_facecolor(colour)
        except ValueError:
            pass

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[compare] saved → {out_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vtk-path",    required=True)
    p.add_argument("--ckpt-base",   required=True,
                   help="Checkpoint WITHOUT pseudo-time stepping (baseline)")
    p.add_argument("--ckpt-pt",     required=True,
                   help="Checkpoint WITH pseudo-time stepping")
    p.add_argument("--label-base",  default="No PT")
    p.add_argument("--label-pt",    default="With PT")
    p.add_argument("--width",       type=int, default=96)
    p.add_argument("--depth",       type=int, default=5)
    p.add_argument("--Re",          type=float, default=40.0)
    p.add_argument("--t-value",     type=float, default=0.0)
    p.add_argument("--out-dir",     default="../runs/compare_pt_vs_base")
    p.add_argument("--nx",          type=int, default=300,
                   help="Grid resolution (x)")
    p.add_argument("--ny",          type=int, default=200,
                   help="Grid resolution (y)")
    p.add_argument("--x-min",       type=float, default=-3.0)
    p.add_argument("--x-max",       type=float, default=12.0)
    p.add_argument("--y-min",       type=float, default=-4.0)
    p.add_argument("--y-max",       type=float, default=4.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compare] device={device}")

    snapshot = load_single_vtk(args.vtk_path, t_value=args.t_value)

    xs = np.linspace(args.x_min, args.x_max, args.nx)
    ys = np.linspace(args.y_min, args.y_max, args.ny)

    print(f"[compare] loading baseline: {args.ckpt_base}")
    model_base = load_model(args.ckpt_base, args.width, args.depth, device)
    print(f"[compare] loading PT model: {args.ckpt_pt}")
    model_pt   = load_model(args.ckpt_pt,   args.width, args.depth, device)

    print("[compare] evaluating baseline …")
    m_base = compute_metrics(model_base, snapshot, xs, ys,
                             args.t_value, args.Re, device)
    print("[compare] evaluating PT model …")
    m_pt   = compute_metrics(model_pt,   snapshot, xs, ys,
                             args.t_value, args.Re, device)

    # Print table
    print("\n" + "="*60)
    print(f"{'Metric':<14} {'Baseline':>12} {'PT':>12} {'Δ':>12}")
    print("-"*60)
    for k in ["rmse_u", "rmse_v", "mae_fu", "mae_fv", "mae_div", "pde_total"]:
        delta = m_pt[k] - m_base[k]
        sign = "✓" if delta < 0 else "✗"
        print(f"{k:<14} {m_base[k]:>12.5f} {m_pt[k]:>12.5f} {delta:>+12.5f}  {sign}")
    print("="*60)

    # Save metrics to txt
    metrics_path = os.path.join(args.out_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"baseline: {args.ckpt_base}\n")
        f.write(f"pt model: {args.ckpt_pt}\n\n")
        f.write(f"{'Metric':<14} {'Baseline':>12} {'PT':>12} {'Δ':>12}\n")
        f.write("-"*60 + "\n")
        for k in ["rmse_u", "rmse_v", "mae_fu", "mae_fv", "mae_div", "pde_total"]:
            delta = m_pt[k] - m_base[k]
            f.write(f"{k:<14} {m_base[k]:>12.5f} {m_pt[k]:>12.5f} {delta:>+12.5f}\n")

    # Save figure
    fig_path = os.path.join(args.out_dir, "comparison.png")
    make_comparison_figure(m_base, m_pt, args.label_base, args.label_pt, fig_path)
    print(f"[compare] metrics → {metrics_path}")


if __name__ == "__main__":
    main()
