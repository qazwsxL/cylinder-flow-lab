"""
diagnose_cfd_pde.py
-------------------

Check whether the CFD ground-truth (u, v) in Re40.vtk is itself a solution of the
steady incompressible Navier-Stokes equations.

WHY THIS MATTERS
================
A PINN that fits CFD data and enforces the PDE simultaneously can never drive
the data loss below the level at which the CFD itself violates the PDE.
If the CFD residual on its own grid is, say, 1e-3, then the data loss of any
PINN that also satisfies the PDE will saturate at roughly that level too.
This script measures that floor.

WHAT IT DOES
============
On each interior CFD point we:
  1. find k nearest neighbours,
  2. fit a local quadratic polynomial in (dx, dy) to u and to v,
  3. read off u_x, u_y, u_xx, u_yy, u_xy and the same for v at the centre,
  4. compute
        - continuity:        c = u_x + v_y
        - vorticity:         omega = v_x - u_y
  5. fit a second local quadratic to omega to read off omega_x, omega_y,
     omega_xx, omega_yy at the centre,
  6. compute the steady vorticity-transport residual
        r_omega = u * omega_x + v * omega_y  -  (1/Re) * (omega_xx + omega_yy)
     (pressure has been eliminated by taking the curl of the momentum eqs).

We report mean / median / p90 / p99 / max of |c|, |omega|, |r_omega|, plus
PNG maps so you can see WHERE the CFD violates NS the most (typically thin
shear layers and the cylinder boundary, exactly where coarse meshes hurt).

USAGE
=====
    python diagnose_cfd_pde.py --vtk Re40.vtk --re 40 --k 24 \
        --xlim -3 12 --ylim -4 4 --out diag_cfd_pde

Output goes into the chosen folder:
    summary.txt           text table of stats
    residual_hist.png     histogram of |r_omega|
    residual_map.png      scatter coloured by |r_omega|
    continuity_map.png    scatter coloured by |c|
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

try:
    import pyvista as pv
except ImportError:
    pv = None


# ------------------------------------------------------------------
# IO
# ------------------------------------------------------------------

def load_vtk(path):
    if pv is None:
        raise ImportError("pyvista is required. pip install pyvista")
    mesh = pv.read(path)
    if mesh.n_cells > 0:
        centers = mesh.cell_centers().points
        data_src = mesh.cell_data
    else:
        centers = mesh.points
        data_src = mesh.point_data
    names_lower = {n.lower(): n for n in data_src.keys()}
    vel_name = None
    for cand in ("u", "velocity", "vel"):
        if cand in names_lower:
            vel_name = names_lower[cand]
            break
    if vel_name is None:
        raise KeyError(f"no velocity field in {path}; have {list(data_src.keys())}")
    vel = np.asarray(data_src[vel_name])
    x = centers[:, 0].astype(np.float64)
    y = centers[:, 1].astype(np.float64)
    u = vel[:, 0].astype(np.float64)
    v = vel[:, 1].astype(np.float64)
    return x, y, u, v


# ------------------------------------------------------------------
# Local quadratic fit -> derivatives
# ------------------------------------------------------------------

def local_quadratic_derivs(x, y, f, tree, k=24, mask=None):
    """
    For every point in (x, y), fit f ~ a + b dx + c dy + d dx^2 + e dx dy + g dy^2
    on its k nearest neighbours. Returns f_x, f_y, f_xx, f_yy, f_xy at each point.
    Points where the fit is degenerate are flagged via a separate quality mask.

    Note: tree is built on the full (x, y); we ALWAYS query k neighbours from it,
    but we only return derivatives at points where mask is True (or all if mask is None).
    """
    n = len(x)
    if mask is None:
        idx_centers = np.arange(n)
    else:
        idx_centers = np.where(mask)[0]

    out = np.full((n, 5), np.nan, dtype=np.float64)  # f_x, f_y, f_xx, f_yy, f_xy
    quality = np.zeros(n, dtype=bool)

    pts = np.column_stack([x, y])

    for i in idx_centers:
        _, nbrs = tree.query(pts[i], k=k)
        dx = x[nbrs] - x[i]
        dy = y[nbrs] - y[i]
        # design matrix
        A = np.column_stack([
            np.ones_like(dx), dx, dy, dx * dx, dx * dy, dy * dy
        ])
        b = f[nbrs]
        # solve in least-squares sense
        try:
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        # at the centre (dx=dy=0):
        out[i, 0] = coef[1]            # f_x  = b
        out[i, 1] = coef[2]            # f_y  = c
        out[i, 2] = 2.0 * coef[3]      # f_xx = 2 d
        out[i, 3] = 2.0 * coef[5]      # f_yy = 2 g
        out[i, 4] = coef[4]            # f_xy = e
        quality[i] = True

    return out, quality


# ------------------------------------------------------------------
# Diagnostic
# ------------------------------------------------------------------

def run(args):
    os.makedirs(args.out, exist_ok=True)

    print(f"[load] {args.vtk}")
    x, y, u, v = load_vtk(args.vtk)
    n_total = len(x)
    print(f"       {n_total} CFD points")

    # restrict to PDE box, outside cylinder, with a small interior buffer so
    # the local quadratic fit is not biased by missing neighbours on a boundary.
    buf = args.boundary_buffer
    interior = (
        (x >= args.xlim[0] + buf) & (x <= args.xlim[1] - buf) &
        (y >= args.ylim[0] + buf) & (y <= args.ylim[1] - buf) &
        ((x ** 2 + y ** 2) >= (args.cyl_R + buf) ** 2)
    )
    n_int = int(interior.sum())
    print(f"[mask] {n_int}/{n_total} points used (interior, buffer={buf})")

    print(f"[tree] building KD-tree on {n_total} points")
    tree = cKDTree(np.column_stack([x, y]))

    print(f"[fit ] local quadratic for u (k={args.k})")
    du, qu = local_quadratic_derivs(x, y, u, tree, k=args.k, mask=interior)
    print(f"[fit ] local quadratic for v (k={args.k})")
    dv, qv = local_quadratic_derivs(x, y, v, tree, k=args.k, mask=interior)

    valid = qu & qv

    u_x  = du[:, 0]; u_y  = du[:, 1]
    u_xx = du[:, 2]; u_yy = du[:, 3]; u_xy = du[:, 4]
    v_x  = dv[:, 0]; v_y  = dv[:, 1]
    v_xx = dv[:, 2]; v_yy = dv[:, 3]; v_xy = dv[:, 4]

    continuity = u_x + v_y
    omega = v_x - u_y

    # second pass: derivatives of omega via local quadratic fit
    print(f"[fit ] local quadratic for omega (k={args.k})")
    # only meaningful where omega is defined
    omega_for_fit = np.where(valid, omega, 0.0)
    do, qo = local_quadratic_derivs(x, y, omega_for_fit, tree, k=args.k, mask=valid)
    omega_x  = do[:, 0]; omega_y  = do[:, 1]
    omega_xx = do[:, 2]; omega_yy = do[:, 3]

    nu = 1.0 / args.re
    r_omega = u * omega_x + v * omega_y - nu * (omega_xx + omega_yy)

    final_mask = valid & qo
    n_use = int(final_mask.sum())
    print(f"[res ] computed on {n_use} points")

    def stats(name, arr, mask):
        a = np.abs(arr[mask])
        return {
            "name": name,
            "mean":   float(a.mean()),
            "median": float(np.median(a)),
            "p90":    float(np.quantile(a, 0.90)),
            "p99":    float(np.quantile(a, 0.99)),
            "max":    float(a.max()),
        }

    summary = [
        stats("|continuity|",    continuity, final_mask),
        stats("|vorticity|",     omega,      final_mask),
        stats("|vort transport|", r_omega,    final_mask),
    ]

    # write summary
    lines = []
    lines.append(f"CFD-PDE residual diagnostic for {args.vtk}")
    lines.append(f"Re = {args.re}")
    lines.append(f"interior points used: {n_use}/{n_total}")
    lines.append(f"k-nearest neighbours: {args.k}")
    lines.append("")
    lines.append(f"{'quantity':<22s}{'mean':>14s}{'median':>14s}{'p90':>14s}{'p99':>14s}{'max':>14s}")
    for s in summary:
        lines.append(
            f"{s['name']:<22s}{s['mean']:14.3e}{s['median']:14.3e}{s['p90']:14.3e}{s['p99']:14.3e}{s['max']:14.3e}"
        )
    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("--------------")
    lines.append("|continuity|        : how well the CFD field is divergence-free.")
    lines.append("|vort transport|    : steady NS residual after eliminating pressure.")
    lines.append("                      This is the FLOOR for any PDE-respecting fit.")
    lines.append("                      If p99 of |vort transport| ~ 1e-3 then a PINN")
    lines.append("                      that ALSO satisfies the PDE cannot get data MAE")
    lines.append("                      much below ~1e-3 either, regardless of n_f, depth,")
    lines.append("                      or training time.")
    txt = "\n".join(lines)
    print()
    print(txt)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(txt + "\n")

    # plots
    a_r = np.abs(r_omega[final_mask])
    a_c = np.abs(continuity[final_mask])
    xm = x[final_mask]; ym = y[final_mask]

    plt.figure(figsize=(7, 4))
    plt.hist(np.log10(a_r + 1e-20), bins=60)
    plt.xlabel("log10 |vort transport residual|")
    plt.ylabel("count")
    plt.title("CFD vorticity-transport residual distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "residual_hist.png"), dpi=140)
    plt.close()

    for arr, fname, label in [
        (a_r, "residual_map.png",   "|vort transport residual|"),
        (a_c, "continuity_map.png", "|continuity residual|"),
    ]:
        # log-scale colouring with safe floor
        c = np.log10(np.clip(arr, 1e-20, None))
        plt.figure(figsize=(8, 4.5))
        sc = plt.scatter(xm, ym, c=c, s=3, cmap="magma")
        plt.colorbar(sc, label=f"log10 {label}")
        plt.gca().add_patch(plt.Circle((0.0, 0.0), args.cyl_R, color="cyan", fill=False))
        plt.gca().set_aspect("equal")
        plt.xlabel("x"); plt.ylabel("y")
        plt.title(f"{label} on CFD points (interior, k={args.k})")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, fname), dpi=140)
        plt.close()

    print()
    print(f"[done] outputs in {args.out}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vtk", type=str, required=True)
    p.add_argument("--re", type=float, default=40.0)
    p.add_argument("--k", type=int, default=24,
                   help="k-nearest neighbours used in local quadratic fit")
    p.add_argument("--xlim", type=float, nargs=2, default=[-3.0, 12.0])
    p.add_argument("--ylim", type=float, nargs=2, default=[-4.0, 4.0])
    p.add_argument("--cyl-R", type=float, default=0.5)
    p.add_argument("--boundary-buffer", type=float, default=0.15,
                   help="exclude points within this distance of any boundary or cylinder")
    p.add_argument("--out", type=str, default="diag_cfd_pde")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
