"""
export_fields.py — full-field health check for the 4 ablation checkpoints.

For each cell it evaluates the model on a grid and stores, vs CFD ground truth:
  u, v, p, vorticity, and the PDE residual map (|f_u|+|f_v|+|div|),
plus rel-L2 for u, v and (gauge-free) pressure, and the mean PDE residual.
Everything is dumped to one .npz so the figure can be built without torch.

Pressure is defined up to a constant (PINN anchors p at the outlet; CFD pMean
has its own gauge), so the pressure error is computed on mean-subtracted fields.

Run on Oscar (needs torch + pyvista), from src/:
    python export_fields.py --vtk-path ../Re40.vtk \
        --runs-root ../runs/ablation \
        --out ../runs/ablation/_export/fields.npz
Then pull ../runs/ablation/_export/ locally.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfp40 import load_single_vtk                                    # noqa: E402
from compare_runs import (                                          # noqa: E402
    eval_on_grid, cfd_on_grid, load_model, pde_residual_on_grid,
)

def discover_cells(runs_root, width, depth):
    """Every subdir of runs_root that has checkpoints/pinn_Re40_single.pt."""
    out = []
    if not os.path.isdir(runs_root):
        return out
    for key in sorted(os.listdir(runs_root)):
        ck = os.path.join(runs_root, key, "checkpoints", "pinn_Re40_single.pt")
        if os.path.exists(ck):
            out.append((key, width, depth))
    return out


def cfd_pressure_on_grid(vtk_path, xs, ys):
    """Interpolate CFD pMean (fallback p) onto the eval grid."""
    import pyvista as pv
    from scipy.interpolate import griddata
    mesh = pv.read(vtk_path)
    if mesh.n_cells > 0:
        centers = mesh.cell_centers().points
        src = mesh.cell_data
    else:
        centers = mesh.points
        src = mesh.point_data
    names = list(src.keys())
    pname = None
    for cand in ["pMean", "p", "pressure", "P"]:
        for n in names:
            if n.lower() == cand.lower():
                pname = n
                break
        if pname:
            break
    if pname is None:
        raise KeyError(f"no pressure array in {vtk_path}; have {names}")
    P = np.asarray(src[pname]).astype(float).ravel()
    X, Y = np.meshgrid(xs, ys)
    p_cfd = griddata(np.stack([centers[:, 0], centers[:, 1]], axis=1),
                     P, (X, Y), method="linear")
    return p_cfd, pname


def rel_l2(a, b, ref, valid):
    """rel-L2 of (a-b) vs ref magnitude over valid mask."""
    da = (a[valid] - b[valid])
    eps = 1e-12
    return float(np.sqrt(np.sum(da**2) / (np.sum(ref[valid]**2) + eps)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk-path", required=True)
    ap.add_argument("--runs-root", default="../runs/ablation")
    ap.add_argument("--out", default="../runs/ablation/_export/fields.npz")
    ap.add_argument("--Re", type=float, default=40.0)
    ap.add_argument("--nx", type=int, default=320)
    ap.add_argument("--ny", type=int, default=200)
    ap.add_argument("--x-min", type=float, default=-3.0)
    ap.add_argument("--x-max", type=float, default=12.0)
    ap.add_argument("--y-min", type=float, default=-4.0)
    ap.add_argument("--y-max", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--depth", type=int, default=5)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export] device={device}")

    snap = load_single_vtk(a.vtk_path, t_value=0.0)
    xs = np.linspace(a.x_min, a.x_max, a.nx)
    ys = np.linspace(a.y_min, a.y_max, a.ny)
    _, _, u_cfd, v_cfd = cfd_on_grid(snap, xs, ys)
    p_cfd, pname = cfd_pressure_on_grid(a.vtk_path, xs, ys)
    print(f"[export] CFD pressure array = {pname}")

    cells = discover_cells(a.runs_root, a.width, a.depth)
    print(f"[export] discovered {len(cells)} cells: {[c[0] for c in cells]}")

    store = {}
    metrics = {}
    present = []
    X = Y = None
    for key, w, d in cells:
        ck = os.path.join(a.runs_root, key, "checkpoints", "pinn_Re40_single.pt")
        if not os.path.exists(ck):
            print(f"[export] SKIP {key}: no checkpoint at {ck}")
            continue
        model = load_model(ck, w, d, device)
        X, Y, u, v, p, om = eval_on_grid(model, xs, ys, 0.0, device)
        fu, fv, div = pde_residual_on_grid(model, xs, ys, 0.0, a.Re, device)
        res = np.abs(fu) + np.abs(fv) + np.abs(div)          # residual magnitude map

        outside = (X**2 + Y**2) > 0.5**2
        v_uv = outside & ~np.isnan(u_cfd) & ~np.isnan(v_cfd) & ~np.isnan(u) & ~np.isnan(v)
        v_p = outside & ~np.isnan(p_cfd) & ~np.isnan(p)

        # gauge-free pressure: subtract the mean over the valid region
        p_ms = p.copy(); pc_ms = p_cfd.copy()
        p_ms[v_p] = p[v_p] - p[v_p].mean()
        pc_ms[v_p] = p_cfd[v_p] - p_cfd[v_p].mean()

        relL2_u = rel_l2(u, u_cfd, u_cfd, v_uv)
        relL2_v = rel_l2(v, v_cfd, v_cfd, v_uv)
        relL2_uv = float(np.sqrt(
            (np.sum((u[v_uv]-u_cfd[v_uv])**2) + np.sum((v[v_uv]-v_cfd[v_uv])**2)) /
            (np.sum(u_cfd[v_uv]**2) + np.sum(v_cfd[v_uv]**2) + 1e-12)))
        relL2_p = rel_l2(p_ms, pc_ms, pc_ms, v_p)
        mean_res = float(np.nanmean(res[outside]))

        store[f"{key}_u"] = u.astype(np.float32)
        store[f"{key}_v"] = v.astype(np.float32)
        store[f"{key}_p"] = p_ms.astype(np.float32)          # mean-subtracted
        store[f"{key}_omega"] = om.astype(np.float32)
        store[f"{key}_res"] = res.astype(np.float32)
        metrics[key] = dict(relL2_uv=relL2_uv, relL2_u=relL2_u,
                            relL2_v=relL2_v, relL2_p=relL2_p, mean_pde_res=mean_res)
        present.append(key)
        print(f"[export] {key:14s} relL2 uv={relL2_uv:.4f} u={relL2_u:.4f} "
              f"v={relL2_v:.4f} p={relL2_p:.4f}  mean|res|={mean_res:.2e}")

    if not present:
        raise SystemExit("[export] no checkpoints under " + a.runs_root)

    store["X"] = X.astype(np.float32)
    store["Y"] = Y.astype(np.float32)
    store["u_cfd"] = u_cfd.astype(np.float32)
    store["v_cfd"] = v_cfd.astype(np.float32)
    pc_ms_full = p_cfd.copy()
    m = (X**2 + Y**2) > 0.5**2
    m &= ~np.isnan(p_cfd)
    pc_ms_full[m] = p_cfd[m] - p_cfd[m].mean()
    store["p_cfd"] = pc_ms_full.astype(np.float32)           # mean-subtracted
    store["_cells"] = np.array(present)

    np.savez_compressed(a.out, **store)
    with open(os.path.join(os.path.dirname(a.out), "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[export] saved -> {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
