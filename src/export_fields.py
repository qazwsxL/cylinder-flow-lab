"""
export_fields.py — evaluate the 4 ablation checkpoints on a grid and dump the
raw field arrays (+ CFD reference + rel-L2 metrics) to a single .npz, so the
comparison figure can be built anywhere without torch.

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
from cfp40 import load_single_vtk                       # noqa: E402
from compare_runs import eval_on_grid, cfd_on_grid, load_model   # noqa: E402 (proven)

# key, width, depth  (all four cells are 96/5)
CELLS = [
    ("baseline",      96, 5),
    ("pt",            96, 5),
    ("cfd_anchor",    96, 5),
    ("pt_cfd_anchor", 96, 5),
]


def rel_l2(u, v, uc, vc, X, Y):
    outside = (X**2 + Y**2) > 0.5**2
    valid = outside & ~np.isnan(uc) & ~np.isnan(vc) & ~np.isnan(u) & ~np.isnan(v)
    du = u[valid] - uc[valid]
    dv = v[valid] - vc[valid]
    ucv = uc[valid]
    vcv = vc[valid]
    eps = 1e-12
    ru = float(np.sqrt(np.sum(du**2) / (np.sum(ucv**2) + eps)))
    rv = float(np.sqrt(np.sum(dv**2) / (np.sum(vcv**2) + eps)))
    ruv = float(np.sqrt((np.sum(du**2) + np.sum(dv**2)) /
                        (np.sum(ucv**2) + np.sum(vcv**2) + eps)))
    return ru, rv, ruv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk-path", required=True)
    ap.add_argument("--runs-root", default="../runs/ablation")
    ap.add_argument("--out", default="../runs/ablation/_export/fields.npz")
    ap.add_argument("--nx", type=int, default=320)
    ap.add_argument("--ny", type=int, default=200)
    ap.add_argument("--x-min", type=float, default=-3.0)
    ap.add_argument("--x-max", type=float, default=12.0)
    ap.add_argument("--y-min", type=float, default=-4.0)
    ap.add_argument("--y-max", type=float, default=4.0)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[export] device={device}")

    snap = load_single_vtk(a.vtk_path, t_value=0.0)
    xs = np.linspace(a.x_min, a.x_max, a.nx)
    ys = np.linspace(a.y_min, a.y_max, a.ny)

    store = {}
    metrics = {}
    present = []
    X = Y = u_cfd = v_cfd = None
    for key, w, d in CELLS:
        ck = os.path.join(a.runs_root, key, "checkpoints", "pinn_Re40_single.pt")
        if not os.path.exists(ck):
            print(f"[export] SKIP {key}: no checkpoint at {ck}")
            continue
        model = load_model(ck, w, d, device)
        X, Y, u, v, p, om = eval_on_grid(model, xs, ys, 0.0, device)
        _, _, u_cfd, v_cfd = cfd_on_grid(snap, xs, ys)
        store[f"{key}_u"] = u.astype(np.float32)
        store[f"{key}_v"] = v.astype(np.float32)
        store[f"{key}_p"] = p.astype(np.float32)
        store[f"{key}_omega"] = om.astype(np.float32)
        ru, rv, ruv = rel_l2(u, v, u_cfd, v_cfd, X, Y)
        metrics[key] = dict(relL2_u=ru, relL2_v=rv, relL2_uv=ruv)
        present.append(key)
        print(f"[export] {key:14s}  relL2_uv={ruv:.4f}  u={ru:.4f}  v={rv:.4f}")

    if not present:
        raise SystemExit("[export] no checkpoints found under " + a.runs_root)

    store["X"] = X.astype(np.float32)
    store["Y"] = Y.astype(np.float32)
    store["u_cfd"] = u_cfd.astype(np.float32)
    store["v_cfd"] = v_cfd.astype(np.float32)
    store["_cells"] = np.array(present)

    np.savez_compressed(a.out, **store)
    with open(os.path.join(os.path.dirname(a.out), "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[export] saved -> {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")
    print(f"[export] metrics -> {os.path.join(os.path.dirname(a.out), 'metrics.json')}")


if __name__ == "__main__":
    main()
