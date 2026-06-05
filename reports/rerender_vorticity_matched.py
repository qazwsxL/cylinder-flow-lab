"""
rerender_vorticity_matched.py
=============================

Re-render the PINN v2 vorticity for the three runs that appear in slide 31
of the v2 progress deck (P1 baseline / P2 strict consistency / P2 all-CFD
anchor prio=2.0) using a SHARED colour scale  vmin = -8, vmax = +8  so the
three panels are visually comparable with the CFD truth.

Run this on a node where torch + the cfp40 / cfp40_v2 environment work
(typically the Oscar GPU node where the training itself was done).

    cd <project_root>     # the folder that contains cfp40_v2.py + Re40.vtk
    python reports/rerender_vorticity_matched.py

Outputs go to  reports/figs_v2/pinn_vorticity_matched/<label>.png .

The script auto-detects the project root by walking one directory up from
its own location, so it works equally on the local Mac path
(/Users/peterchen/Desktop/cylinder-flow-lab) and on the Oscar path
(/oscar/home/jchen790/cylinder flow lab).
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------------------
# Resolve the project root: the directory containing cfp40_v2.py and
# Re40.vtk.  This script lives in <root>/reports/ , so go one up.
# ---------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

print(f"[rerender] project root = {ROOT}")
for f in ("cfp40_v2.py", "cfp40.py", "Re40.vtk"):
    found = os.path.exists(os.path.join(ROOT, f))
    print(f"[rerender]   {f}: {'OK' if found else 'MISSING'}")

import cfp40_v2 as v2                                                   # noqa
from cfp40 import evaluate_on_grid                                      # noqa

OUT = os.path.join(ROOT, "reports", "figs_v2", "pinn_vorticity_matched")
os.makedirs(OUT, exist_ok=True)

# (label, ckpt_dir, model_kwargs)
RUNS = [
    ("P1_baseline",
     "runs/v2_alldata_p2_sweep_quick/P1/checkpoints",
     dict(width=32, depth=3)),

    ("P2_strict_consist",
     "runs/v2_consistency_quick/P2/checkpoints",
     dict(width=32, depth=3)),

    ("P2_allCFD_prio2",
     "runs/v2_alldata_p2_sweep_quick/P2_prio2/checkpoints",
     dict(width=32, depth=3)),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[rerender] device = {device}")

xlim = (-8.0, 12.0)
ylim = (-8.0,  8.0)
VMIN, VMAX = -8.0, 8.0


def render_one(label, save_dir, model_kwargs):
    print(f"\n[rerender] === {label} ===  ckpt_dir = {save_dir}")
    model = v2.load_model_for_viz_v2(save_dir, device, **model_kwargs)
    X, Y, U, V, W = evaluate_on_grid(model, device=device, t_val=0.0,
                                     nx=400, ny=200,
                                     xlim=xlim, ylim=ylim)
    print(f"             vorticity range = [{np.nanmin(W):+.3f}, {np.nanmax(W):+.3f}]")
    print(f"             saturated at ±8 for matched colorbar.")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    im = ax.pcolormesh(X, Y, W, shading="auto", cmap="coolwarm",
                       vmin=VMIN, vmax=VMAX)
    # draw cylinder
    theta = np.linspace(0, 2 * np.pi, 80)
    ax.fill(0.5 * np.cos(theta), 0.5 * np.sin(theta),
            color="0.2", edgecolor="black")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"PINN v2 — vorticity   {label}   (vmin / vmax = ±8)")
    plt.colorbar(im, ax=ax, shrink=0.85, label="ω")
    out_path = os.path.join(OUT, f"{label}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"             wrote {out_path}")


for label, rel_dir, mk in RUNS:
    save_dir = os.path.join(ROOT, rel_dir)
    if not os.path.exists(os.path.join(save_dir, "pinn_Re40_single.pt")) \
       and not os.path.exists(os.path.join(save_dir, "pinn_latest.pt")):
        print(f"\n[rerender] !! checkpoint missing for {label}: {save_dir}")
        continue
    render_one(label, save_dir, mk)

print(f"\n[rerender] done.  outputs in {OUT}")
