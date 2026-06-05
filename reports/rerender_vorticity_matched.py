"""
rerender_vorticity_matched.py
=============================
Re-render PINN v2 vorticity for the three runs we compare in the v2 progress
slide using a *matching* colorbar (vmin=-8, vmax=+8) so the panels are
visually comparable with the CFD truth.

Run on the GPU node (after the runs have produced their checkpoints):

    python reports/rerender_vorticity_matched.py

Outputs go to reports/figs_v2/pinn_vorticity_matched/{name}.png .
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = "/Users/peterchen/Desktop/cylinder-flow-lab"
sys.path.insert(0, ROOT)
import cfp40_v2 as v2                                                  # noqa

OUT = os.path.join(ROOT, "reports", "figs_v2", "pinn_vorticity_matched")
os.makedirs(OUT, exist_ok=True)

RUNS = [
    # (label, ckpt_dir, model_kwargs)
    ("P1_baseline",        "runs/v2_alldata_p2_sweep_quick/P1",       dict(width=32, depth=3)),
    ("P2_strict_consist",  "runs/v2_consistency_quick/P2",            dict(width=32, depth=3)),
    ("P2_allCFD_prio2",    "runs/v2_alldata_p2_sweep_quick/P2_prio2", dict(width=32, depth=3)),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xlim = (-8.0, 12.0); ylim = (-8.0, 8.0)
VMIN, VMAX = -8.0, 8.0

for label, ckpt_dir, mk in RUNS:
    save_dir = os.path.join(ROOT, ckpt_dir, "checkpoints")
    model = v2.load_model_for_viz_v2(save_dir, device, **mk)
    X, Y, U, V, W = v2.evaluate_on_grid(model, device=device, t_val=0.0,
                                        xlim=xlim, ylim=ylim)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    im = ax.pcolormesh(X, Y, W, shading="auto", cmap="coolwarm",
                       vmin=VMIN, vmax=VMAX)
    theta = np.linspace(0, 2*np.pi, 80)
    ax.fill(0.5*np.cos(theta), 0.5*np.sin(theta), color="0.2", ec="black")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_title(f"PINN v2 — vorticity   {label}   (vmin/vmax = ±8)")
    plt.colorbar(im, ax=ax, shrink=0.85, label="ω")
    out = os.path.join(OUT, f"{label}.png")
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
