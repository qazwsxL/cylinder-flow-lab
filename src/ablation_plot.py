"""
ablation_plot.py — reference-style ablation figure for the Re=40 study.

Builds a single multi-panel figure comparing the FULL recipe against each
"ingredient removed" variant, in the spirit of the SOAP+pseudo-time reference
slide, but adapted to our STEADY Re=40 case (single snapshot: no period /
Cd-Cl time series, so the time-axis panels are replaced by field rel-L2).

Panels:
  (A) Convergence — total loss vs optimizer step, one curve per variant.
  (B) Field reconstruction error — mean field rel-L2 (u,v combined) per variant
      [this is the core "ingredient contribution" bar chart].
  (C) rel-L2 split by component (u vs v), grouped bars.
  (D) Qualitative row — vorticity field per variant.
  (E) Metrics table.

It REUSES the (already-verified) evaluation helpers from compare_runs.py so the
autograd / grid logic is identical to the working comparison script.

Run from src/:
    python ablation_plot.py --vtk-path ../Re40.vtk \
        --runs-root ../runs/ablation \
        --out-dir   ../runs/ablation/_summary \
        --width 96 --depth 5
"""

import argparse
import os
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfp40 import load_single_vtk                       # noqa: E402
from compare_runs import (                              # noqa: E402  (proven helpers)
    eval_on_grid,
    cfd_on_grid,
    load_model,
    mask_cylinder,
)

# ----------------------------------------------------------------------------
# Variant registry — must match the keys/save-dirs in scripts/run_ablation_re40.sh
# Each entry carries its OWN width/depth (nets differ across runs) and an
# optional explicit checkpoint path (relative to --repo-root). When "ckpt" is
# None, the checkpoint is looked up at <runs-root>/<key>/checkpoints/... .
# Missing checkpoints are skipped gracefully.
#
# The first two are REFERENCE runs (pre-existing v2 checkpoints); the rest are
# the budget-matched ablation variants produced by run_ablation_re40.sh.
# ----------------------------------------------------------------------------
VARIANTS = [
    # 2x2 factorial: pseudo-time (PT) x CFD data anchor. All 96/5, same recipe.
    # key,              label,                     color,     width, depth, ckpt
    dict(key="baseline",      label="baseline (no PT, no anchor)", color="#7f7f7f",
         width=96, depth=5, ckpt=None),
    dict(key="pt",            label="PT (no anchor)", color="#1f77b4",
         width=96, depth=5, ckpt=None),
    dict(key="cfd_anchor",    label="CFD anchor (no PT)", color="#9467bd",
         width=96, depth=5, ckpt=None),
    dict(key="pt_cfd_anchor", label="PT + CFD anchor", color="#2ca02c",
         width=96, depth=5, ckpt=None),
]

CKPT_REL = "checkpoints/pinn_Re40_single.pt"
LOG_REL = "train.log"


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def field_relL2(model, snapshot, xs, ys, t_val, device):
    """Relative L2 of (u, v) against CFD over the valid fluid region."""
    X, Y, u, v, p, omega = eval_on_grid(model, xs, ys, t_val, device)
    _, _, u_cfd, v_cfd = cfd_on_grid(snapshot, xs, ys)

    outside = (X**2 + Y**2) > 0.5**2
    valid = outside & ~np.isnan(u_cfd) & ~np.isnan(v_cfd) & ~np.isnan(u) & ~np.isnan(v)

    du = (u[valid] - u_cfd[valid])
    dv = (v[valid] - v_cfd[valid])
    uc = u_cfd[valid]
    vc = v_cfd[valid]

    eps = 1e-12
    relL2_u = float(np.sqrt(np.sum(du**2) / (np.sum(uc**2) + eps)))
    relL2_v = float(np.sqrt(np.sum(dv**2) / (np.sum(vc**2) + eps)))
    relL2_uv = float(np.sqrt((np.sum(du**2) + np.sum(dv**2)) /
                             (np.sum(uc**2) + np.sum(vc**2) + eps)))
    return dict(relL2_u=relL2_u, relL2_v=relL2_v, relL2_uv=relL2_uv,
                X=X, Y=Y, omega=omega)


# ----------------------------------------------------------------------------
# Log parsing → convergence curve
# ----------------------------------------------------------------------------
_ADAM_RE = re.compile(r"\[Adam\]\s*ep=\s*(\d+)\s+total=([-\d.eE+]+)")
_BFGS_RE = re.compile(r"\[BFGS\]\s*batch=\s*\d+\s+call=\s*(\d+)\s+total=([-\d.eE+]+)")


def parse_convergence(log_path):
    """Return (steps, losses) with Adam epochs first, then BFGS calls offset."""
    if not os.path.exists(log_path):
        return None, None
    steps, losses = [], []
    last_adam = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _ADAM_RE.search(line)
            if m:
                ep = int(m.group(1))
                last_adam = max(last_adam, ep)
                steps.append(ep)
                losses.append(float(m.group(2)))
                continue
            m = _BFGS_RE.search(line)
            if m:
                call = int(m.group(1))
                steps.append(last_adam + call)
                losses.append(float(m.group(2)))
    if not steps:
        return None, None
    return np.asarray(steps, dtype=float), np.asarray(losses, dtype=float)


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def build_figure(results, out_path):
    present = [r for r in results if r["metrics"] is not None]
    n = len(present)
    fig = plt.figure(figsize=(4.6 * max(n, 3), 11))
    gs = gridspec.GridSpec(3, max(n, 3), figure=fig, hspace=0.42, wspace=0.32,
                           height_ratios=[1.0, 1.15, 1.15])
    fig.suptitle("Ablation (Re=40, steady): full recipe vs removing each ingredient",
                 fontsize=14, y=0.99)

    # ---- Row 0: vorticity per variant (qualitative) ----
    for i, r in enumerate(present):
        ax = fig.add_subplot(gs[0, i])
        m = r["metrics"]
        X, Y, om = m["X"], m["Y"], m["omega"]
        om_m = mask_cylinder(om, X, Y)
        vmax = np.nanpercentile(np.abs(om_m), 98)
        im = ax.pcolormesh(X, Y, om_m, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="auto")
        ax.add_patch(plt.Circle((0, 0), 0.5, color="white", zorder=5))
        ax.set_aspect("equal")
        ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max())
        ax.set_title(f"vorticity — {r['label']}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # ---- Row 1, col 0: convergence ----
    ax_conv = fig.add_subplot(gs[1, 0])
    for r in present:
        s, l = r["steps"], r["losses"]
        if s is not None:
            ax_conv.semilogy(s, l, label=r["label"], color=r["color"], lw=1.3)
    ax_conv.set_xlabel("optimizer step (Adam epochs, then BFGS calls)")
    ax_conv.set_ylabel("total loss")
    ax_conv.set_title("Convergence")
    ax_conv.grid(True, which="both", alpha=0.3)
    ax_conv.legend(fontsize=7, loc="upper right")

    # ---- Row 1, col 1: main bar chart (combined rel-L2) ----
    ax_bar = fig.add_subplot(gs[1, 1]) if max(n, 3) > 1 else fig.add_subplot(gs[2, 0])
    labels = [r["label"] for r in present]
    vals = [r["metrics"]["relL2_uv"] for r in present]
    colors = [r["color"] for r in present]
    bars = ax_bar.bar(range(len(present)), vals, color=colors)
    ax_bar.set_xticks(range(len(present)))
    ax_bar.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax_bar.set_ylabel("mean field rel-L2 (u,v)")
    ax_bar.set_title("Field reconstruction error")
    for b, v in zip(bars, vals):
        ax_bar.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)

    # ---- Row 1, col 2: grouped bars by component ----
    col2 = 2 if max(n, 3) > 2 else 0
    ax_cmp = fig.add_subplot(gs[1, col2]) if max(n, 3) > 2 else fig.add_subplot(gs[2, 1])
    idx = np.arange(len(present))
    w = 0.38
    ax_cmp.bar(idx - w / 2, [r["metrics"]["relL2_u"] for r in present], w,
               label="u", color="#4c72b0")
    ax_cmp.bar(idx + w / 2, [r["metrics"]["relL2_v"] for r in present], w,
               label="v", color="#dd8452")
    ax_cmp.set_xticks(idx)
    ax_cmp.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax_cmp.set_ylabel("rel-L2 by component")
    ax_cmp.set_title("rel-L2: u vs v")
    ax_cmp.legend(fontsize=8)

    # ---- Row 2: metrics table (spanning) ----
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")
    header = ["variant", "rel-L2 (u,v)", "rel-L2 u", "rel-L2 v"]
    rows = [[r["label"],
             f"{r['metrics']['relL2_uv']:.4f}",
             f"{r['metrics']['relL2_u']:.4f}",
             f"{r['metrics']['relL2_v']:.4f}"] for r in present]
    tbl = ax_tbl.table(cellText=rows, colLabels=header, loc="center",
                       cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    # highlight the best (lowest combined rel-L2)
    best = int(np.argmin([r["metrics"]["relL2_uv"] for r in present]))
    for c in range(len(header)):
        tbl[best + 1, c].set_facecolor("#d4f1c4")   # +1: row 0 is the header

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ablation] saved figure -> {out_path}")


# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vtk-path", required=True)
    p.add_argument("--runs-root", default="../runs/ablation")
    p.add_argument("--repo-root", default="..",
                   help="Base dir for variants with an explicit ckpt path "
                        "(reference runs). Default '..' since you run from src/.")
    p.add_argument("--out-dir", default="../runs/ablation/_summary")
    p.add_argument("--width", type=int, default=96,
                   help="Fallback net width for variants that don't set one")
    p.add_argument("--depth", type=int, default=5,
                   help="Fallback net depth for variants that don't set one")
    p.add_argument("--t-value", type=float, default=0.0)
    p.add_argument("--nx", type=int, default=300)
    p.add_argument("--ny", type=int, default=200)
    p.add_argument("--x-min", type=float, default=-3.0)
    p.add_argument("--x-max", type=float, default=12.0)
    p.add_argument("--y-min", type=float, default=-4.0)
    p.add_argument("--y-max", type=float, default=4.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablation] device={device}")

    snapshot = load_single_vtk(args.vtk_path, t_value=args.t_value)
    xs = np.linspace(args.x_min, args.x_max, args.nx)
    ys = np.linspace(args.y_min, args.y_max, args.ny)

    results = []
    for v in VARIANTS:
        key, label, color = v["key"], v["label"], v["color"]
        width = v.get("width") or args.width
        depth = v.get("depth") or args.depth
        if v.get("ckpt"):                       # explicit reference checkpoint
            ckpt = os.path.join(args.repo_root, v["ckpt"])
            log = os.path.join(os.path.dirname(os.path.dirname(ckpt)), LOG_REL)
        else:                                    # ablation variant under runs-root
            ckpt = os.path.join(args.runs_root, key, CKPT_REL)
            log = os.path.join(args.runs_root, key, LOG_REL)
        r = dict(key=key, label=label, color=color,
                 metrics=None, steps=None, losses=None)
        if not os.path.exists(ckpt):
            print(f"[ablation] SKIP {key}: checkpoint not found ({ckpt})")
            results.append(r)
            continue
        print(f"[ablation] evaluating {key} (net {width}/{depth}) ...")
        model = load_model(ckpt, width, depth, device)
        r["metrics"] = field_relL2(model, snapshot, xs, ys, args.t_value, device)
        r["steps"], r["losses"] = parse_convergence(log)
        results.append(r)

    present = [r for r in results if r["metrics"] is not None]
    if not present:
        raise SystemExit("[ablation] No checkpoints found under "
                         f"{args.runs_root}. Run scripts/run_ablation_re40.sh first.")

    # console + csv summary
    print("\n" + "=" * 64)
    print(f"{'variant':<26}{'relL2_uv':>10}{'relL2_u':>10}{'relL2_v':>10}")
    print("-" * 64)
    csv_path = os.path.join(args.out_dir, "ablation_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("variant,label,relL2_uv,relL2_u,relL2_v\n")
        for r in present:
            m = r["metrics"]
            print(f"{r['label']:<26}{m['relL2_uv']:>10.4f}"
                  f"{m['relL2_u']:>10.4f}{m['relL2_v']:>10.4f}")
            f.write(f"{r['key']},{r['label']},{m['relL2_uv']:.6f},"
                    f"{m['relL2_u']:.6f},{m['relL2_v']:.6f}\n")
    print("=" * 64)

    fig_path = os.path.join(args.out_dir, "ablation_comparison.png")
    build_figure(results, fig_path)
    print(f"[ablation] metrics -> {csv_path}")


if __name__ == "__main__":
    main()
