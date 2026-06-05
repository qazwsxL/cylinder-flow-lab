"""
parse_and_plot_v2.py
====================
Parse training logs + ckpt-monitor numbers for the three small-net (32/3)
v2 runs we are comparing:

  (A) P1 baseline       — data-only, all CFD pts
  (B) P2 strict consist — PDE-only, NO data anchor
  (C) P2 all-CFD prio=2 — DATA + PDE, all CFD pts, data_priority = 2.0

Outputs to reports/figs_v2/:
  fig01_loss_curves.png      — DATA / PDE loss vs iter, 3 runs
  fig02_metric_bars.png      — mae_u, mae_v, mean|duv|/|u_true|, rel_l2_speed
  fig03_vorticity_panel.png  — CFD truth + 3 PINN outputs side-by-side
"""

import os, re, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyvista as pv

ROOT  = "/sessions/dazzling-great-fermi/mnt/cylinder-flow-lab"
LOG   = os.path.join(ROOT, "logs")
RUNS  = os.path.join(ROOT, "runs")
OUT   = os.path.join(ROOT, "reports", "figs_v2")
os.makedirs(OUT, exist_ok=True)

# -----------------------------------------------------------------
# 1. Define the three runs and their log slices
# -----------------------------------------------------------------
# Phase 1 baseline is shared across all the *_quick logs — they all start with the
# same data-only P1 (same seed + same setup). Use the one from alldata_p2_quick,
# whose Phase-2 prio=2 we also plot (same code path, fewest inconsistencies).

RUNS_SPEC = [
    {
        "name":  "P1 baseline (data-only)",
        "color": "tab:blue",
        "log":   os.path.join(LOG, "pinn_re40_alldata_p2_quick_2766394.out"),
        "phase_tag": "[ALLDATA-P2] Phase 1",   # log marker that opens this phase
        "next_tag":  "[timing] Phase 1 took",  # log marker that closes this phase
        "viz_png": os.path.join(RUNS, "v2_alldata_p2_sweep_quick/P1/viz/v2_vorticity.png"),
    },
    {
        "name":  "P2 strict consistency (no data)",
        "color": "tab:red",
        "log":   os.path.join(LOG, "pinn_re40_consist_quick_2750989.out"),
        "phase_tag": "[QUICK] Phase 2",
        "next_tag":  "[timing] Phase 2 took",
        "viz_png": os.path.join(RUNS, "v2_consistency_quick/P2/viz/v2_vorticity.png"),
    },
    {
        "name":  "P2 all-CFD anchor, prio=2.0",
        "color": "tab:green",
        "log":   os.path.join(LOG, "pinn_re40_alldata_p2_quick_2766394.out"),
        "phase_tag": "[ALLDATA-P2] Phase 2 — priority=2 ",
        "next_tag":  "[ALLDATA-P2] Phase 2 — priority=5",
        "viz_png": os.path.join(RUNS, "v2_alldata_p2_sweep_quick/P2_prio2/viz/v2_vorticity.png"),
    },
]

# -----------------------------------------------------------------
# 2. Log parsers
# -----------------------------------------------------------------
ADAM_RE = re.compile(
    r"\[Adam-v2\] ep=\s*(\d+)\s+total=([0-9.eE+\-]+)\s+"
    r"PDE=([0-9.eE+\-]+).*?DATA=([0-9.eE+\-]+)"
)
BFGS_RE = re.compile(
    r"\[BFGS\] batch=\s*\d+\s+call=\s*(\d+)\s+total=([0-9.eE+\-]+)\s+"
    r"PDE=([0-9.eE+\-]+).*?DATA=([0-9.eE+\-]+)"
)
MON_RE = re.compile(
    r"\[(post-BFGS|post-Adam|phase-start)\] CFD monitor \| "
    r"mae_u=([0-9.eE+\-]+) mae_v=([0-9.eE+\-]+) "
    r"rmse_u=([0-9.eE+\-]+) rmse_v=([0-9.eE+\-]+) "
    r"relL2_u=([0-9.eE+\-]+) relL2_v=([0-9.eE+\-]+) "
    r"mean\|duv\|=([0-9.eE+\-]+) n=(\d+)"
)


def slice_phase(text, start_tag, end_tag):
    """Return the substring between start_tag (inclusive) and end_tag (exclusive)."""
    i = text.find(start_tag)
    if i < 0:
        raise RuntimeError(f"start_tag not found: {start_tag!r}")
    j = text.find(end_tag, i + 1)
    if j < 0:
        j = len(text)  # tail
    return text[i:j]


def parse_phase(text):
    """Extract Adam iters and BFGS iters from one phase chunk."""
    adam_iters, adam_total, adam_pde, adam_data = [], [], [], []
    for m in ADAM_RE.finditer(text):
        adam_iters.append(int(m.group(1)))
        adam_total.append(float(m.group(2)))
        adam_pde.append(float(m.group(3)))
        adam_data.append(float(m.group(4)))

    bfgs_call, bfgs_total, bfgs_pde, bfgs_data = [], [], [], []
    for m in BFGS_RE.finditer(text):
        bfgs_call.append(int(m.group(1)))
        bfgs_total.append(float(m.group(2)))
        bfgs_pde.append(float(m.group(3)))
        bfgs_data.append(float(m.group(4)))

    monitors = []
    for m in MON_RE.finditer(text):
        monitors.append({
            "tag":      m.group(1),
            "mae_u":    float(m.group(2)),
            "mae_v":    float(m.group(3)),
            "rmse_u":   float(m.group(4)),
            "rmse_v":   float(m.group(5)),
            "rel_l2_u": float(m.group(6)),
            "rel_l2_v": float(m.group(7)),
            "mean_duv": float(m.group(8)),
            "n":        int(m.group(9)),
        })

    return {
        "adam_ep":   np.asarray(adam_iters),
        "adam_tot":  np.asarray(adam_total),
        "adam_pde":  np.asarray(adam_pde),
        "adam_data": np.asarray(adam_data),
        "bfgs_call": np.asarray(bfgs_call),
        "bfgs_tot":  np.asarray(bfgs_total),
        "bfgs_pde":  np.asarray(bfgs_pde),
        "bfgs_data": np.asarray(bfgs_data),
        "monitors":  monitors,
    }


for r in RUNS_SPEC:
    with open(r["log"]) as f:
        text = f.read()
    chunk = slice_phase(text, r["phase_tag"], r["next_tag"])
    r["data"] = parse_phase(chunk)
    last_mon = r["data"]["monitors"][-1] if r["data"]["monitors"] else None
    r["last_mon"] = last_mon
    print(f"{r['name']:40s}  adam={len(r['data']['adam_ep'])}  "
          f"bfgs={len(r['data']['bfgs_call'])}  last_mon={last_mon and last_mon['tag']}  "
          f"mae_u={last_mon and last_mon['mae_u']:.4e}")


# -----------------------------------------------------------------
# 3. CFD-mesh statistics for the speed-normalization
# -----------------------------------------------------------------
stats = np.load("/sessions/dazzling-great-fermi/mnt/outputs/cfd_speed_stats.npz")
mean_speed = float(stats["mean_speed"])
rms_speed  = float(stats["rms_speed"])
L2_u_true  = float(stats["L2_u"])
L2_v_true  = float(stats["L2_v"])
L2_speed_true = float(stats["L2_speed"])
N_cells    = int(stats["n"])
print(f"\nCFD truth: mean_speed={mean_speed:.4f}  rms_speed={rms_speed:.4f}  "
      f"N={N_cells}  ||u||={L2_u_true:.2f}  ||v||={L2_v_true:.2f}  "
      f"||v||/||u||={L2_v_true/L2_u_true:.3f}")


# -----------------------------------------------------------------
# 4. FIGURE 1 — loss curves
# -----------------------------------------------------------------
# Plot only the loss terms that are actually in each run's objective:
#   P1 baseline      → DATA (PDE term is hard-zero by --data-only)
#   P2 strict consis → PDE  (DATA term is hard-zero by --pde-only)
#   P2 all-CFD prio2 → DATA + PDE  (both active)
# Solid = Adam phase, dashed = BFGS phase.  Iteration axis = Adam ep then
# (max Adam ep) + BFGS call so the two phases sit side-by-side.

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))
ax_d, ax_p = axes

# Track what's actually active in the loss
ACTIVE = {
    "P1 baseline (data-only)":            {"data": True,  "pde": False},
    "P2 strict consistency (no data)":    {"data": False, "pde": True},
    "P2 all-CFD anchor, prio=2.0":        {"data": True,  "pde": True},
}

for r in RUNS_SPEC:
    d = r["data"]
    if d["adam_ep"].size:
        adam_x = d["adam_ep"]
        bfgs_offset = d["adam_ep"].max()
    else:
        adam_x = np.array([])
        bfgs_offset = 0
    bfgs_x = bfgs_offset + d["bfgs_call"]

    act = ACTIVE[r["name"]]
    if act["data"]:
        ax_d.plot(adam_x,  d["adam_data"],  color=r["color"], lw=1.6)
        ax_d.plot(bfgs_x,  d["bfgs_data"],  color=r["color"], lw=1.6, ls="--")
    if act["pde"]:
        ax_p.plot(adam_x,  d["adam_pde"],   color=r["color"], lw=1.6)
        ax_p.plot(bfgs_x,  d["bfgs_pde"],   color=r["color"], lw=1.6, ls="--")

for ax, title in [(ax_d, "DATA loss   (mean (u_pred − u_cfd)²+(v_pred − v_cfd)²)"),
                  (ax_p, "PDE residual loss  (collocation Navier–Stokes momentum)")]:
    ax.set_yscale("log")
    ax.set_xlabel("iteration   (Adam epoch ⟶ BFGS call, offset by max Adam epoch)")
    ax.set_ylabel("loss value")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_title(title, fontsize=10.5)

# Clean legend
handles = [plt.Line2D([0], [0], color=r["color"], lw=2.0, label=r["name"])
           for r in RUNS_SPEC]
handles.append(plt.Line2D([0], [0], color="0.3", lw=2.0, ls="-",  label="Adam stage"))
handles.append(plt.Line2D([0], [0], color="0.3", lw=2.0, ls="--", label="BFGS stage"))
ax_d.legend(handles=handles, fontsize=8.5, loc="upper right", framealpha=0.95)

# Annotate the runs that don't have one of the loss terms
ax_d.text(0.02, 0.04,
          "P2 strict consistency: DATA term is OFF in the loss (not plotted)",
          transform=ax_d.transAxes, fontsize=8, color="tab:red", style="italic")
ax_p.text(0.02, 0.04,
          "P1 baseline: PDE term is OFF in the loss (not plotted)",
          transform=ax_p.transAxes, fontsize=8, color="tab:blue", style="italic")

fig.suptitle("v2 (small net, width=32 / depth=3) — training loss curves for the three runs",
             y=1.02, fontsize=12)
fig.tight_layout()
out = os.path.join(OUT, "fig01_loss_curves.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {out}")

# -----------------------------------------------------------------
# 4b. FIGURE 1b — full-mesh MAE_u over iterations (from CFD monitor)
# -----------------------------------------------------------------
# Build per-run series of (iter, mae_u, mae_v) from the parsed monitors.

def monitor_iter_xs(r):
    d = r["data"]
    adam_max = d["adam_ep"].max() if d["adam_ep"].size else 0
    bfgs_max = d["bfgs_call"].max() if d["bfgs_call"].size else 0
    # CFD monitor is printed at: phase-start, every 100 Adam epochs, post-Adam, post-BFGS
    iters, mae_u, mae_v, mduv = [], [], [], []
    adam_ep_seen = 0
    for mn in d["monitors"]:
        if mn["tag"] == "phase-start":
            iters.append(0)
        elif mn["tag"] == "post-Adam":
            iters.append(adam_max)
        elif mn["tag"] == "post-BFGS":
            iters.append(adam_max + bfgs_max)
        mae_u.append(mn["mae_u"])
        mae_v.append(mn["mae_v"])
        mduv.append(mn["mean_duv"])
    # Adam-internal monitors (Adam-v2 ep=NN] CFD monitor) need a different regex
    return np.array(iters), np.array(mae_u), np.array(mae_v), np.array(mduv)

ADAM_MON_RE = re.compile(
    r"\[Adam-v2 ep=\s*(\d+)\] CFD monitor \| "
    r"mae_u=([0-9.eE+\-]+) mae_v=([0-9.eE+\-]+) "
    r"rmse_u=([0-9.eE+\-]+) rmse_v=([0-9.eE+\-]+) "
    r"relL2_u=([0-9.eE+\-]+) relL2_v=([0-9.eE+\-]+) "
    r"mean\|duv\|=([0-9.eE+\-]+) n=\d+"
)

def full_monitor_series(r):
    """Combine phase-start + Adam-internal CFD monitor + post-Adam + post-BFGS."""
    with open(r["log"]) as f:
        text = f.read()
    chunk = slice_phase(text, r["phase_tag"], r["next_tag"])
    d = r["data"]
    adam_max = d["adam_ep"].max() if d["adam_ep"].size else 0
    bfgs_max = d["bfgs_call"].max() if d["bfgs_call"].size else 0

    iters, mae_u, mae_v, mduv = [], [], [], []

    # phase-start
    for mn in d["monitors"]:
        if mn["tag"] == "phase-start":
            iters.append(0); mae_u.append(mn["mae_u"]); mae_v.append(mn["mae_v"]); mduv.append(mn["mean_duv"])
            break

    # Adam-internal monitors
    for m in ADAM_MON_RE.finditer(chunk):
        iters.append(int(m.group(1)))
        mae_u.append(float(m.group(2)))
        mae_v.append(float(m.group(3)))
        mduv.append(float(m.group(8)))

    # post-Adam and post-BFGS
    for mn in d["monitors"]:
        if mn["tag"] == "post-Adam":
            iters.append(adam_max); mae_u.append(mn["mae_u"]); mae_v.append(mn["mae_v"]); mduv.append(mn["mean_duv"])
        if mn["tag"] == "post-BFGS":
            iters.append(adam_max + bfgs_max); mae_u.append(mn["mae_u"]); mae_v.append(mn["mae_v"]); mduv.append(mn["mean_duv"])

    return np.array(iters), np.array(mae_u), np.array(mae_v), np.array(mduv)


fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
ax_u, ax_v, ax_s = axes

for r in RUNS_SPEC:
    it, mu, mv, mdv = full_monitor_series(r)
    if it.size == 0:
        continue
    ax_u.plot(it, mu,    color=r["color"], lw=1.6, marker="o", ms=4, label=r["name"])
    ax_v.plot(it, mv,    color=r["color"], lw=1.6, marker="o", ms=4)
    ax_s.plot(it, mdv / mean_speed, color=r["color"], lw=1.6, marker="o", ms=4)

for ax, title in [
    (ax_u, "full-mesh  mae_u  (CFD-monitor, N=7456 cells)"),
    (ax_v, "full-mesh  mae_v"),
    (ax_s, "full-mesh  mean|du,dv| / mean speed\n(velocity-vector error normalized by √(u²+v²))"),
]:
    ax.set_yscale("log")
    ax.set_xlabel("iteration   (Adam epoch ⟶ BFGS call)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_title(title, fontsize=10.5)

ax_u.legend(fontsize=8.5, loc="upper right", framealpha=0.95)

fig.suptitle("v2 (small net) — full-mesh velocity error over training (CFD-monitor numbers)",
             y=1.03, fontsize=12)
fig.tight_layout()
out = os.path.join(OUT, "fig01b_mae_curves.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {out}")


# -----------------------------------------------------------------
# 5. FIGURE 2 — final metric bar chart
# -----------------------------------------------------------------
labels = [r["name"] for r in RUNS_SPEC]
colors = [r["color"] for r in RUNS_SPEC]

mae_u  = [r["last_mon"]["mae_u"]    for r in RUNS_SPEC]
mae_v  = [r["last_mon"]["mae_v"]    for r in RUNS_SPEC]
mduv   = [r["last_mon"]["mean_duv"] for r in RUNS_SPEC]
rmse_u = [r["last_mon"]["rmse_u"]   for r in RUNS_SPEC]
rmse_v = [r["last_mon"]["rmse_v"]   for r in RUNS_SPEC]
rel_l2_v_orig = [r["last_mon"]["rel_l2_v"] for r in RUNS_SPEC]

# Mentor's request: don't divide by ||v||; divide by speed magnitude sqrt(u²+v²) instead.
# (1) Per-point mean: mean|duv| / mean_speed.
# (2) L2:  ||(du,dv)||₂ / ||(u,v)||₂ = sqrt(N*(rmse_u² + rmse_v²)) / L2_speed_true
norm_per_point = [m / mean_speed for m in mduv]
norm_L2 = [np.sqrt(N_cells * (ru**2 + rv**2)) / L2_speed_true
           for ru, rv in zip(rmse_u, rmse_v)]

fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
x = np.arange(len(labels))
short_labels = ["P1\nbaseline", "P2 strict\n(no data)", "P2 all-CFD\nprio=2"]

panels = [
    (axs[0], mae_u,           "mae_u  (4869-pt eval)"),
    (axs[1], mae_v,           "mae_v"),
    (axs[2], norm_per_point,  "mean|du,dv| /  mean speed\n(uses √(u²+v²), not |v|)"),
    (axs[3], norm_L2,         "||(du,dv)||₂  /  ||(u,v)||₂\n(velocity-vector rel-L2)"),
]
for ax, vals, title in panels:
    bars = ax.bar(x, vals, color=colors, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.25)

fig.suptitle("Velocity-error comparison across the three runs  "
             "(small net, width=32 / depth=3, N=7456 CFD cells)", y=1.05)
fig.tight_layout()
out = os.path.join(OUT, "fig02_metric_bars.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {out}")


# Also dump a table of the same numbers next to the rel_l2_v that was on the slide
print("\n=== Final-iteration metrics table ===")
print(f"{'run':40s}  {'mae_u':>9s} {'mae_v':>9s} {'mean|duv|':>10s} "
      f"{'mduv/speed':>11s} {'L2_vec/L2_uv':>13s} {'OLD rel_l2_v':>13s}")
for r, mu, mv, mdv, np1, nL2, rlv in zip(
        RUNS_SPEC, mae_u, mae_v, mduv, norm_per_point, norm_L2, rel_l2_v_orig):
    print(f"{r['name']:40s}  {mu:9.4f} {mv:9.4f} {mdv:10.4f} "
          f"{np1:11.4f} {nL2:13.4f} {rlv:13.4f}")


# -----------------------------------------------------------------
# 6. FIGURE 3 — vorticity comparison
# -----------------------------------------------------------------
# CFD truth is computed by sampling UMean on a regular grid (ImageData.sample
# is the correct API; .interpolate with a small radius leaves holes).
# PINN panels reuse the pre-rendered v2_vorticity.png files since the
# checkpoints are .pt files and torch is unavailable on this sandbox.

def cfd_vorticity_image():
    m = pv.read(os.path.join(ROOT, "Re40.vtk"))
    m_pts = m.cell_data_to_point_data()
    nx, ny = 400, 200
    xs = np.linspace(-8, 12, nx)
    ys = np.linspace(-8,  8, ny)
    grid = pv.ImageData(
        dimensions=(nx, ny, 1),
        spacing=(xs[1] - xs[0], ys[1] - ys[0], 1.0),
        origin=(xs[0], ys[0], 0.0),
    )
    probed = grid.sample(m_pts)
    U = np.asarray(probed.point_data["UMean"]).reshape(ny, nx, 3)

    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    du_dy = np.gradient(U[..., 0], dy, axis=0)
    dv_dx = np.gradient(U[..., 1], dx, axis=1)
    W = dv_dx - du_dy

    X, Y = np.meshgrid(xs, ys)
    mask_cyl = (X**2 + Y**2) < 0.5**2
    W[mask_cyl] = np.nan
    return xs, ys, W

xs, ys, W_cfd = cfd_vorticity_image()
print(f"CFD vorticity range: [{np.nanmin(W_cfd):+.2f}, {np.nanmax(W_cfd):+.2f}]  "
      f"|peak| = {np.nanmax(np.abs(W_cfd)):.2f}")

# Render two versions: (a) full domain comparison matching the PINN PNGs,
# (b) zoomed CFD-truth near-wake for reference.

# === (a) Full-domain panel ===
fig, axs = plt.subplots(2, 2, figsize=(14.0, 7.6))
axs = axs.flatten()
vlim = 8.0

im0 = axs[0].pcolormesh(xs, ys, W_cfd, shading="auto", cmap="coolwarm",
                        vmin=-vlim, vmax=vlim)
axs[0].set_title("CFD truth — vorticity from UMean", fontsize=11)
axs[0].set_xlim(-8, 12); axs[0].set_ylim(-8, 8)
axs[0].set_aspect("equal")
plt.colorbar(im0, ax=axs[0], shrink=0.85, label="ω")
theta = np.linspace(0, 2*np.pi, 80)
axs[0].fill(0.5*np.cos(theta), 0.5*np.sin(theta), color="0.2", ec="black")

for i, r in enumerate(RUNS_SPEC, start=1):
    img = mpimg.imread(r["viz_png"])
    axs[i].imshow(img)
    axs[i].set_title(f"{r['name']}\nPINN vorticity (matplotlib default ±range)",
                     fontsize=10.5)
    axs[i].set_xticks([]); axs[i].set_yticks([])

fig.suptitle("Vorticity (full v2 domain -8..12 × -8..8): CFD truth vs three PINN runs  "
             "[small net 32/3]", y=1.00, fontsize=11.5)
fig.tight_layout()
out = os.path.join(OUT, "fig03_vorticity_panel.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {out}")

# === (b) Zoomed CFD near-wake reference ===
fig, ax = plt.subplots(figsize=(7.2, 4.5))
im = ax.pcolormesh(xs, ys, W_cfd, shading="auto", cmap="coolwarm",
                   vmin=-vlim, vmax=vlim)
ax.fill(0.5*np.cos(theta), 0.5*np.sin(theta), color="0.2", ec="black")
ax.set_xlim(-2, 10); ax.set_ylim(-3, 3); ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("CFD truth — vorticity near wake  (|peak| ≈ 8, recirculation bubble visible)")
plt.colorbar(im, ax=ax, shrink=0.85, label="ω")
fig.tight_layout()
out = os.path.join(OUT, "fig03b_cfd_vorticity_zoom.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"  wrote {out}")

print("\nDone.")
