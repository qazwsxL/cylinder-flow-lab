"""
run_analysis_re40.py
====================

One-shot analysis script that mirrors analysis_updated_re40_clean.ipynb but is
self-contained, deterministic, and writes every figure + summary to disk so the
results can be folded back into the slide deck.

Run on the GPU node (or any node with the trained checkpoint, torch and
pyvista):

    python run_analysis_re40.py \
        --vtk-path Re40.vtk \
        --save-dir checkpoints_sanity_A \
        --out-dir  analysis_runA \
        --width 96 --depth 5

Outputs in --out-dir:
    summary.txt
    01_velocity_scatter.png
    02_velocity_error_map.png
    03_pde_residual_map.png
    04_pde_residual_hist.png
    05_wall_noslip.png
    06_inlet_topbottom.png
    07_vorticity_cfd.png
    07b_vorticity_cfd_vtk.png   (only if pyvista is available)
    07_vorticity_pinn.png
    07_vorticity_diff.png
"""
import os, argparse, json, math, sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Make cfp40 importable from the cwd (same directory as this script normally).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfp40                                                            # noqa


# ----------------------------------------------------------------- helpers
def percentile(a, q):
    return float(np.quantile(np.asarray(a), q))


def write_summary(path, blocks):
    with open(path, "w") as f:
        for title, kv in blocks:
            f.write(f"=== {title} ===\n")
            for k, v in kv.items():
                if isinstance(v, float):
                    f.write(f"  {k:<28s} = {v:.6e}\n")
                else:
                    f.write(f"  {k:<28s} = {v}\n")
            f.write("\n")


# ============================================================ VELOCITY
def velocity_check(model, snapshot, device, xlim, ylim, radius=0.5):
    sx, sy = snapshot.x, snapshot.y
    su, sv = snapshot.u, snapshot.v
    in_box = (sx >= xlim[0]) & (sx <= xlim[1]) & (sy >= ylim[0]) & (sy <= ylim[1])
    out_cyl = (sx ** 2 + sy ** 2) >= radius ** 2
    m = in_box & out_cyl
    x = torch.tensor(sx[m], device=device, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(sy[m], device=device, dtype=torch.float32).view(-1, 1)
    t = torch.full_like(x, float(snapshot.t))
    _, u_p, v_p, _ = cfp40.model_uvp(model, x, y, t)
    u_p = u_p.detach().cpu().numpy().ravel()
    v_p = v_p.detach().cpu().numpy().ravel()
    eu = u_p - su[m]
    ev = v_p - sv[m]
    mag = np.sqrt(eu ** 2 + ev ** 2)
    speed = np.sqrt(su[m] ** 2 + sv[m] ** 2)

    metrics = {
        "n_points":          int(m.sum()),
        "speed_true_mean":   float(speed.mean()),
        "mae_u":             float(np.mean(np.abs(eu))),
        "mae_v":             float(np.mean(np.abs(ev))),
        "rmse_u":            float(np.sqrt(np.mean(eu ** 2))),
        "rmse_v":            float(np.sqrt(np.mean(ev ** 2))),
        "max_abs_eu":        float(np.max(np.abs(eu))),
        "max_abs_ev":        float(np.max(np.abs(ev))),
        "rel_l2_u":          float(np.linalg.norm(eu) / np.linalg.norm(su[m])),
        "rel_l2_v":          float(np.linalg.norm(ev) / max(np.linalg.norm(sv[m]), 1e-30)),
        "rel_l2_uv":         float(np.linalg.norm(np.concatenate([eu, ev])) /
                                   np.linalg.norm(np.concatenate([su[m], sv[m]]))),
    }
    detail = {"x": sx[m], "y": sy[m], "u_true": su[m], "v_true": sv[m],
              "u_pred": u_p, "v_pred": v_p, "abs_err_mag": mag}
    return metrics, detail


# ========================================================== PDE residual
def pde_residual_diagnostics(model, device, xlim, ylim, Re=40.0, n_f=20000, radius=0.5):
    x_f, y_f, t_f = cfp40.sample_collocation(
        n_f, device=device, t_value=0.0, xlim=xlim, ylim=ylim,
        cylinder_center=(0.0, 0.0), radius=radius,
    )
    f_u, f_v, _div = cfp40.compute_pde_residuals(model, x_f, y_f, t_f, Re=Re)
    fu = f_u.detach().cpu().numpy().ravel()
    fv = f_v.detach().cpu().numpy().ravel()
    mag = np.sqrt(fu ** 2 + fv ** 2)

    x_np = x_f.detach().cpu().numpy().ravel()
    y_np = y_f.detach().cpu().numpy().ravel()
    near_cyl = (x_np ** 2 + y_np ** 2) < (radius + 0.5) ** 2
    far = ~near_cyl

    metrics = {
        "n_collocation":             int(len(mag)),
        "pde_rmse_mag":              float(np.sqrt(np.mean(mag ** 2))),
        "pde_mean_mag":              float(np.mean(mag)),
        "pde_p90_mag":               percentile(mag, 0.90),
        "pde_p99_mag":               percentile(mag, 0.99),
        "near_cyl_pde_rmse_mag":     float(np.sqrt(np.mean(mag[near_cyl] ** 2)))
                                     if near_cyl.any() else float("nan"),
        "far_pde_rmse_mag":          float(np.sqrt(np.mean(mag[far] ** 2)))
                                     if far.any() else float("nan"),
    }
    detail = {"x": x_np, "y": y_np, "f_u": fu, "f_v": fv, "mag": mag}
    return metrics, detail


# ============================================================ Boundary diag
def boundary_diagnostics(model, device, xlim, ylim, n=2000, radius=0.5):
    # inlet
    yi = torch.linspace(ylim[0], ylim[1], n, device=device).view(-1, 1)
    xi = torch.full_like(yi, float(xlim[0]))
    ti = torch.zeros_like(yi)
    _, u_i, v_i, _ = cfp40.model_uvp(model, xi, yi, ti)

    # top / bottom
    xt = torch.linspace(xlim[0], xlim[1], n, device=device).view(-1, 1)
    yt = torch.full_like(xt, float(ylim[1]))
    tt = torch.zeros_like(xt)
    _, u_t, v_t, _ = cfp40.model_uvp(model, xt, yt, tt)
    yb = torch.full_like(xt, float(ylim[0]))
    _, u_b, v_b, _ = cfp40.model_uvp(model, xt, yb, tt)

    # cylinder wall
    th = torch.linspace(0.0, 2 * math.pi, n, device=device).view(-1, 1)
    xw = radius * torch.cos(th)
    yw = radius * torch.sin(th)
    tw = torch.zeros_like(th)
    _, u_w, v_w, _ = cfp40.model_uvp(model, xw, yw, tw)

    def t(z): return z.detach().cpu().numpy().ravel()
    metrics = {
        "inlet_u_minus_1_rmse":     float(np.sqrt(np.mean((t(u_i) - 1.0) ** 2))),
        "inlet_v_rmse":             float(np.sqrt(np.mean(t(v_i) ** 2))),
        "top_u_minus_1_rmse":       float(np.sqrt(np.mean((t(u_t) - 1.0) ** 2))),
        "bottom_u_minus_1_rmse":    float(np.sqrt(np.mean((t(u_b) - 1.0) ** 2))),
        "top_v_rmse":               float(np.sqrt(np.mean(t(v_t) ** 2))),
        "bottom_v_rmse":            float(np.sqrt(np.mean(t(v_b) ** 2))),
        "wall_speed_max":           float(np.max(np.sqrt(t(u_w) ** 2 + t(v_w) ** 2))),
        "wall_speed_p95":           percentile(np.sqrt(t(u_w) ** 2 + t(v_w) ** 2), 0.95),
        "wall_u_max_abs":           float(np.max(np.abs(t(u_w)))),
        "wall_v_max_abs":           float(np.max(np.abs(t(v_w)))),
    }
    detail = {
        "theta": t(th), "u_wall": t(u_w), "v_wall": t(v_w),
        "wall_speed": np.sqrt(t(u_w) ** 2 + t(v_w) ** 2),
        "yi": t(yi), "u_inlet": t(u_i), "v_inlet": t(v_i),
        "x_top": t(xt), "u_top": t(u_t), "v_top": t(v_t),
        "u_bot": t(u_b), "v_bot": t(v_b),
    }
    return metrics, detail


# ========================================================== Vorticity FD on grid
def fd_vorticity_from_cfd(snapshot, nx=260, ny=140, xlim=(-2.0, 10.0), ylim=(-3.0, 3.0)):
    xs = np.linspace(xlim[0], xlim[1], nx, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    U = np.full((ny, nx), np.nan, dtype=np.float32)
    V = np.full((ny, nx), np.nan, dtype=np.float32)
    C = np.zeros((ny, nx), dtype=np.int32)

    ix = np.clip(np.round((snapshot.x - xlim[0]) / (xlim[1] - xlim[0]) * (nx - 1)).astype(int), 0, nx - 1)
    iy = np.clip(np.round((snapshot.y - ylim[0]) / (ylim[1] - ylim[0]) * (ny - 1)).astype(int), 0, ny - 1)

    Usum = np.zeros_like(U); Vsum = np.zeros_like(V)
    for k in range(len(snapshot.x)):
        Usum[iy[k], ix[k]] += snapshot.u[k]
        Vsum[iy[k], ix[k]] += snapshot.v[k]
        C[iy[k], ix[k]] += 1
    mask = C > 0
    U[mask] = Usum[mask] / C[mask]
    V[mask] = Vsum[mask] / C[mask]

    dx = (xlim[1] - xlim[0]) / (nx - 1)
    dy = (ylim[1] - ylim[0]) / (ny - 1)
    Vy_x = np.gradient(V, dx, axis=1)
    Uy_y = np.gradient(U, dy, axis=0)
    W = Vy_x - Uy_y
    cyl = (X ** 2 + Y ** 2) <= 0.5 ** 2
    W[cyl] = np.nan
    return X, Y, W


def pinn_vorticity_on_grid(model, device, nx=260, ny=140, xlim=(-2.0, 10.0), ylim=(-3.0, 3.0)):
    xs = np.linspace(xlim[0], xlim[1], nx, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    x = torch.tensor(X.reshape(-1, 1), device=device).float()
    y = torch.tensor(Y.reshape(-1, 1), device=device).float()
    t = torch.zeros_like(x)
    omega = cfp40.model_vorticity(model, x, y, t)
    W = omega.detach().cpu().numpy().reshape(ny, nx)
    cyl = (X ** 2 + Y ** 2) <= 0.5 ** 2
    W[cyl] = np.nan
    return X, Y, W


# =========================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk-path", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--out-dir",  required=True)
    ap.add_argument("--width",   type=int, default=96)
    ap.add_argument("--depth",   type=int, default=5)
    ap.add_argument("--re",      type=float, default=40.0)
    ap.add_argument("--xlim",    type=float, nargs=2, default=[-3.0, 12.0])
    ap.add_argument("--ylim",    type=float, nargs=2, default=[-4.0,  4.0])
    ap.add_argument("--vort-xlim", type=float, nargs=2, default=[-2.0, 10.0])
    ap.add_argument("--vort-ylim", type=float, nargs=2, default=[-3.0,  3.0])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    snapshot = cfp40.load_single_vtk(args.vtk_path, t_value=0.0)
    model = cfp40.load_model_for_viz(args.save_dir, device,
                                     width=args.width, depth=args.depth)
    model.eval()

    # ---------- 1. velocity ----------
    print("[1/4] velocity check on CFD points")
    vel_m, vel_d = velocity_check(model, snapshot, device,
                                  args.xlim, args.ylim)
    for k, v in vel_m.items():
        print(f"  {k:<24s} = {v}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].scatter(vel_d["u_true"], vel_d["u_pred"], s=4, alpha=0.4)
    lim = max(abs(vel_d["u_true"]).max(), abs(vel_d["u_pred"]).max()) * 1.05
    ax[0].plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax[0].set_xlabel("u (CFD)"); ax[0].set_ylabel("u (PINN)")
    ax[0].set_title(f"u — rel_l2 = {vel_m['rel_l2_u']:.2e}")
    ax[0].grid(alpha=0.3)
    ax[1].scatter(vel_d["v_true"], vel_d["v_pred"], s=4, alpha=0.4, color="C1")
    lim = max(abs(vel_d["v_true"]).max(), abs(vel_d["v_pred"]).max()) * 1.05
    ax[1].plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax[1].set_xlabel("v (CFD)"); ax[1].set_ylabel("v (PINN)")
    ax[1].set_title(f"v — rel_l2 = {vel_m['rel_l2_v']:.2e}")
    ax[1].grid(alpha=0.3)
    fig.suptitle("Velocity agreement on CFD point cloud", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "01_velocity_scatter.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    sc = plt.scatter(vel_d["x"], vel_d["y"], c=vel_d["abs_err_mag"], s=5,
                     cmap="magma")
    plt.colorbar(sc, label="|velocity error|")
    plt.gca().add_patch(plt.Circle((0, 0), 0.5, color="cyan", fill=False))
    plt.gca().set_aspect("equal")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"Velocity error map  (mae_u={vel_m['mae_u']:.2e}, mae_v={vel_m['mae_v']:.2e})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "02_velocity_error_map.png"), dpi=150)
    plt.close()

    # ---------- 2. PDE residual ----------
    print("[2/4] PDE residual diagnostics")
    pde_m, pde_d = pde_residual_diagnostics(model, device,
                                            args.xlim, args.ylim,
                                            Re=args.re, n_f=20000)
    for k, v in pde_m.items():
        print(f"  {k:<24s} = {v}")

    plt.figure(figsize=(8, 4.5))
    sc = plt.scatter(pde_d["x"], pde_d["y"], c=np.log10(pde_d["mag"] + 1e-20),
                     s=5, cmap="magma")
    plt.colorbar(sc, label="log10 |PDE residual|")
    plt.gca().add_patch(plt.Circle((0, 0), 0.5, color="cyan", fill=False))
    plt.gca().set_aspect("equal")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"PDE residual map  (rmse={pde_m['pde_rmse_mag']:.2e}, p99={pde_m['pde_p99_mag']:.2e})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "03_pde_residual_map.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(np.log10(pde_d["mag"] + 1e-20), bins=60)
    plt.xlabel("log10 |PDE residual|"); plt.ylabel("count")
    plt.title("PDE residual distribution on collocation points")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "04_pde_residual_hist.png"), dpi=150)
    plt.close()

    # ---------- 3. boundary ----------
    print("[3/4] boundary diagnostics")
    bc_m, bc_d = boundary_diagnostics(model, device, args.xlim, args.ylim, n=2000)
    for k, v in bc_m.items():
        print(f"  {k:<24s} = {v}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.0))
    order = np.argsort(bc_d["theta"])
    ax[0].plot(bc_d["theta"][order], bc_d["u_wall"][order], label="u_wall")
    ax[0].plot(bc_d["theta"][order], bc_d["v_wall"][order], label="v_wall")
    ax[0].plot(bc_d["theta"][order], bc_d["wall_speed"][order], label="|u_wall|", lw=2)
    ax[0].set_xlabel("θ on cylinder"); ax[0].set_ylabel("velocity")
    ax[0].set_title(f"Cylinder wall  max|u|={bc_m['wall_u_max_abs']:.1e}, max|v|={bc_m['wall_v_max_abs']:.1e}")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(bc_d["yi"], bc_d["u_inlet"], label="u inlet  (target=1)")
    ax[1].plot(bc_d["yi"], bc_d["v_inlet"], label="v inlet  (target=0)")
    ax[1].axhline(1.0, color="grey", ls="--", lw=0.5)
    ax[1].axhline(0.0, color="grey", ls="--", lw=0.5)
    ax[1].set_xlabel("y"); ax[1].set_ylabel("velocity at x=xmin")
    ax[1].set_title(f"Inlet  rmse(u-1)={bc_m['inlet_u_minus_1_rmse']:.2e}")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "05_wall_noslip.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].plot(bc_d["x_top"], bc_d["u_top"], label="u top  (target=1)")
    ax[0].plot(bc_d["x_top"], bc_d["v_top"], label="v top  (target=0)")
    ax[0].axhline(1.0, color="grey", ls="--", lw=0.5)
    ax[0].axhline(0.0, color="grey", ls="--", lw=0.5)
    ax[0].set_title(f"Top boundary  rmse(u-1)={bc_m['top_u_minus_1_rmse']:.2e}")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("velocity"); ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(bc_d["x_top"], bc_d["u_bot"], label="u bottom  (target=1)")
    ax[1].plot(bc_d["x_top"], bc_d["v_bot"], label="v bottom  (target=0)")
    ax[1].axhline(1.0, color="grey", ls="--", lw=0.5)
    ax[1].axhline(0.0, color="grey", ls="--", lw=0.5)
    ax[1].set_title(f"Bottom boundary  rmse(u-1)={bc_m['bottom_u_minus_1_rmse']:.2e}")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("velocity"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "06_inlet_topbottom.png"), dpi=150)
    plt.close()

    # ---------- 4. vorticity comparison ----------
    print("[4/4] vorticity comparison")
    Xv, Yv, W_cfd = fd_vorticity_from_cfd(snapshot, xlim=args.vort_xlim, ylim=args.vort_ylim)
    _,  _,  W_pinn = pinn_vorticity_on_grid(model, device, xlim=args.vort_xlim, ylim=args.vort_ylim)
    diff = W_pinn - W_cfd
    diff_finite = diff[np.isfinite(diff)]
    omega_metrics = {
        "vort_rmse":     float(np.sqrt(np.mean(diff_finite ** 2))) if diff_finite.size else float("nan"),
        "vort_max_abs":  float(np.max(np.abs(diff_finite))) if diff_finite.size else float("nan"),
        "vort_cfd_max":  float(np.nanmax(W_cfd)),
        "vort_cfd_min":  float(np.nanmin(W_cfd)),
        "vort_pinn_max": float(np.nanmax(W_pinn)),
        "vort_pinn_min": float(np.nanmin(W_pinn)),
    }
    for k, v in omega_metrics.items():
        print(f"  {k:<24s} = {v}")

    vmax = max(abs(np.nanmin(W_cfd)), abs(np.nanmax(W_cfd)),
               abs(np.nanmin(W_pinn)), abs(np.nanmax(W_pinn)))
    for field, name, fname in [
        (W_cfd,  "CFD vorticity (FD on grid)",     "07_vorticity_cfd.png"),
        (W_pinn, "PINN vorticity",                 "07_vorticity_pinn.png"),
        (diff,   "Vorticity error  (PINN − CFD)",  "07_vorticity_diff.png"),
    ]:
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(Xv, Yv, field, shading="auto", cmap="coolwarm",
                       vmin=-vmax, vmax=vmax)
        plt.colorbar(label="ω_z")
        plt.gca().add_patch(plt.Circle((0, 0), 0.5, color="k", fill=False))
        plt.gca().set_aspect("equal")
        plt.xlabel("x"); plt.ylabel("y"); plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, fname), dpi=150)
        plt.close()

    # optional: VTK-derivative-based CFD vorticity (more accurate than FD)
    try:
        import pyvista as pv
        mesh = pv.read(args.vtk_path)
        if "U" in mesh.point_data:
            src = mesh
        elif "U" in mesh.cell_data:
            src = mesh.cell_data_to_point_data()
        else:
            src = None
        if src is not None:
            der = src.compute_derivative(scalars="U", gradient=True, vorticity=True)
            vort = np.asarray(der.point_data.get("vorticity",
                                                 der.cell_data.get("vorticity")))
            ozz = vort[:, 2]
            pts = der.points
            plt.figure(figsize=(8, 4))
            sc = plt.scatter(pts[:, 0], pts[:, 1], c=ozz, s=6, cmap="coolwarm",
                             vmin=-vmax, vmax=vmax)
            plt.colorbar(sc, label="ω_z (VTK derivative)")
            plt.gca().add_patch(plt.Circle((0, 0), 0.5, color="k", fill=False))
            plt.gca().set_aspect("equal")
            plt.xlabel("x"); plt.ylabel("y")
            plt.title("CFD vorticity from VTK gradient")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "07b_vorticity_cfd_vtk.png"), dpi=150)
            plt.close()
    except Exception as e:
        print(f"[skip] pyvista vorticity overlay failed: {e}")

    # ---------- summary ----------
    write_summary(os.path.join(args.out_dir, "summary.txt"), [
        ("Velocity",          vel_m),
        ("PDE residual",      pde_m),
        ("Boundary",          bc_m),
        ("Vorticity error",   omega_metrics),
    ])
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({"velocity": vel_m, "pde": pde_m, "bc": bc_m,
                   "vorticity": omega_metrics}, f, indent=2)
    print(f"\n[done] outputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
