"""
sanity_check_re40.py
====================

Pre-training consistency check between the CFD snapshot (Re40.vtk) and the
PINN setup we are about to use (cfp40_v2.py / cfp40.py).

We want to answer FIVE questions BEFORE running any training:

  Q1. What's actually in the VTK?
        instantaneous vs time-mean? is there pressure? are there fluctuations?
        which array does cfp40.load_single_vtk pick?

  Q2. Does the CFD geometry match the PINN setup?
        bbox, cylinder radius/center, domain extent vs --x-min/--x-max etc.

  Q3. Do the CFD boundary VALUES match the BCs the PINN encodes?
        - inlet (x=PINN x_min): is u≈1, v≈0?
        - top   (y=PINN y_max): is u≈1, v≈0?
        - bottom(y=PINN y_min): same
        - wall  (r=R)        : is u=v=0?

  Q4. Is the CFD field STEADY and SYMMETRIC at Re=40?
        - magnitude of UPrime2Mean (turbulent stresses) — should be ~0
        - U vs UMean difference — should be ~0
        - top/bottom symmetry of u(x,y) and antisymmetry of v(x,y)

  Q5. Does the CFD field SATISFY THE NS EQUATION the PINN tries to solve?
        For steady incompressible NS at Re=40:
            u u_x + v u_y + p_x − (1/Re) (u_xx + u_yy) = 0
            u v_x + v v_y + p_y − (1/Re) (v_xx + v_yy) = 0
            u_x + v_y                                  = 0
        We finite-difference the CFD field on a structured grid (interpolated
        from the unstructured cell centers) and report the residual magnitudes.
        If these are large then no PINN with these settings can ever drive the
        PDE residual + the data loss to zero simultaneously — that's a setup
        bug, not a training problem.

Outputs:
  - prints a structured report
  - writes diagnostic PNGs and a JSON summary into  ./sanity_check_re40/

Usage:
  python sanity_check_re40.py --vtk-path Re40.vtk \
        --x-min -3 --x-max 12 --y-min -4 --y-max 4 --Re 40 --radius 0.5
"""

from __future__ import annotations

import os
import json
import argparse

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------

def _pick(arr_names, candidates):
    lower = {n.lower(): n for n in arr_names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_all_fields(vtk_path: str) -> dict:
    """Return dict of (centers_xyz, plus available arrays as 2D vec or scalar)."""
    mesh = pv.read(vtk_path)
    if mesh.n_cells > 0:
        pts = mesh.cell_centers().points
        src = mesh.cell_data
    else:
        pts = mesh.points
        src = mesh.point_data
    keys = list(src.keys())
    fields = {"x": pts[:, 0].astype(np.float64),
              "y": pts[:, 1].astype(np.float64),
              "z": pts[:, 2].astype(np.float64),
              "_arrays_in_vtk": keys}
    # velocity (instantaneous and mean)
    for tag, candidates in [
        ("U",     ["U", "u", "velocity", "Velocity", "vel"]),
        ("UMean", ["UMean", "Umean", "Uavg", "U_mean", "velocity_mean"]),
        ("UPrime2Mean", ["UPrime2Mean", "UprimeMean", "Reynolds_stress"]),
    ]:
        name = _pick(keys, candidates)
        if name is None:
            continue
        a = np.asarray(src[name])
        fields[tag] = a
    # pressure
    for tag, candidates in [
        ("p",     ["p", "pressure"]),
        ("pMean", ["pMean", "pmean", "p_mean"]),
        ("pPrime2Mean", ["pPrime2Mean"]),
    ]:
        name = _pick(keys, candidates)
        if name is None:
            continue
        a = np.asarray(src[name])
        fields[tag] = a
    return fields


# ----------------------------------------------------------------------
# Boundary slice helpers
# ----------------------------------------------------------------------

def near_strip(x, y, xref=None, yref=None, tol=0.3):
    if xref is not None:
        return np.where(np.abs(x - xref) < tol)[0]
    if yref is not None:
        return np.where(np.abs(y - yref) < tol)[0]
    raise ValueError("xref or yref must be set")


def near_cylinder_ring(x, y, R=0.5, tol=0.05):
    r = np.sqrt(x**2 + y**2)
    return np.where(np.abs(r - R) < tol)[0]


# ----------------------------------------------------------------------
# Structured-grid interpolation for finite-difference NS residual check
# ----------------------------------------------------------------------

def interp_to_grid(x, y, val, nx, ny, xlim, ylim, method="linear"):
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([x, y])
    Z = griddata(pts, val, (X, Y), method=method)
    return X, Y, Z


def central_diff(F, h, axis):
    G = np.zeros_like(F)
    if axis == 0:  # along y
        G[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2 * h)
    else:           # axis 1, along x
        G[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2 * h)
    return G


def laplacian(F, hx, hy):
    Lxx = np.zeros_like(F)
    Lyy = np.zeros_like(F)
    Lxx[:, 1:-1] = (F[:, 2:] - 2 * F[:, 1:-1] + F[:, :-2]) / (hx ** 2)
    Lyy[1:-1, :] = (F[2:, :] - 2 * F[1:-1, :] + F[:-2, :]) / (hy ** 2)
    return Lxx + Lyy


def cfd_ns_residual(x, y, u, v, p, Re, xlim, ylim, nx=320, ny=200, mask_radius=0.6):
    """Interpolate CFD onto a structured grid, FD the NS residual."""
    X, Y, U = interp_to_grid(x, y, u, nx, ny, xlim, ylim)
    _, _, V = interp_to_grid(x, y, v, nx, ny, xlim, ylim)
    if p is not None:
        _, _, P = interp_to_grid(x, y, p, nx, ny, xlim, ylim)
    else:
        P = None

    hx = (xlim[1] - xlim[0]) / (nx - 1)
    hy = (ylim[1] - ylim[0]) / (ny - 1)

    Ux = central_diff(U, hx, axis=1)
    Uy = central_diff(U, hy, axis=0)
    Vx = central_diff(V, hx, axis=1)
    Vy = central_diff(V, hy, axis=0)
    Uxx_yy = laplacian(U, hx, hy)
    Vxx_yy = laplacian(V, hx, hy)

    div = Ux + Vy

    if P is not None:
        Px = central_diff(P, hx, axis=1)
        Py = central_diff(P, hy, axis=0)
        f_u = U * Ux + V * Uy + Px - (1.0 / Re) * Uxx_yy
        f_v = U * Vx + V * Vy + Py - (1.0 / Re) * Vxx_yy
    else:
        # without pressure we can only check  curl(NS)  (eliminates ∇p)
        f_u = U * Ux + V * Uy - (1.0 / Re) * Uxx_yy
        f_v = U * Vx + V * Vy - (1.0 / Re) * Vxx_yy
        # this includes pressure-gradient contribution; its magnitude is
        # what an inviscid-balance term would equal pressure gradient

    # mask out cylinder + boundary cells (FD is invalid there)
    R2 = (X ** 2 + Y ** 2) <= mask_radius ** 2
    edge = np.zeros_like(R2, dtype=bool)
    edge[:2, :] = True; edge[-2:, :] = True; edge[:, :2] = True; edge[:, -2:] = True
    bad = R2 | edge | np.isnan(U) | np.isnan(V)
    if P is not None:
        bad |= np.isnan(P)

    return {
        "X": X, "Y": Y, "U": U, "V": V, "P": P,
        "div": div, "f_u": f_u, "f_v": f_v, "bad": bad,
        "hx": hx, "hy": hy,
    }


def stats(arr, mask=None):
    if mask is not None:
        a = arr[~mask]
    else:
        a = arr.ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "rmse": float(np.sqrt(np.mean(a ** 2))),
        "max_abs": float(np.max(np.abs(a))),
        "p90_abs": float(np.percentile(np.abs(a), 90)),
        "p99_abs": float(np.percentile(np.abs(a), 99)),
    }


# ----------------------------------------------------------------------
# Symmetry check
# ----------------------------------------------------------------------

def symmetry_check(x, y, u, v, xlim, ylim, nx=200, ny=200):
    X, Y, U = interp_to_grid(x, y, u, nx, ny, xlim, ylim)
    _, _, V = interp_to_grid(x, y, v, nx, ny, xlim, ylim)
    # mirror across y=0 (flip rows)
    U_mirror = U[::-1, :]
    V_mirror = -V[::-1, :]   # v should be antisymmetric
    # mask cylinder + nan
    R2 = (X ** 2 + Y ** 2) <= 0.6 ** 2
    bad = R2 | np.isnan(U) | np.isnan(V) | np.isnan(U_mirror) | np.isnan(V_mirror)
    du = (U - U_mirror)
    dv = (V - V_mirror)
    return {
        "u_symmetry_rmse":   float(np.sqrt(np.nanmean(du[~bad] ** 2))),
        "u_symmetry_maxabs": float(np.max(np.abs(du[~bad]))),
        "v_antisym_rmse":    float(np.sqrt(np.nanmean(dv[~bad] ** 2))),
        "v_antisym_maxabs":  float(np.max(np.abs(dv[~bad]))),
        "u_max":             float(np.nanmax(np.abs(U[~bad]))),
        "v_max":             float(np.nanmax(np.abs(V[~bad]))),
    }


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def fmt(x): return f"{x:.4e}" if isinstance(x, float) else str(x)


def section(title):
    bar = "=" * 72
    print(); print(bar); print(title); print(bar)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vtk-path", type=str, default="Re40.vtk")
    p.add_argument("--out-dir", type=str, default="sanity_check_re40")
    p.add_argument("--x-min", type=float, default=-3.0)
    p.add_argument("--x-max", type=float, default=12.0)
    p.add_argument("--y-min", type=float, default=-4.0)
    p.add_argument("--y-max", type=float, default=4.0)
    p.add_argument("--Re", type=float, default=40.0)
    p.add_argument("--radius", type=float, default=0.5)
    p.add_argument("--nx", type=int, default=320)
    p.add_argument("--ny", type=int, default=200)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    xlim = (args.x_min, args.x_max)
    ylim = (args.y_min, args.y_max)

    summary: dict = {}

    # -----------------------------------------------------------------
    section("Q1. What's in the VTK?")
    # -----------------------------------------------------------------
    fields = load_all_fields(args.vtk_path)
    print(f"  arrays present       : {fields['_arrays_in_vtk']}")
    print(f"  CFD bbox x           : [{fields['x'].min():.3f}, {fields['x'].max():.3f}]")
    print(f"  CFD bbox y           : [{fields['y'].min():.3f}, {fields['y'].max():.3f}]")
    print(f"  CFD bbox z           : [{fields['z'].min():.3f}, {fields['z'].max():.3f}]")
    print(f"  n cell-centers       : {fields['x'].size}")

    has_p     = "p"     in fields
    has_pMean = "pMean" in fields
    has_U     = "U"     in fields
    has_Umean = "UMean" in fields
    has_Up2   = "UPrime2Mean" in fields
    print(f"  velocity:   U={has_U}   UMean={has_Umean}   UPrime2Mean={has_Up2}")
    print(f"  pressure:   p={has_p}   pMean={has_pMean}")

    summary["Q1"] = {
        "arrays": fields["_arrays_in_vtk"],
        "bbox":   {"x":[float(fields['x'].min()), float(fields['x'].max())],
                   "y":[float(fields['y'].min()), float(fields['y'].max())],
                   "z":[float(fields['z'].min()), float(fields['z'].max())]},
        "n_cells": int(fields['x'].size),
        "has":     {"p":has_p, "pMean":has_pMean,
                    "U":has_U, "UMean":has_Umean, "UPrime2Mean":has_Up2},
    }

    print("  -> cfp40.load_single_vtk picks 'U' first (instantaneous), NOT 'UMean'.")

    # -----------------------------------------------------------------
    section("Q2. Geometry consistency")
    # -----------------------------------------------------------------
    in_box = ((fields["x"] >= args.x_min) & (fields["x"] <= args.x_max) &
              (fields["y"] >= args.y_min) & (fields["y"] <= args.y_max))
    out_cyl = (fields["x"] ** 2 + fields["y"] ** 2) >= args.radius ** 2
    in_pinn_box = in_box & out_cyl
    print(f"  PINN box: x∈[{args.x_min},{args.x_max}], y∈[{args.y_min},{args.y_max}]")
    print(f"  CFD points inside PINN box (& outside cyl) : {int(in_pinn_box.sum())}/{fields['x'].size}")
    print(f"  -> CFD domain is {'larger than' if fields['x'].min() < args.x_min or fields['x'].max() > args.x_max or fields['y'].min() < args.y_min or fields['y'].max() > args.y_max else 'smaller-than-or-equal-to'} PINN box")

    summary["Q2"] = {"n_inside_pinn_box": int(in_pinn_box.sum()),
                     "n_total": int(fields['x'].size)}

    # -----------------------------------------------------------------
    section("Q3. Boundary VALUES — does CFD match PINN BCs?")
    # -----------------------------------------------------------------
    # we use UMean if available (steady), otherwise U
    Uvec = fields["UMean"] if has_Umean else fields["U"]
    src_tag = "UMean" if has_Umean else "U"
    u_all = Uvec[:, 0].astype(np.float64)
    v_all = Uvec[:, 1].astype(np.float64)
    print(f"  using velocity source: {src_tag}")

    # inlet: in CFD, the inlet is at x = CFD x_min. Check both PINN x_min slice and CFD x_min slice
    bcs_summary = {}
    for label, sel in [
        (f"PINN inlet  (x≈{args.x_min})", near_strip(fields["x"], fields["y"], xref=args.x_min, tol=0.3)),
        (f"CFD  inlet  (x≈{fields['x'].min():.2f})", near_strip(fields["x"], fields["y"], xref=fields['x'].min(), tol=0.3)),
        (f"PINN top    (y≈{args.y_max})", near_strip(fields["x"], fields["y"], yref=args.y_max, tol=0.3)),
        (f"CFD  top    (y≈{fields['y'].max():.2f})", near_strip(fields["x"], fields["y"], yref=fields['y'].max(), tol=0.3)),
        (f"PINN bot    (y≈{args.y_min})", near_strip(fields["x"], fields["y"], yref=args.y_min, tol=0.3)),
        (f"CFD  bot    (y≈{fields['y'].min():.2f})", near_strip(fields["x"], fields["y"], yref=fields['y'].min(), tol=0.3)),
    ]:
        if sel.size == 0:
            print(f"  {label:40s}  no points")
            continue
        u_b = u_all[sel]; v_b = v_all[sel]
        rmse_u_minus_1 = float(np.sqrt(np.mean((u_b - 1.0) ** 2)))
        rmse_v         = float(np.sqrt(np.mean(v_b ** 2)))
        max_u_minus_1  = float(np.max(np.abs(u_b - 1.0)))
        max_v          = float(np.max(np.abs(v_b)))
        u_mean = float(np.mean(u_b))
        v_mean = float(np.mean(v_b))
        print(f"  {label:40s}  n={sel.size:5d}  <u>={u_mean:+.3e}  <v>={v_mean:+.3e}  "
              f"rmse(u-1)={rmse_u_minus_1:.2e}  rmse(v)={rmse_v:.2e}")
        bcs_summary[label.strip()] = {
            "n": int(sel.size), "u_mean": u_mean, "v_mean": v_mean,
            "rmse_u_minus_1": rmse_u_minus_1, "rmse_v": rmse_v,
            "max_u_minus_1": max_u_minus_1, "max_v": max_v,
        }

    sel = near_cylinder_ring(fields["x"], fields["y"], R=args.radius, tol=0.05)
    if sel.size > 0:
        u_w = u_all[sel]; v_w = v_all[sel]
        rmse_u = float(np.sqrt(np.mean(u_w**2))); rmse_v = float(np.sqrt(np.mean(v_w**2)))
        print(f"  Cylinder wall ring (r≈R={args.radius})    "
              f"n={sel.size:5d}  rmse(u)={rmse_u:.2e}  rmse(v)={rmse_v:.2e}  "
              f"max|u|={float(np.max(np.abs(u_w))):.2e}  max|v|={float(np.max(np.abs(v_w))):.2e}")
        bcs_summary["cylinder_wall_ring"] = {"n": int(sel.size),
                                              "rmse_u": rmse_u, "rmse_v": rmse_v}
    summary["Q3"] = bcs_summary

    # -----------------------------------------------------------------
    section("Q4. Steadiness & symmetry (Re=40 should be both)")
    # -----------------------------------------------------------------
    if has_U and has_Umean:
        diff = fields["U"] - fields["UMean"]
        rmse_diff = float(np.sqrt(np.mean(diff ** 2)))
        max_diff = float(np.max(np.abs(diff)))
        print(f"  ||U − UMean||  rmse={rmse_diff:.4e}   max={max_diff:.4e}")
        if rmse_diff > 1e-6:
            print("  ** U is NOT identical to UMean: snapshot 'U' carries phase / fluctuation. **")
            print("     If you train on 'U' but solve steady NS in PINN, you are forcing the")
            print("     network to fit a non-steady target with a steady model — irreducible.")
        summary.setdefault("Q4", {})["U_vs_Umean"] = {"rmse": rmse_diff, "max": max_diff}
    if has_Up2:
        up2 = fields["UPrime2Mean"]
        rms_up2 = float(np.sqrt(np.mean(up2 ** 2)))
        max_up2 = float(np.max(np.abs(up2)))
        print(f"  UPrime2Mean    rmse={rms_up2:.4e}   max={max_up2:.4e}")
        if rms_up2 > 1e-4:
            print("  ** Non-zero turbulent stresses → CFD has unsteady fluctuation content. **")
        summary.setdefault("Q4", {})["UPrime2Mean"] = {"rmse": rms_up2, "max": max_up2}

    sym = symmetry_check(fields["x"], fields["y"], u_all, v_all, xlim, ylim,
                         nx=200, ny=200)
    print(f"  u(x,y) − u(x,−y)   rmse={sym['u_symmetry_rmse']:.4e}   max={sym['u_symmetry_maxabs']:.4e}    (u_max={sym['u_max']:.3e})")
    print(f"  v(x,y) + v(x,−y)   rmse={sym['v_antisym_rmse']:.4e}    max={sym['v_antisym_maxabs']:.4e}    (v_max={sym['v_max']:.3e})")
    if sym["v_antisym_rmse"] / max(sym["v_max"], 1e-12) > 0.1:
        print("  ** Symmetry significantly broken — Re may be above ~47 (vortex shedding) "
              "or domain off-center. **")
    summary.setdefault("Q4", {})["symmetry"] = sym

    # -----------------------------------------------------------------
    section("Q5. Does the CFD field SATISFY PINN's NS @ Re={:.1f}?".format(args.Re))
    # -----------------------------------------------------------------
    p_use = None
    p_tag = "none"
    if has_pMean:
        p_use = fields["pMean"].astype(np.float64); p_tag = "pMean"
    elif has_p:
        p_use = fields["p"].astype(np.float64); p_tag = "p"
    print(f"  using pressure source: {p_tag}")

    res = cfd_ns_residual(fields["x"], fields["y"], u_all, v_all, p_use,
                          Re=args.Re, xlim=xlim, ylim=ylim,
                          nx=args.nx, ny=args.ny,
                          mask_radius=args.radius + 0.1)

    bad = res["bad"]
    s_div = stats(res["div"], mask=bad)
    s_fu  = stats(res["f_u"], mask=bad)
    s_fv  = stats(res["f_v"], mask=bad)
    print(f"  ∇·u (continuity)        rmse={s_div['rmse']:.4e}   p99|.|={s_div['p99_abs']:.4e}   max={s_div['max_abs']:.4e}")
    print(f"  x-momentum residual     rmse={s_fu['rmse']:.4e}   p99|.|={s_fu['p99_abs']:.4e}   max={s_fu['max_abs']:.4e}")
    print(f"  y-momentum residual     rmse={s_fv['rmse']:.4e}   p99|.|={s_fv['p99_abs']:.4e}   max={s_fv['max_abs']:.4e}")
    if p_tag == "none":
        print("  (pressure unavailable: residuals INCLUDE the missing ∇p term, so they")
        print("   look ~|∇p|, not the true PDE residual.)")

    # near-cyl vs far stats
    R_near = 1.5
    near = ((res["X"] ** 2 + res["Y"] ** 2) < R_near ** 2) & ~bad
    far  = ~near & ~bad
    s_fu_near = stats(res["f_u"], mask=~near | bad)
    s_fu_far  = stats(res["f_u"], mask=~far  | bad)
    print(f"  x-momentum   near-cyl(r<{R_near})  rmse={s_fu_near['rmse']:.4e}")
    print(f"               far                   rmse={s_fu_far['rmse']:.4e}")

    summary["Q5"] = {
        "pressure_source": p_tag,
        "Re_used": args.Re,
        "div":  s_div, "f_u": s_fu, "f_v": s_fv,
        "f_u_near_cyl": s_fu_near, "f_u_far": s_fu_far,
    }

    # diagnostic plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for ax, F, title in [
        (axs[0, 0], np.where(bad, np.nan, res["U"]),    f"CFD u  ({src_tag})"),
        (axs[0, 1], np.where(bad, np.nan, res["V"]),    f"CFD v  ({src_tag})"),
        (axs[1, 0], np.where(bad, np.nan, res["div"]),  "∇·u (continuity)"),
        (axs[1, 1], np.where(bad, np.nan, res["f_u"]),  "x-momentum residual"),
    ]:
        cmap = "coolwarm"
        vmax = float(np.nanpercentile(np.abs(F), 99))
        im = ax.pcolormesh(res["X"], res["Y"], F, cmap=cmap, vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_aspect("equal"); ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "ns_residual_maps.png")
    plt.savefig(fig_path, dpi=140); plt.close(fig)
    print(f"  -> wrote {fig_path}")

    # symmetry plot
    fig = plt.figure(figsize=(7, 4))
    plt.scatter(fields["x"], fields["y"], c=v_all, s=2, cmap="coolwarm",
                vmin=-np.percentile(np.abs(v_all), 99), vmax=np.percentile(np.abs(v_all), 99))
    plt.colorbar(label="v")
    th = np.linspace(0, 2*np.pi, 200)
    plt.plot(args.radius*np.cos(th), args.radius*np.sin(th), "k-", lw=1)
    plt.gca().set_aspect("equal"); plt.title("v(x,y) — should be antisymmetric across y=0")
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "v_field_for_symmetry.png")
    plt.savefig(fig_path, dpi=140); plt.close(fig)
    print(f"  -> wrote {fig_path}")

    # -----------------------------------------------------------------
    section("VERDICT")
    # -----------------------------------------------------------------
    issues = []
    if has_U and has_Umean and summary["Q4"]["U_vs_Umean"]["rmse"] > 1e-6:
        issues.append("CFD has 'U' (instantaneous) ≠ 'UMean' (time-averaged). "
                      "If cfp40 is reading 'U', a steady PINN will fight the snapshot's phase. "
                      "USE 'UMean' INSTEAD.")
    if has_Up2 and summary["Q4"]["UPrime2Mean"]["rmse"] > 1e-4:
        issues.append("UPrime2Mean is non-zero → CFD captured unsteady fluctuations. "
                      "Re might be above the steady regime, or this is a snapshot, not the steady solution.")
    if summary["Q4"]["symmetry"]["v_antisym_rmse"] / max(summary["Q4"]["symmetry"]["v_max"], 1e-12) > 0.1:
        issues.append("Top-bottom symmetry is significantly broken at Re=40.")
    if summary["Q5"]["f_u"]["rmse"] > 1.0:
        issues.append(f"NS x-momentum residual on CFD field is {summary['Q5']['f_u']['rmse']:.2e} — "
                      "the CFD field, run through the PINN's PDE definition, does NOT satisfy NS. "
                      "Likely cause: wrong Re, wrong velocity array (instantaneous vs mean), "
                      "or wrong pressure normalization.")
    if not issues:
        print("  No major issues detected. Setup looks consistent with CFD.")
    else:
        for i, msg in enumerate(issues, 1):
            print(f"  [{i}] {msg}")

    # save json
    out_json = os.path.join(args.out_dir, "summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"\nFull summary -> {out_json}")


if __name__ == "__main__":
    main()
