"""
cfp40_v2.py
===========

A single, opinionated rebuild of the Re=40 PINN setup that takes mentor's two
points seriously:

  (1) "Use variable / adaptive weights" — Adam stage runs with a small
      AdaptiveLossWeights manager; after Adam we *auto-normalize* every
      surviving loss term by its EMA, so the BFGS objective is essentially
      unit-magnitude across all terms. SSBroyden's quasi-Hessian then handles
      the rest of the conditioning, which is the operational meaning of
      "second-order self-preconditions, you don't need to hand-tune".

  (2) The actual research goal is "make a PINN that REPRODUCES the cylinder
      flow field," not "fit the CFD point cloud." So the trade-off between
      data and physics is the wrong axis to optimize on. Instead, we remove
      as many of those Pareto trade-offs as possible by HARD-ENCODING the
      structure of the problem into the network itself:

        * stream-function output (∇·u = 0 by construction)         [kept]
        * no-slip on the cylinder via wall lifting                  [kept]
        * inlet / top / bottom free-stream u=1, v=0 via outer
          lifting + a y-baseline                                    [NEW]
        * spatial Fourier features at the input to fight spectral
          bias on the wake / shear layer                            [NEW]

      What survives in the loss is then:

          L = w_pde * L_pde + w_out * L_outlet [+ w_data * L_data, optional]

      i.e. essentially "satisfy NS in the interior" + "outlet pressure /
      out-flow gauge". inlet / top / bottom / wall losses ARE STILL
      computed for monitoring but their weights are zero — they should
      register as ~1e-6 to 1e-8 by construction.

This file is intentionally a thin wrapper. It re-uses sampling, dataset,
PDE residual, run_scipy_bfgs, and visualization from cfp40.py. The only
genuinely new objects are:

    MLPStreamPressureHardBC    new model
    AdaptiveLossWeightsV2      same idea, fewer keys
    compute_total_loss_v2      drops inlet / top_bottom
    auto_normalize_from_ema    mentor's "二阶法自动 precondition"
    run_adam_v2 / train_re40_single_v2 / main

Run:
    python cfp40_v2.py \
        --vtk-path Re40.vtk \
        --save-dir checkpoints_v2 \
        --viz-dir  viz_v2 \
        --epochs-adam 4000 --maxiter-bfgs 12000

Add --use-data to also include CFD points as a soft anchor (default off).
"""

from __future__ import annotations

import os
import sys
import math
import argparse

import numpy as np

import torch
import torch.nn as nn

# We re-use most of cfp40 verbatim. The only bits we override are the model,
# the loss composition, and the VTK loader (we want UMean/pMean by default,
# not the snapshot 'U' that cfp40.load_single_vtk picks).
from cfp40 import (
    set_seed,
    SingleSnapshotDataset,
    Snapshot,
    grad,
    model_uvp,
    model_vorticity,
    compute_pde_loss,
    compute_pde_residuals,
    sample_inlet,
    sample_top_bottom,
    sample_outlet,
    sample_cylinder_wall_uniform,
    compute_data_loss,
    run_scipy_bfgs,
    safe_load_checkpoint,
    atomic_torch_save,
    save_training_state,
    evaluate_against_cfd,
    evaluate_on_grid,
)


# ==========================================================
# VTK loader — defaults to UMean / pMean (steady time-average)
# ==========================================================

def _pick_array(arr_names, candidates):
    lower = {n.lower(): n for n in arr_names}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_single_vtk_v2(vtk_path: str, t_value: float = 0.0,
                       prefer_mean: bool = True) -> Snapshot:
    """
    Load a snapshot from VTK and pick the time-AVERAGE velocity field by
    default (UMean), not the instantaneous one (U).

    sanity_check_re40.py confirmed that for Re40.vtk, ||U − UMean||_rmse ≈ 9e-4
    and there are non-zero turbulent stresses. A steady PINN must NOT be
    trained against the snapshot 'U' — it would be fitting phase / fluctuation
    that the steady NS model cannot represent.

    Set prefer_mean=False to fall back to instantaneous 'U' (only do this if
    you know the case is laminar steady and the VTK doesn't carry a UMean).
    """
    import pyvista as pv  # imported here so cfp40_v2 can be imported even
                          # without pyvista (e.g. in unit tests)
    if not os.path.exists(vtk_path):
        raise FileNotFoundError(f"VTK file not found: {vtk_path}")
    mesh = pv.read(vtk_path)
    if mesh.n_cells > 0:
        centers = mesh.cell_centers().points
        src = mesh.cell_data
    else:
        centers = mesh.points
        src = mesh.point_data
    keys = list(src.keys())

    if prefer_mean:
        candidates = ["UMean", "Umean", "U_mean", "Uavg",
                      "U", "u", "velocity", "Velocity", "vel"]
    else:
        candidates = ["U", "u", "velocity", "Velocity", "vel",
                      "UMean", "Umean"]

    name = _pick_array(keys, candidates)
    if name is None:
        raise KeyError(f"No usable velocity array in {vtk_path}. Available: {keys}")
    vel = np.asarray(src[name])
    if vel.ndim != 2 or vel.shape[1] < 2:
        raise ValueError(f"Velocity array {name} has unexpected shape {vel.shape}")

    x = centers[:, 0].astype(np.float32)
    y = centers[:, 1].astype(np.float32)
    u = vel[:, 0].astype(np.float32)
    v = vel[:, 1].astype(np.float32)

    print(f"[v2-loader] {os.path.basename(vtk_path)}  cells={len(x)}  "
          f"velocity_array='{name}'  prefer_mean={prefer_mean}")
    return Snapshot(t=float(t_value), x=x, y=y, u=u, v=v)


# ==========================================================
# Model — hard BCs + Fourier features
# ==========================================================

class MLPStreamPressureHardBC(nn.Module):
    """
    (x, y, t) -> (psi, p)
    with hard-encoded BCs:

        wall (cylinder)        : u = v = 0 via wall_lift(r)
        inlet  (x = x_min)     : u = 1, v = 0 via outer_lift_x and y-baseline
        top    (y = y_max)     : u = 1, v = 0 via outer_lift_y_top
        bottom (y = y_min)     : u = 1, v = 0 via outer_lift_y_bot
        outlet (x = x_max)     : LEFT FREE — handled in loss as p=0 + ∂u/∂x=0

    Construction:

        psi(x,y,t) = wall_lift(r) * [ y + outer_lift(x,y) * NN_psi(γ(x,y),t) ]
        p  (x,y,t) =                                       NN_p  (γ(x,y),t)

    where γ is a Fourier-feature embedding of (x, y).

    Why y is the baseline:
        u_∞ = ∂ψ/∂y, v_∞ = -∂ψ/∂x. Setting ψ = y in the far field
        immediately gives u = 1, v = 0 — i.e. uniform free stream.

    Why tanh^2 lifts:
        tanh²(s) = 0 AND d/ds tanh²(s) = 0 at s = 0. So at the boundary
        BOTH the function and its normal derivative vanish, which means
        u and v at the boundary equal exactly the baseline values (1, 0)
        regardless of what the network outputs.

    The pressure gauge is fixed at the outlet via the outlet loss term
    (p=0 there). We do NOT lift p, since pressure has no physical BC of
    its own except this single anchor.
    """

    def __init__(self,
                 width: int = 128,
                 depth: int = 6,
                 t_ref: float = 0.0,
                 t_scale: float = 1.0,
                 cylinder_center=(0.0, 0.0),
                 radius: float = 0.5,
                 lift_delta: float = 0.15,
                 # outer lifting deltas: how thick the BC layer is
                 outer_delta_x: float = 0.5,
                 outer_delta_y: float = 0.5,
                 # domain limits used by the lift functions
                 x_min: float = -3.0,
                 y_min: float = -4.0,
                 y_max: float = 4.0,
                 # Fourier features
                 fourier_features: int = 32,
                 fourier_sigma: float = 2.0,
                 # use deterministic feature draw so resume-from-ckpt is reproducible
                 fourier_seed: int = 0):
        super().__init__()

        self.t_ref = float(t_ref)
        self.t_scale = float(t_scale)
        self.xc = float(cylinder_center[0])
        self.yc = float(cylinder_center[1])
        self.radius = float(radius)
        self.lift_delta = float(lift_delta)
        self.outer_delta_x = float(outer_delta_x)
        self.outer_delta_y = float(outer_delta_y)
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.fourier_features = int(fourier_features)

        # Fourier feature matrix B ∈ R^{F x 2}, fixed (not learnable).
        # gamma(x,y) = [sin(2π B [x;y]), cos(2π B [x;y])]
        # Storing as buffer so it travels with the checkpoint.
        g = torch.Generator().manual_seed(int(fourier_seed))
        B = torch.randn(self.fourier_features, 2, generator=g) * float(fourier_sigma)
        self.register_buffer("B", B)

        # Input dim = 2*F (Fourier xy) + 3 (t, sin t, cos t)
        in_dim = 2 * self.fourier_features + 3

        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    # ---- embeddings ----------------------------------------------------

    def fourier_embed(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # xy: (N, 2). proj = xy @ B^T : (N, F).
        xy = torch.cat([x, y], dim=1)
        proj = 2.0 * math.pi * (xy @ self.B.t())
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)

    def time_embed(self, t: torch.Tensor) -> torch.Tensor:
        t_norm = (t - self.t_ref) / self.t_scale
        return torch.cat([
            t_norm,
            torch.sin(2.0 * math.pi * t_norm),
            torch.cos(2.0 * math.pi * t_norm),
        ], dim=1)

    # ---- BC lift functions --------------------------------------------

    def wall_lift(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt((x - self.xc) ** 2 + (y - self.yc) ** 2 + 1e-12)
        s = (r - self.radius) / self.lift_delta
        return torch.tanh(s) ** 2

    def outer_lift(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Vanishes (with zero normal derivative) on inlet, top and bottom
        boundaries; goes smoothly to 1 in the interior.

        Outlet is left free.
        """
        s_in = (x - self.x_min) / self.outer_delta_x
        s_top = (self.y_max - y) / self.outer_delta_y
        s_bot = (y - self.y_min) / self.outer_delta_y
        return (torch.tanh(s_in) ** 2) * (torch.tanh(s_top) ** 2) * (torch.tanh(s_bot) ** 2)

    # ---- forward ------------------------------------------------------

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        inp = torch.cat([self.fourier_embed(x, y), self.time_embed(t)], dim=1)
        raw = self.net(inp)

        psi_raw = raw[:, 0:1]
        p = raw[:, 1:2]

        Lw = self.wall_lift(x, y)
        Lo = self.outer_lift(x, y)

        # ψ = wall_lift(r) · [y + outer_lift(x,y) · psi_raw]
        psi = Lw * (y + Lo * psi_raw)

        return torch.cat([psi, p], dim=1)


# ==========================================================
# Adaptive weights — only the surviving terms
# ==========================================================

V2_LOSS_KEYS = ("pde", "outlet", "wall_uv", "wall_psi", "inlet", "top_bottom", "data")
# inlet / top_bottom / wall_* are kept in the dict so existing run_scipy_bfgs
# (which references them by name) doesn't break, but their default weights are
# 0 → they cost nothing.


class AdaptiveLossWeightsV2:
    """
    Same EMA-based scheme as AdaptiveLossWeights in cfp40, but only over the
    losses that actually carry weight (pde, outlet, [data]). The hard-encoded
    losses (inlet, top_bottom, wall_uv, wall_psi) are tracked for monitoring
    only — their weights stay pinned at 0.
    """

    ACTIVE = ("pde", "outlet", "data")

    def __init__(self, alpha: float = 0.9, temp: float = 0.5, eps: float = 1e-12,
                 use_data: bool = False):
        self.alpha = float(alpha)
        self.temp = float(temp)
        self.eps = float(eps)
        self.use_data = bool(use_data)
        self.ema: dict = {}
        # Bias: PDE matters most (it's the actual physics we're solving for).
        # Outlet is just a 1D anchor, so a small base. Data, if used, is a soft
        # nudge.
        self.base = {
            "pde": 1.0,
            "outlet": 0.2,
            "data": 0.5 if self.use_data else 0.0,
        }
        self.current = default_fixed_weights_v2(use_data=self.use_data)

    def update(self, raw_losses: dict):
        # update EMA over ALL tracked terms (so we can later inspect
        # hard-encoded ones)
        for k, v in raw_losses.items():
            val = float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
            if k not in self.ema:
                self.ema[k] = val
            else:
                self.ema[k] = self.alpha * self.ema[k] + (1.0 - self.alpha) * val

        # rebalance ONLY the active terms
        active_vals = np.array([self.ema.get(k, 0.0) for k in self.ACTIVE if self.base[k] > 0.0],
                               dtype=np.float64)
        mean_ema = float(np.mean(active_vals) + self.eps) if active_vals.size > 0 else 1.0

        new_weights = default_fixed_weights_v2(use_data=self.use_data)
        for k in self.ACTIVE:
            b = self.base[k]
            if b == 0.0:
                continue
            ratio = (self.ema.get(k, mean_ema) + self.eps) / mean_ema
            w = b * (ratio ** self.temp)
            new_weights[k] = float(np.clip(w, 0.2 * b, 5.0 * b))
        self.current = new_weights

    def get(self) -> dict:
        return dict(self.current)

    def get_ema(self) -> dict:
        return dict(self.ema)


def default_fixed_weights_v2(use_data: bool = False) -> dict:
    """
    Defaults for the *hard-BC* setup. Anything that's hard-encoded gets weight 0.

    We keep all keys that run_scipy_bfgs expects so we don't have to fork it.
    """
    return {
        "pde": 1.0,
        "outlet": 0.2,
        "wall_uv": 0.0,
        "wall_psi": 0.0,
        "inlet": 0.0,
        "top_bottom": 0.0,
        "data": 0.5 if use_data else 0.0,
    }


# ==========================================================
# Loss composition (Adam stage)
# ==========================================================

def compute_bc_losses_for_monitoring(model, device, t_value, xlim, ylim,
                                     n_inlet=512, n_tb=512, n_wall=1024, n_outlet=1024):
    """
    Lighter-weight BC loss evaluator for monitoring only. We use it to:
      - confirm hard-encoded BCs are numerically zero
      - keep the outlet loss as an active term (non-zero weight)
    """
    xi, yi, ti = sample_inlet(n_inlet, device=device, t_value=t_value, x0=xlim[0], ylim=ylim)
    _, u_i, v_i, _ = model_uvp(model, xi, yi, ti)
    loss_inlet = ((u_i - 1.0) ** 2).mean() + (v_i ** 2).mean()

    xtb, ytb, ttb = sample_top_bottom(n_tb, device=device, t_value=t_value, xlim=xlim, yabs=ylim[1])
    _, u_tb, v_tb, _ = model_uvp(model, xtb, ytb, ttb)
    loss_tb = ((u_tb - 1.0) ** 2).mean() + (v_tb ** 2).mean()

    xw, yw, tw = sample_cylinder_wall_uniform(n_wall, device=device, t_value=t_value)
    psi_w, u_w, v_w, _ = model_uvp(model, xw, yw, tw)
    loss_wall_uv = (u_w ** 2).mean() + (v_w ** 2).mean()
    loss_wall_psi = (psi_w ** 2).mean()

    xo, yo, to = sample_outlet(n_outlet, device=device, t_value=t_value, x0=xlim[1], ylim=ylim)
    _, u_o, v_o, p_o = model_uvp(model, xo, yo, to)
    u_x_o = grad(u_o, xo)
    v_x_o = grad(v_o, xo)
    loss_out = (p_o ** 2).mean() + (u_x_o ** 2).mean() + (v_x_o ** 2).mean()

    return {
        "inlet": loss_inlet,
        "top_bottom": loss_tb,
        "wall_uv": loss_wall_uv,
        "wall_psi": loss_wall_psi,
        "outlet": loss_out,
    }


def sample_collocation_v2(n_f, device, t_value, xlim, ylim,
                          radius=0.5, near_cyl_frac=0.5, near_cyl_radius=2.5):
    """
    Adam-stage collocation sampler that biases points toward the wake / near
    cylinder. Better than uniform — concentration where the residual is hardest.
    """
    n_near = int(n_f * near_cyl_frac)
    n_far = n_f - n_near

    # uniform far field
    x_f = torch.empty((n_far, 1), device=device).uniform_(xlim[0], xlim[1])
    y_f = torch.empty((n_far, 1), device=device).uniform_(ylim[0], ylim[1])

    # near-cylinder annulus (rejection sample to stay outside cylinder)
    n_keep = 0
    parts_x, parts_y = [], []
    while n_keep < n_near:
        m = max(n_near * 2, 16)
        cx = torch.empty((m, 1), device=device).uniform_(-near_cyl_radius, near_cyl_radius)
        cy = torch.empty((m, 1), device=device).uniform_(-near_cyl_radius, near_cyl_radius)
        r = torch.sqrt(cx ** 2 + cy ** 2)
        ok = (r > radius * 1.02) & (r < near_cyl_radius)
        cx = cx[ok].view(-1, 1)
        cy = cy[ok].view(-1, 1)
        take = min(int(cx.shape[0]), n_near - n_keep)
        if take > 0:
            parts_x.append(cx[:take])
            parts_y.append(cy[:take])
            n_keep += take
    if parts_x:
        x_n = torch.cat(parts_x, dim=0)
        y_n = torch.cat(parts_y, dim=0)
        x_f = torch.cat([x_f, x_n], dim=0)
        y_f = torch.cat([y_f, y_n], dim=0)

    # exclude any point that snuck inside the cylinder
    keep = (x_f ** 2 + y_f ** 2) > (radius ** 2)
    x_f = x_f[keep].view(-1, 1)
    y_f = y_f[keep].view(-1, 1)
    t_f = torch.full_like(x_f, float(t_value))
    return x_f, y_f, t_f


def compute_total_loss_v2(model, dataset, device, n_f, n_data, Re, t_value,
                          weight_manager: AdaptiveLossWeightsV2,
                          xlim, ylim,
                          use_data: bool):
    # PDE (interior collocation)
    x_f, y_f, t_f = sample_collocation_v2(n_f, device=device, t_value=t_value,
                                          xlim=xlim, ylim=ylim)
    loss_pde = compute_pde_loss(model, x_f, y_f, t_f, Re=Re)

    # BCs (hard-encoded; we evaluate them for monitoring + outlet anchor)
    bc = compute_bc_losses_for_monitoring(model, device, t_value, xlim, ylim)

    # Optional data loss (CFD anchor)
    if use_data and n_data > 0:
        t_d, x_d, y_d, u_d, v_d = dataset.sample(n_data, wake_frac=0.5, xlim=xlim, ylim=ylim)
        loss_data = compute_data_loss(model, t_d, x_d, y_d, u_d, v_d)
    else:
        loss_data = torch.zeros((), device=device)

    raw = {
        "pde": loss_pde,
        "outlet": bc["outlet"],
        "wall_uv": bc["wall_uv"],
        "wall_psi": bc["wall_psi"],
        "inlet": bc["inlet"],
        "top_bottom": bc["top_bottom"],
        "data": loss_data,
    }
    weight_manager.update(raw)
    w = weight_manager.get()

    total = (
        w["pde"] * loss_pde
        + w["outlet"] * bc["outlet"]
        + w["data"] * loss_data
        # hard-encoded terms have weight 0 by default; keep them in the formula
        # so it's structurally identical to BFGS objective (cleaner Hessian
        # warmup if you ever flip a weight non-zero)
        + w["wall_uv"] * bc["wall_uv"]
        + w["wall_psi"] * bc["wall_psi"]
        + w["inlet"] * bc["inlet"]
        + w["top_bottom"] * bc["top_bottom"]
    )

    logs = {
        "L": total,
        "pde": loss_pde,
        "outlet": bc["outlet"],
        "wall_uv": bc["wall_uv"],
        "wall_psi": bc["wall_psi"],
        "inlet": bc["inlet"],
        "top_bottom": bc["top_bottom"],
        "data": loss_data,
        "weights": w,
    }
    return total, logs


# ==========================================================
# Auto-normalize (mentor's "二阶法自动 precondition")
# ==========================================================

def auto_normalize_from_ema(weight_manager: AdaptiveLossWeightsV2,
                            use_data: bool,
                            eps: float = 1e-12) -> dict:
    """
    Convert the Adam-stage EMA of each loss term into a *fixed* weight set
    for BFGS, such that every active loss enters the BFGS objective at
    O(1) magnitude. SSBroyden's quasi-Hessian then handles the remaining
    parameter-space conditioning — the user's mentor's claim that "you
    don't need to tune weights with second-order" is operationally
    implemented here.

    Hard-encoded terms (inlet, top_bottom, wall_*) keep weight 0 — they
    are residuals, not objectives.
    """
    ema = weight_manager.get_ema()
    weights = default_fixed_weights_v2(use_data=use_data)
    for k in ("pde", "outlet", "data"):
        if weights[k] == 0.0:
            continue
        weights[k] = 1.0 / (ema.get(k, 1.0) + eps)
    return weights


# ==========================================================
# Adam stage (v2)
# ==========================================================

def run_adam_v2(model, dataset, device, save_dir,
                epochs_adam, n_f, n_data, lr_adam, Re, t_value,
                xlim, ylim, use_data: bool):
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
    weight_manager = AdaptiveLossWeightsV2(use_data=use_data)
    frozen_weights = weight_manager.get()

    for epoch in range(1, epochs_adam + 1):
        model.train()
        adam.zero_grad()
        loss, logs = compute_total_loss_v2(
            model=model, dataset=dataset, device=device,
            n_f=n_f, n_data=n_data, Re=Re, t_value=t_value,
            weight_manager=weight_manager,
            xlim=xlim, ylim=ylim, use_data=use_data,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        adam.step()

        cur = float(loss.detach().cpu().item())
        frozen_weights = logs["weights"]

        if cur < best_loss:
            best_loss = cur
            save_training_state(
                model, save_dir, epoch, best_loss, "adam_v2",
                extra_state={"frozen_weights": frozen_weights,
                             "ema": weight_manager.get_ema()},
            )
        else:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "stage": "adam_v2",
                    "frozen_weights": frozen_weights,
                    "ema": weight_manager.get_ema(),
                },
                os.path.join(save_dir, "pinn_latest.pt"),
            )

        if epoch % 20 == 0:
            w = logs["weights"]
            # Use .item() so requires_grad tensors are detached + scalarized
            # in one step (silences the autograd UserWarning).
            print(
                f"[Adam-v2] ep={epoch:5d}  total={cur:.3e}  "
                f"PDE={logs['pde'].item():.3e} (w={w['pde']:.3f})  "
                f"OUT={logs['outlet'].item():.3e} (w={w['outlet']:.3f})  "
                f"DATA={logs['data'].item():.3e} (w={w['data']:.3f})  "
                f"|  hard-BC residuals: "
                f"in={logs['inlet'].item():.2e} "
                f"tb={logs['top_bottom'].item():.2e} "
                f"wuv={logs['wall_uv'].item():.2e} "
                f"wpsi={logs['wall_psi'].item():.2e}"
            )

    return best_loss, frozen_weights, weight_manager


# ==========================================================
# Top-level training (v2)
# ==========================================================

def train_re40_single_v2(model: nn.Module,
                         snapshot: Snapshot,
                         device: torch.device,
                         save_dir: str,
                         epochs_adam: int = 4000,
                         maxiter_bfgs: int = 12000,
                         iters_per_batch: int = 200,
                         n_f: int = 50000,
                         n_data: int = 20000,
                         lr_adam: float = 1e-3,
                         gtol_bfgs: float = 0.0,
                         Re: float = 40.0,
                         t_value: float = 0.0,
                         xlim=(-3.0, 12.0),
                         ylim=(-4.0, 4.0),
                         method_bfgs: str = "SSBroyden1",
                         full_data_bfgs: bool = True,
                         frozen_colloc_bfgs: bool = True,
                         use_data: bool = False):
    os.makedirs(save_dir, exist_ok=True)
    dataset = SingleSnapshotDataset(snapshot, device=device)

    # ---------- 1) Adam stage with adaptive weights ----------
    print("[v2] === Stage 1: Adam with adaptive (variable) weights ===")
    best_adam, frozen_weights, wm = run_adam_v2(
        model=model, dataset=dataset, device=device, save_dir=save_dir,
        epochs_adam=epochs_adam, n_f=n_f, n_data=n_data, lr_adam=lr_adam,
        Re=Re, t_value=t_value, xlim=xlim, ylim=ylim, use_data=use_data,
    )

    # ---------- 2) Auto-normalize for BFGS ----------
    bfgs_weights = auto_normalize_from_ema(wm, use_data=use_data)
    print("[v2] === Stage 2: Auto-normalized BFGS weights (from Adam EMA) ===")
    for k, v in bfgs_weights.items():
        ema = wm.get_ema().get(k, float("nan"))
        print(f"        {k:12s} weight={v:.4e}    (Adam EMA={ema:.4e})")

    # ---------- 3) BFGS / SSBroyden ----------
    if maxiter_bfgs > 0:
        print(f"[v2] === Stage 3: SSBroyden ({method_bfgs}) ===")
        run_scipy_bfgs(
            model=model, dataset=dataset, device=device, save_dir=save_dir,
            start_epoch=epochs_adam, maxiter=maxiter_bfgs,
            n_f=n_f, n_data=n_data, Re=Re, t_value=t_value,
            fixed_weights=bfgs_weights,
            iters_per_batch=iters_per_batch, gtol=gtol_bfgs, disp=False,
            xlim=xlim, ylim=ylim, method_bfgs=method_bfgs,
            n_cfd_pde=0, lambda_pde_cfd=1.0,
            full_data_bfgs=full_data_bfgs and use_data,
            frozen_colloc_bfgs=frozen_colloc_bfgs,
            cfd_pde_wall_buffer=0.0, cfd_pde_edge_buffer=0.0,
            data_only=False,
            use_all_cfd_data=False,
        )

    # ---------- 4) Save canonical name ----------
    final_path = os.path.join(save_dir, "pinn_Re40_single.pt")
    if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
        # ensure we have a usable canonical checkpoint name
        latest = os.path.join(save_dir, "pinn_latest.pt")
        if os.path.exists(latest):
            import shutil
            shutil.copyfile(latest, final_path)
    print(f"[v2] training done. Saved -> {final_path}")


# ==========================================================
# Visualization helper that uses the v2 model
# ==========================================================

def load_model_for_viz_v2(save_dir, device, **model_kwargs):
    model = MLPStreamPressureHardBC(**model_kwargs).to(device)
    ckpt_main = os.path.join(save_dir, "pinn_Re40_single.pt")
    ckpt_fall = os.path.join(save_dir, "pinn_latest.pt")
    path = ckpt_main if (os.path.exists(ckpt_main) and os.path.getsize(ckpt_main) > 0) else ckpt_fall
    state = safe_load_checkpoint(path, device)
    if "model" not in state:
        raise KeyError(f"Checkpoint {path} has no 'model' key")
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[viz-v2] loaded {path}")
    return model


def visualize_v2(save_dir, device, out_dir, snapshot, xlim, ylim, **model_kwargs):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    model = load_model_for_viz_v2(save_dir, device, **model_kwargs)
    if snapshot is not None:
        evaluate_against_cfd(model, snapshot, device, xlim=xlim, ylim=ylim)
    X, Y, U, V, W = evaluate_on_grid(model, device=device, t_val=0.0, xlim=xlim, ylim=ylim)
    speed = np.sqrt(U ** 2 + V ** 2)
    for field, label, fname in [
        (W, "vorticity", "v2_vorticity.png"),
        (speed, "|u|", "v2_speed.png"),
        (U, "u", "v2_u.png"),
        (V, "v", "v2_v.png"),
    ]:
        fig = plt.figure(figsize=(7, 4))
        cmap = "coolwarm" if label in ("vorticity", "v") else "viridis"
        plt.pcolormesh(X, Y, field, shading="auto", cmap=cmap)
        plt.colorbar(label=label)
        plt.gca().set_aspect("equal")
        plt.title(f"PINN v2 — {label}")
        out_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"[viz-v2] {out_path}")


# ==========================================================
# CLI
# ==========================================================

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--vtk-path", type=str, required=True)
    p.add_argument("--save-dir", type=str, default="checkpoints_v2")
    p.add_argument("--viz-dir", type=str, default="viz_v2")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--epochs-adam", type=int, default=4000)
    p.add_argument("--maxiter-bfgs", type=int, default=12000)
    p.add_argument("--iters-per-batch", type=int, default=200)
    p.add_argument("--n-f", type=int, default=50000)
    p.add_argument("--n-data", type=int, default=20000)
    p.add_argument("--lr-adam", type=float, default=1e-3)
    p.add_argument("--gtol-bfgs", type=float, default=0.0)
    p.add_argument("--method-bfgs", type=str, default="SSBroyden1")
    p.add_argument("--use-data", action="store_true",
                   help="Include CFD points as a soft anchor. Default off — "
                        "the goal is data-free reconstruction.")
    # Domain defaults expanded to where CFD's free-stream BC actually holds.
    # sanity_check_re40.py confirmed:
    #   - CFD true inlet at x=-9.75: rmse(u-1)=2e-5, rmse(v)=5e-4   (clean)
    #   - PINN box at x=-3.0      : rmse(u-1)=3e-2, rmse(v)=4e-2    (off by 5%)
    # so hard-encoded u=1, v=0 is only physical near the larger CFD edges.
    p.add_argument("--x-min", type=float, default=-8.0)
    p.add_argument("--x-max", type=float, default=12.0)
    p.add_argument("--y-min", type=float, default=-8.0)
    p.add_argument("--y-max", type=float, default=8.0)
    p.add_argument("--prefer-mean", action="store_true", default=True,
                   help="Read 'UMean' from VTK (steady time-average). Default ON. "
                        "Use --use-instantaneous to read snapshot 'U' instead.")
    p.add_argument("--use-instantaneous", dest="prefer_mean", action="store_false",
                   help="Force reading the instantaneous 'U' velocity array. "
                        "Only do this for laminar steady cases without a UMean.")
    p.add_argument("--lift-delta", type=float, default=0.15)
    p.add_argument("--outer-delta-x", type=float, default=0.5)
    p.add_argument("--outer-delta-y", type=float, default=0.5)
    p.add_argument("--fourier-features", type=int, default=32)
    p.add_argument("--fourier-sigma", type=float, default=2.0)
    p.add_argument("--fourier-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--viz-only", action="store_true")
    p.add_argument("--resume-from", type=str, default=None)
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    snapshot = load_single_vtk_v2(args.vtk_path, t_value=0.0,
                                   prefer_mean=args.prefer_mean)
    xlim = (args.x_min, args.x_max)
    ylim = (args.y_min, args.y_max)

    model_kwargs = dict(
        width=args.width, depth=args.depth,
        lift_delta=args.lift_delta,
        outer_delta_x=args.outer_delta_x,
        outer_delta_y=args.outer_delta_y,
        x_min=args.x_min, y_min=args.y_min, y_max=args.y_max,
        fourier_features=args.fourier_features,
        fourier_sigma=args.fourier_sigma,
        fourier_seed=args.fourier_seed,
    )

    if not args.viz_only:
        model = MLPStreamPressureHardBC(**model_kwargs).to(device)
        n_params = sum(pr.numel() for pr in model.parameters())
        print(f"[v2] Model: width={args.width}, depth={args.depth}, "
              f"Fourier F={args.fourier_features} σ={args.fourier_sigma}, "
              f"params={n_params}")

        if args.resume_from is not None:
            if not os.path.exists(args.resume_from):
                raise FileNotFoundError(f"--resume-from path does not exist: {args.resume_from}")
            state = safe_load_checkpoint(args.resume_from, device)
            if "model" not in state:
                raise KeyError(f"checkpoint {args.resume_from} has no 'model' key")
            model.load_state_dict(state["model"], strict=False)
            print(f"[v2] resumed from {args.resume_from}")

        train_re40_single_v2(
            model=model, snapshot=snapshot, device=device,
            save_dir=args.save_dir,
            epochs_adam=args.epochs_adam,
            maxiter_bfgs=args.maxiter_bfgs,
            iters_per_batch=args.iters_per_batch,
            n_f=args.n_f, n_data=args.n_data,
            lr_adam=args.lr_adam, gtol_bfgs=args.gtol_bfgs,
            Re=40.0, t_value=0.0,
            xlim=xlim, ylim=ylim,
            method_bfgs=args.method_bfgs,
            use_data=args.use_data,
        )

    visualize_v2(
        save_dir=args.save_dir, device=device, out_dir=args.viz_dir,
        snapshot=snapshot, xlim=xlim, ylim=ylim, **model_kwargs,
    )


if __name__ == "__main__":
    main()
