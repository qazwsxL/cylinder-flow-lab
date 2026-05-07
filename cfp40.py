
import os
import sys
import math
import argparse
import warnings
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from scipy.optimize import minimize
from scipy.linalg import cholesky, LinAlgError

try:
    import pyvista as pv
except Exception:
    pv = None


# ----------------------------------------------------------------------
# Custom BFGS (SSBroyden / SSBFGS variants) — must be loaded explicitly,
# scipy's stock _minimize_bfgs does NOT accept the `method_bfgs` option
# and will silently fall back to standard BFGS if you only pass
# method="BFGS". Mirrors the pattern used in cfp.py.
# ----------------------------------------------------------------------
_OPT_DIR_CANDIDATES = [
    os.path.dirname(os.path.abspath(__file__)),
    "/oscar/home/jchen790/cylinder flow lab",
]
_CUSTOM_MIN_BFGS = None
for _d in _OPT_DIR_CANDIDATES:
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)
try:
    from _optimize import _minimize_bfgs as _CUSTOM_MIN_BFGS  # type: ignore
    print("[OK] _optimize._minimize_bfgs loaded — SSBroyden / SSBFGS variants available")
except Exception as _e:
    warnings.warn(
        f"_optimize._minimize_bfgs NOT loaded ({_e}). "
        "BFGS will silently fall back to scipy's standard BFGS — "
        "SSBroyden1 / SSBroyden2 / SSBFGS_OL / SSBFGS_AB will be IGNORED.",
        RuntimeWarning,
    )
    _CUSTOM_MIN_BFGS = None


# ==========================================================
# Utilities
# ==========================================================

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def safe_load_checkpoint(path: str, device: torch.device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if os.path.getsize(path) == 0:
        raise RuntimeError(f"Checkpoint is empty: {path}")
    return torch.load(path, map_location=device)


# ==========================================================
# Model
# ==========================================================

class MLPStreamPressure(nn.Module):
    """
    (x, y, t) -> (psi, p)

    Main change from the original version:
    - hard no-slip enforcement on the cylinder wall through stream-function lifting
          psi = lift(r) * psi_raw
      with lift(R)=0 and d lift / d r (R)=0.
    - Therefore u = dpsi/dy and v = -dpsi/dx are exactly zero on the wall.
    """

    def __init__(self, width: int = 48, depth: int = 3,
                 t_ref: float = 0.0, t_scale: float = 1.0,
                 cylinder_center=(0.0, 0.0), radius: float = 0.5,
                 lift_delta: float = 0.15):
        super().__init__()
        self.t_ref = t_ref
        self.t_scale = t_scale
        self.xc = float(cylinder_center[0])
        self.yc = float(cylinder_center[1])
        self.radius = float(radius)
        self.lift_delta = float(lift_delta)

        layers = []
        in_dim = 5
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def time_embed(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_norm = (t - self.t_ref) / self.t_scale
        return torch.cat([
            x,
            y,
            t_norm,
            torch.sin(2.0 * math.pi * t_norm),
            torch.cos(2.0 * math.pi * t_norm),
        ], dim=1)

    def wall_lift(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r = torch.sqrt((x - self.xc) ** 2 + (y - self.yc) ** 2 + 1e-12)
        s = (r - self.radius) / self.lift_delta
        return torch.tanh(s) ** 2

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        inp = self.time_embed(x, y, t)
        raw = self.net(inp)

        psi_raw = raw[:, 0:1]
        p = raw[:, 1:2]
        psi = self.wall_lift(x, y) * psi_raw

        return torch.cat([psi, p], dim=1)


# ==========================================================
# Adaptive weights
# ==========================================================

class AdaptiveLossWeights:
    """
    Adaptive weighting for Adam stage.

    Maintains EMA of each raw loss. Losses that remain relatively large get
    increased weight; losses that are already small get reduced weight.
    We still keep a prior preference that wall and data matter more.
    """

    def __init__(self, alpha: float = 0.9, temp: float = 0.5, eps: float = 1e-8):
        self.alpha = float(alpha)
        self.temp = float(temp)
        self.eps = float(eps)
        self.ema = {}
        self.base = {
            "pde": 1.5,
            "inlet": 1.0,
            "top_bottom": 1.0,
            "wall_uv": 1.0,
            "wall_psi": 0.2,
            "outlet": 1.0,
            "data": 4.0,
        }
        self.current = self.base.copy()

    def update(self, raw_losses: dict):
        for k, v in raw_losses.items():
            if k not in self.base:
                continue
            val = float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
            if k not in self.ema:
                self.ema[k] = val
            else:
                self.ema[k] = self.alpha * self.ema[k] + (1.0 - self.alpha) * val

        ema_vals = np.array([self.ema[k] for k in self.base.keys()], dtype=np.float64)
        mean_ema = float(np.mean(ema_vals) + self.eps)

        new_weights = {}
        for k, b in self.base.items():
            ratio = (self.ema[k] + self.eps) / mean_ema
            w = b * (ratio ** self.temp)
            new_weights[k] = float(np.clip(w, 0.2 * b, 5.0 * b))
        self.current = new_weights

    def get(self):
        return self.current.copy()


# ==========================================================
# Autograd helpers
# ==========================================================

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def model_uvp(model: nn.Module, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    out = model(torch.cat([x, y, t], dim=1))
    psi = out[:, 0:1]
    p = out[:, 1:2]
    u = grad(psi, y)
    v = -grad(psi, x)
    return psi, u, v, p


def model_vorticity(model: nn.Module, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor):
    _, u, v, _ = model_uvp(model, x, y, t)
    u_y = grad(u, y)
    v_x = grad(v, x)
    return v_x - u_y


# ==========================================================
# Single VTK data
# ==========================================================

@dataclass
class Snapshot:
    t: float
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray


def _pick_array_name(names, candidates):
    lower_map = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_single_vtk(vtk_path: str, t_value: float = 0.0) -> Snapshot:
    if pv is None:
        raise ImportError("pyvista is required to read VTK files. pip install pyvista")
    if not os.path.exists(vtk_path):
        raise FileNotFoundError(f"VTK file not found: {vtk_path}")

    mesh = pv.read(vtk_path)

    if mesh.n_cells > 0:
        centers = mesh.cell_centers().points
        data_src = mesh.cell_data
    else:
        centers = mesh.points
        data_src = mesh.point_data

    array_names = list(data_src.keys())
    vel_name = _pick_array_name(array_names, ["U", "u", "velocity", "Velocity", "vel"])
    if vel_name is None:
        raise KeyError(f"Could not find velocity array in {vtk_path}. Available: {array_names}")

    vel = np.asarray(data_src[vel_name])
    if vel.ndim != 2 or vel.shape[1] < 2:
        raise ValueError(f"Velocity array {vel_name} has unexpected shape {vel.shape}")

    x = centers[:, 0].astype(np.float32)
    y = centers[:, 1].astype(np.float32)
    u = vel[:, 0].astype(np.float32)
    v = vel[:, 1].astype(np.float32)

    print(f"Loaded {os.path.basename(vtk_path)}  cells={len(x)}")
    return Snapshot(t=float(t_value), x=x, y=y, u=u, v=v)


# ==========================================================
# Dataset
# ==========================================================

class SingleSnapshotDataset:
    def __init__(self, snapshot: Snapshot, device: torch.device):
        self.snapshot = snapshot
        self.device = device

    def _domain_candidate_ids(self, xlim=None, ylim=None, radius: float = 0.5,
                               wall_buffer: float = 0.0, edge_buffer: float = 0.0):
        """
        Return CFD fluid-cell ids inside the current PDE box.

        wall_buffer  : exclude CFD points whose distance to the cylinder is < this.
        edge_buffer  : exclude CFD points within this distance of the box boundary.

        Use a positive buffer when sampling CFD points for the PDE residual
        loss: the boundary-layer cells in coarse CFD are NOT a good PDE
        solution (diagnose_cfd_pde.py confirms residuals of 1~10 there) and
        forcing the PINN to satisfy the PDE on those points fights the data
        loss and stalls training.
        """
        s = self.snapshot
        x = s.x
        y = s.y
        domain_mask = np.ones_like(x, dtype=bool)
        if xlim is not None:
            x0 = float(xlim[0]) + edge_buffer
            x1 = float(xlim[1]) - edge_buffer
            domain_mask &= (x >= x0) & (x <= x1)
        if ylim is not None:
            y0 = float(ylim[0]) + edge_buffer
            y1 = float(ylim[1]) - edge_buffer
            domain_mask &= (y >= y0) & (y <= y1)
        eff_R = float(radius) + float(wall_buffer)
        domain_mask &= ((x ** 2 + y ** 2) >= eff_R ** 2)
        ids = np.where(domain_mask)[0]
        if len(ids) == 0:
            raise ValueError(
                f"No CFD fluid points remain inside domain xlim={xlim}, ylim={ylim}, "
                f"wall_buffer={wall_buffer}, edge_buffer={edge_buffer}"
            )
        return ids

    def _sample_ids(self, ids, n):
        ids = np.asarray(ids, dtype=np.int64)
        if n <= 0:
            return np.empty((0,), dtype=np.int64)
        return np.random.choice(ids, size=n, replace=len(ids) < n)

    def _wake_id_sets(self, candidate_ids, xlim=None, ylim=None):
        """Broad wake and core wake ids, used for data fitting and CFD-PDE residual."""
        s = self.snapshot
        x = s.x[candidate_ids]
        y = s.y[candidate_ids]

        x_max = float(xlim[1]) if xlim is not None else 12.0
        y_abs = max(abs(float(ylim[0])), abs(float(ylim[1]))) if ylim is not None else 4.0

        broad_x1 = min(x_max, 8.0)
        broad_y = min(y_abs, 2.0)
        broad = (x >= 0.25) & (x <= broad_x1) & (np.abs(y) <= broad_y)

        core_x0 = max(float(xlim[0]) if xlim is not None else -2.0, 0.75)
        core_x1 = min(x_max, 5.0)
        core_y = min(y_abs, 1.25)
        core = (x >= core_x0) & (x <= core_x1) & (np.abs(y) <= core_y)

        broad_ids = candidate_ids[np.where(broad)[0]]
        core_ids = candidate_ids[np.where(core)[0]]
        if len(broad_ids) == 0:
            broad_ids = candidate_ids
        if len(core_ids) == 0:
            core_ids = broad_ids
        return broad_ids, core_ids

    def sample(self, n_total: int, wake_frac: float = 0.7, xlim=None, ylim=None):
        """Sample CFD data for velocity fitting, biased toward the near wake."""
        s = self.snapshot
        all_ids = self._domain_candidate_ids(xlim=xlim, ylim=ylim, radius=0.5)
        broad_ids, core_ids = self._wake_id_sets(all_ids, xlim=xlim, ylim=ylim)

        n_core = int(0.40 * n_total)
        n_broad = int(max(0.0, wake_frac - 0.40) * n_total)
        n_rest = n_total - n_core - n_broad

        ids = np.concatenate([
            self._sample_ids(core_ids, n_core),
            self._sample_ids(broad_ids, n_broad),
            self._sample_ids(all_ids, n_rest),
        ])
        np.random.shuffle(ids)

        t = torch.full((len(ids), 1), float(s.t), device=self.device)
        x = torch.tensor(s.x[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        y = torch.tensor(s.y[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        u = torch.tensor(s.u[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        v = torch.tensor(s.v[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        return t, x, y, u, v

    def all_points(self, xlim=None, ylim=None, radius: float = 0.5):
        """
        Return EVERY CFD fluid point in the PDE domain (no sampling). Use this
        for BFGS data fitting if you actually want the data MSE to keep falling
        below 1e-4: random subsampling inside the BFGS loop adds gradient noise
        that BFGS interprets as nonconvergence and stalls.
        """
        s = self.snapshot
        ids = self._domain_candidate_ids(xlim=xlim, ylim=ylim, radius=radius)
        t = torch.full((len(ids), 1), float(s.t), device=self.device)
        x = torch.tensor(s.x[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        y = torch.tensor(s.y[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        u = torch.tensor(s.u[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        v = torch.tensor(s.v[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        return t, x, y, u, v

    def sample_domain_xy(self, n_total: int, xlim=None, ylim=None, radius: float = 0.5,
                          wall_buffer: float = 0.0, edge_buffer: float = 0.0):
        """Sample CFD mesh points for PDE residual, biased toward high-error wake zones."""
        s = self.snapshot
        all_ids = self._domain_candidate_ids(
            xlim=xlim, ylim=ylim, radius=radius,
            wall_buffer=wall_buffer, edge_buffer=edge_buffer,
        )
        broad_ids, core_ids = self._wake_id_sets(all_ids, xlim=xlim, ylim=ylim)

        n_core = int(0.50 * n_total)
        n_broad = int(0.30 * n_total)
        n_rest = n_total - n_core - n_broad
        ids = np.concatenate([
            self._sample_ids(core_ids, n_core),
            self._sample_ids(broad_ids, n_broad),
            self._sample_ids(all_ids, n_rest),
        ])
        np.random.shuffle(ids)

        t = torch.full((len(ids), 1), float(s.t), device=self.device)
        x = torch.tensor(s.x[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        y = torch.tensor(s.y[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        return t, x, y



# ==========================================================
# Sampling
# ==========================================================

def normalize_domain(xlim, ylim):
    xlim = tuple(float(v) for v in xlim)
    ylim = tuple(float(v) for v in ylim)
    if len(xlim) != 2 or len(ylim) != 2:
        raise ValueError("xlim and ylim must be length-2 tuples")
    if not (xlim[0] < xlim[1] and ylim[0] < ylim[1]):
        raise ValueError(f"Invalid domain: xlim={xlim}, ylim={ylim}")
    return xlim, ylim


def split_collocation_counts(n_f: int):
    # global + broad wake + core wake + near-cylinder + front/separation
    n1 = int(0.20 * n_f)   # global background
    n2 = int(0.25 * n_f)   # broad wake
    n3 = int(0.25 * n_f)   # core wake / shear layer
    n4 = int(0.15 * n_f)   # near cylinder annulus
    n5 = n_f - n1 - n2 - n3 - n4  # front / immediate separation
    return n1, n2, n3, n4, n5


def compute_focus_regions(xlim, ylim, radius=0.5):
    x0, x1 = xlim
    y0, y1 = ylim
    x_span = x1 - x0
    y_span = y1 - y0

    wake_x0 = max(0.0, x0)
    wake_x1 = min(x1, max(4.0, 0.75 * x1))
    wake_y = min(max(1.5, 0.45 * y_span), max(abs(y0), abs(y1)))

    annulus_outer = radius + min(0.45, 0.12 * x_span)
    front_x0 = max(x0, -2.0)
    front_x1 = min(x1, 1.5)
    sep_x0 = max(x0, -0.5)
    sep_x1 = min(x1, 4.0)

    return {
        "wake_x": (wake_x0, wake_x1),
        "wake_y": (-wake_y, wake_y),
        "annulus_outer": annulus_outer,
        "front_x": (front_x0, front_x1),
        "front_y": (max(y0, -2.5), min(y1, 2.5)),
        "sep_x": (sep_x0, sep_x1),
        "sep_y": (max(y0, -2.5), min(y1, 2.5)),
    }


def sample_training_batches(dataset, n_data, n_f, device, t_value, xlim, ylim, radius=0.5):
    x_f, y_f, t_f = sample_collocation(
        n_f=n_f,
        device=device,
        t_value=t_value,
        xlim=xlim,
        ylim=ylim,
        radius=radius,
    )
    t_d, x_d, y_d, u_d, v_d = dataset.sample(n_data, wake_frac=0.8, xlim=xlim, ylim=ylim)
    return (x_f, y_f, t_f), (t_d, x_d, y_d, u_d, v_d)


def outside_cylinder_mask(x, y, xc=0.0, yc=0.0, radius=0.5):
    return (x - xc) ** 2 + (y - yc) ** 2 >= radius ** 2


def sample_collocation(n_f: int,
                       device: torch.device,
                       t_value: float = 0.0,
                       xlim=(-3.0, 12.0),
                       ylim=(-4.0, 4.0),
                       cylinder_center=(0.0, 0.0),
                       radius=0.5):
    xlim, ylim = normalize_domain(xlim, ylim)
    xc, yc = cylinder_center
    n1, n2, n3, n4, n5 = split_collocation_counts(n_f)
    focus = compute_focus_regions(xlim, ylim, radius=radius)

    # 1) global background
    x1 = np.random.uniform(xlim[0], xlim[1], size=n1)
    y1 = np.random.uniform(ylim[0], ylim[1], size=n1)

    # 2) broad wake region
    x2 = np.random.uniform(focus["wake_x"][0], focus["wake_x"][1], size=n2)
    y2 = np.random.uniform(focus["wake_y"][0], focus["wake_y"][1], size=n2)

    # 3) core wake / shear-layer region
    core_x0 = max(xlim[0], 0.75)
    core_x1 = min(xlim[1], 5.0)
    core_y = min(max(abs(ylim[0]), abs(ylim[1])), 1.25)
    x3 = np.random.uniform(core_x0, core_x1, size=n3)
    y3 = np.random.uniform(-core_y, core_y, size=n3)

    # 4) tighter near-cylinder annulus
    theta4 = np.random.uniform(0.0, 2 * np.pi, size=n4)
    r4 = np.random.uniform(radius + 0.02, focus["annulus_outer"], size=n4)
    x4 = xc + r4 * np.cos(theta4)
    y4 = yc + r4 * np.sin(theta4)

    # 5) front / separation-focused sampling
    n5a = int(0.4 * n5)
    n5b = n5 - n5a

    x5a = np.random.uniform(focus["front_x"][0], focus["front_x"][1], size=n5a)
    y5a = np.random.uniform(focus["front_y"][0], focus["front_y"][1], size=n5a)

    x5b = np.random.uniform(focus["sep_x"][0], focus["sep_x"][1], size=n5b)
    y5b = np.random.uniform(focus["sep_y"][0], focus["sep_y"][1], size=n5b)

    x5 = np.concatenate([x5a, x5b])
    y5 = np.concatenate([y5a, y5b])

    x = np.concatenate([x1, x2, x3, x4, x5])
    y = np.concatenate([y1, y2, y3, y4, y5])
    t = np.full_like(x, fill_value=t_value, dtype=np.float32)

    domain_mask = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    cyl_mask = outside_cylinder_mask(x, y, xc=xc, yc=yc, radius=radius)
    mask = domain_mask & cyl_mask
    x, y, t = x[mask], y[mask], t[mask]

    return (
        torch.tensor(x, dtype=torch.float32, device=device).view(-1, 1),
        torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1),
        torch.tensor(t, dtype=torch.float32, device=device).view(-1, 1),
    )


def sample_inlet(n, device, t_value=0.0, x0=-5.0, ylim=(-10.0, 10.0)):
    x = torch.full((n, 1), x0, device=device)
    y = torch.empty((n, 1), device=device).uniform_(ylim[0], ylim[1])
    t = torch.full((n, 1), float(t_value), device=device)
    return x, y, t


def sample_top_bottom(n, device, t_value=0.0, xlim=(-5.0, 25.0), yabs=10.0):
    n2 = n // 2
    x_t = torch.empty((n2, 1), device=device).uniform_(xlim[0], xlim[1])
    y_t = torch.full((n2, 1), yabs, device=device)
    t_t = torch.full((n2, 1), float(t_value), device=device)
    x_b = torch.empty((n - n2, 1), device=device).uniform_(xlim[0], xlim[1])
    y_b = torch.full((n - n2, 1), -yabs, device=device)
    t_b = torch.full((n - n2, 1), float(t_value), device=device)
    return torch.cat([x_t, x_b]), torch.cat([y_t, y_b]), torch.cat([t_t, t_b])


def sample_outlet(n, device, t_value=0.0, x0=25.0, ylim=(-10.0, 10.0)):
    x = torch.full((n, 1), x0, device=device)
    y = torch.empty((n, 1), device=device).uniform_(ylim[0], ylim[1])
    t = torch.full((n, 1), float(t_value), device=device)
    return x, y, t


def sample_cylinder_wall_uniform(n, device, t_value=0.0, radius=0.5, center=(0.0, 0.0)):
    xc, yc = center
    theta = torch.linspace(0.0, 2 * math.pi, n + 1, device=device)[:-1].view(-1, 1)
    x = xc + radius * torch.cos(theta)
    y = yc + radius * torch.sin(theta)
    t = torch.full((n, 1), float(t_value), device=device)
    return x, y, t


# ==========================================================
# Losses
# ==========================================================

def compute_pde_residuals(model, x, y, t, Re=40.0):
    _, u, v, p = model_uvp(model, x, y, t)

    u_t = grad(u, t)
    v_t = grad(v, t)
    u_x = grad(u, x)
    v_x = grad(v, x)
    u_y = grad(u, y)
    v_y = grad(v, y)
    p_x = grad(p, x)
    p_y = grad(p, y)
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)
    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)

    nu = 1.0 / Re
    f_u = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_v = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    # continuity (divergence). For a stream-function model this is exactly zero
    # analytically, so this should drop very fast and never dominate. Including
    # it as a separate term gives the optimizer a clear, well-conditioned signal
    # that the spatial derivatives must be consistent.
    div = u_x + v_y
    return f_u, f_v, div


def compute_pde_loss(model, x, y, t, Re=40.0):
    f_u, f_v, div = compute_pde_residuals(model, x, y, t, Re=Re)
    return (f_u ** 2).mean() + (f_v ** 2).mean() + (div ** 2).mean()


def compute_bc_losses(model, device, t_value=0.0,
                      xlim=(-3.0, 12.0), ylim=(-4.0, 4.0),
                      n_inlet=4096, n_tb=4096, n_wall=8192, n_outlet=4096):
    # Larger BC sample counts. BCs are cheap (no second derivatives on most),
    # and undersampled BCs are a common reason the interior fit stalls.
    # inlet
    xi, yi, ti = sample_inlet(n_inlet, device=device, t_value=t_value, x0=xlim[0], ylim=ylim)
    _, u_i, v_i, _ = model_uvp(model, xi, yi, ti)
    loss_inlet = ((u_i - 1.0) ** 2).mean() + (v_i ** 2).mean()

    # top/bottom
    xtb, ytb, ttb = sample_top_bottom(n_tb, device=device, t_value=t_value, xlim=xlim, yabs=ylim[1])
    _, u_tb, v_tb, _ = model_uvp(model, xtb, ytb, ttb)
    loss_tb = ((u_tb - 1.0) ** 2).mean() + (v_tb ** 2).mean()

    # wall (the no-slip is hard-enforced by lifting, so loss_wall_uv should be
    # ~0 numerically; we keep it as a sanity term, and add a tangential-pressure
    # smoothness check via psi).
    xw, yw, tw = sample_cylinder_wall_uniform(n_wall, device=device, t_value=t_value)
    psi_w, u_w, v_w, _ = model_uvp(model, xw, yw, tw)
    loss_wall_uv = (u_w ** 2).mean() + (v_w ** 2).mean()
    loss_wall_psi = (psi_w ** 2).mean()

    # outlet
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


def compute_data_loss(model, t_d, x_d, y_d, u_d, v_d):
    _, u, v, _ = model_uvp(model, x_d, y_d, t_d)
    return ((u - u_d) ** 2).mean() + ((v - v_d) ** 2).mean()


def default_fixed_weights():
    # Tuned for "drive absolute velocity error toward 1e-4 ~ 1e-5".
    # Data weight is much higher than PDE so the network first locks to CFD;
    # PDE then refines what data alone underconstrains (pressure, between-points).
    return {
        "pde": 1.0,
        "inlet": 2.0,
        "top_bottom": 2.0,
        "wall_uv": 10.0,
        "wall_psi": 0.2,
        "outlet": 1.0,
        "data": 50.0,
    }


def compute_total_loss(model, dataset, device, epoch, n_f, n_data, Re, t_value,
                       weight_manager=None,
                       fixed_weights=None,
                       x_f=None, y_f=None, t_f=None,
                       xlim=(-3.0, 12.0), ylim=(-4.0, 4.0),
                       n_cfd_pde=0, lambda_pde_cfd=1.0,
                       cfd_pde_wall_buffer=0.0, cfd_pde_edge_buffer=0.0):
    if x_f is None:
        (x_f, y_f, t_f), (t_d, x_d, y_d, u_d, v_d) = sample_training_batches(
            dataset=dataset,
            n_data=n_data,
            n_f=n_f,
            device=device,
            t_value=t_value,
            xlim=xlim,
            ylim=ylim,
            radius=0.5,
        )
    else:
        t_d, x_d, y_d, u_d, v_d = dataset.sample(n_data, wake_frac=0.8, xlim=xlim, ylim=ylim)

    loss_pde_colloc = compute_pde_loss(model, x_f, y_f, t_f, Re=Re)

    if n_cfd_pde > 0:
        t_cfd_pde, x_cfd_pde, y_cfd_pde = dataset.sample_domain_xy(
            n_cfd_pde, xlim=xlim, ylim=ylim, radius=0.5,
            wall_buffer=cfd_pde_wall_buffer, edge_buffer=cfd_pde_edge_buffer,
        )
        loss_pde_cfd = compute_pde_loss(model, x_cfd_pde, y_cfd_pde, t_cfd_pde, Re=Re)
    else:
        loss_pde_cfd = torch.zeros((), device=device)

    loss_pde = loss_pde_colloc + lambda_pde_cfd * loss_pde_cfd
    bc = compute_bc_losses(model, device=device, t_value=t_value, xlim=xlim, ylim=ylim)
    loss_data = compute_data_loss(model, t_d, x_d, y_d, u_d, v_d)

    raw_losses = {
        "pde": loss_pde,
        "pde_colloc": loss_pde_colloc,
        "pde_cfd": loss_pde_cfd,
        "inlet": bc["inlet"],
        "top_bottom": bc["top_bottom"],
        "wall_uv": bc["wall_uv"],
        "wall_psi": bc["wall_psi"],
        "outlet": bc["outlet"],
        "data": loss_data,
    }

    if weight_manager is not None:
        weight_manager.update(raw_losses)
        w = weight_manager.get()
    elif fixed_weights is not None:
        w = fixed_weights
    else:
        w = default_fixed_weights()

    total = (
        w["pde"] * raw_losses["pde"] +
        w["inlet"] * raw_losses["inlet"] +
        w["top_bottom"] * raw_losses["top_bottom"] +
        w["wall_uv"] * raw_losses["wall_uv"] +
        w["wall_psi"] * raw_losses["wall_psi"] +
        w["outlet"] * raw_losses["outlet"] +
        w["data"] * raw_losses["data"]
    )

    logs = {
        "L": total,
        "PDE": raw_losses["pde"],
        "PDE_COLLOC": raw_losses["pde_colloc"],
        "PDE_CFD": raw_losses["pde_cfd"],
        "INLET": raw_losses["inlet"],
        "TB": raw_losses["top_bottom"],
        "WALL_UV": raw_losses["wall_uv"],
        "WALL_PSI": raw_losses["wall_psi"],
        "OUTLET": raw_losses["outlet"],
        "DATA": raw_losses["data"],
        "weights": w,
    }
    return total, logs


# ==========================================================
# Checkpoint helpers
# ==========================================================

def save_training_state(model, save_dir, epoch, best_loss, stage, extra_state=None):
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "stage": stage,
    }
    if extra_state is not None:
        state.update(extra_state)
    atomic_torch_save(state, os.path.join(save_dir, "pinn_latest.pt"))
    atomic_torch_save(state, os.path.join(save_dir, "pinn_Re40_single.pt"))


# ==========================================================
# Adam training
# ==========================================================

def run_adam(model, dataset, device, save_dir,
             epochs_adam, n_f, n_data, lr_adam, Re, t_value,
             xlim=(-3.0, 12.0), ylim=(-4.0, 4.0),
             n_cfd_pde=0, lambda_pde_cfd=1.0,
             cfd_pde_wall_buffer=0.0, cfd_pde_edge_buffer=0.0):
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    adam = torch.optim.Adam(model.parameters(), lr=lr_adam)
    weight_manager = AdaptiveLossWeights()
    frozen_weights = default_fixed_weights()

    for epoch in range(1, epochs_adam + 1):
        model.train()
        adam.zero_grad()

        loss, logs = compute_total_loss(
            model=model,
            dataset=dataset,
            device=device,
            epoch=epoch,
            n_f=n_f,
            n_data=n_data,
            Re=Re,
            t_value=t_value,
            weight_manager=weight_manager,
            xlim=xlim,
            ylim=ylim,
            n_cfd_pde=n_cfd_pde,
            lambda_pde_cfd=lambda_pde_cfd,
            cfd_pde_wall_buffer=cfd_pde_wall_buffer,
            cfd_pde_edge_buffer=cfd_pde_edge_buffer,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        adam.step()

        cur = float(loss.detach().cpu().item())
        frozen_weights = logs["weights"]

        if cur < best_loss:
            best_loss = cur
            save_training_state(
                model, save_dir, epoch, best_loss, "adam",
                extra_state={"frozen_weights": frozen_weights},
            )
        else:
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "stage": "adam",
                    "frozen_weights": frozen_weights,
                },
                os.path.join(save_dir, "pinn_latest.pt"),
            )

        if epoch % 20 == 0:
            w = logs["weights"]
            print(
                f"[Adam] ep={epoch:5d}  "
                f"total={logs['L'].item():.3e}  "
                f"PDE={logs['PDE'].item():.3e} [colloc={logs['PDE_COLLOC'].item():.3e}, cfd={logs['PDE_CFD'].item():.3e}] (w={w['pde']:.3f})  "
                f"IN={logs['INLET'].item():.3e} (w={w['inlet']:.3f})  "
                f"TB={logs['TB'].item():.3e} (w={w['top_bottom']:.3f})  "
                f"WALL={logs['WALL_UV'].item():.3e} (w={w['wall_uv']:.3f})  "
                f"OUT={logs['OUTLET'].item():.3e} (w={w['outlet']:.3f})  "
                f"DATA={logs['DATA'].item():.3e} (w={w['data']:.3f})"
            )

    return best_loss, frozen_weights


# ==========================================================
# Batched BFGS
# ==========================================================

def run_scipy_bfgs(model: nn.Module,
                   dataset: SingleSnapshotDataset,
                   device: torch.device,
                   save_dir: str,
                   start_epoch: int,
                   maxiter: int,
                   n_f: int,
                   n_data: int,
                   Re: float,
                   t_value: float,
                   fixed_weights: dict,
                   iters_per_batch: int = 100,
                   gtol: float = 0.0,
                   disp: bool = False,
                   xlim=(-3.0, 12.0),
                   ylim=(-4.0, 4.0),
                   method_bfgs: str = "SSBroyden1",
                   n_cfd_pde: int = 0,
                   lambda_pde_cfd: float = 1.0,
                   full_data_bfgs: bool = True,
                   frozen_colloc_bfgs: bool = True,
                   cfd_pde_wall_buffer: float = 0.0,
                   cfd_pde_edge_buffer: float = 0.0):
    os.makedirs(save_dir, exist_ok=True)

    best = {"loss": float("inf")}
    total_calls = {"n": 0}
    batch_no = {"i": 0}

    initial_weights = (
        parameters_to_vector([p.detach() for p in model.parameters()])
        .cpu().numpy().astype(np.float64)
    )
    H0 = np.eye(initial_weights.size, dtype=np.float64)

    # ---------- frozen tensors for deterministic BFGS objective ----------
    # If frozen_colloc_bfgs: build ONE collocation set up front and reuse it.
    # If full_data_bfgs: pull ALL CFD points up front and reuse them.
    # This turns the BFGS objective into a deterministic function of the
    # weights, which is what BFGS / SSBroyden actually needs to converge to
    # 1e-4 ~ 1e-6. Stochastic resampling each call caps you at ~1e-3.
    if frozen_colloc_bfgs:
        x_f_fix, y_f_fix, t_f_fix = sample_collocation(
            n_f=n_f, device=device, t_value=t_value,
            xlim=xlim, ylim=ylim, radius=0.5,
        )
        if n_cfd_pde > 0:
            t_cfd_fix, x_cfd_fix, y_cfd_fix = dataset.sample_domain_xy(
                n_cfd_pde, xlim=xlim, ylim=ylim, radius=0.5,
                wall_buffer=cfd_pde_wall_buffer, edge_buffer=cfd_pde_edge_buffer,
            )
        else:
            t_cfd_fix = x_cfd_fix = y_cfd_fix = None
    else:
        x_f_fix = y_f_fix = t_f_fix = None
        t_cfd_fix = x_cfd_fix = y_cfd_fix = None

    if full_data_bfgs:
        t_d_fix, x_d_fix, y_d_fix, u_d_fix, v_d_fix = dataset.all_points(
            xlim=xlim, ylim=ylim, radius=0.5,
        )
        print(f"[BFGS] full data fitting: {len(x_d_fix)} CFD points pinned per call")
    else:
        t_d_fix = x_d_fix = y_d_fix = u_d_fix = v_d_fix = None

    def loss_and_gradient(weights: np.ndarray):
        total_calls["n"] += 1
        epoch_now = start_epoch + total_calls["n"]

        w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        with torch.no_grad():
            vector_to_parameters(w_tensor, model.parameters())

        model.train()
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        if frozen_colloc_bfgs:
            x_f, y_f, t_f = x_f_fix, y_f_fix, t_f_fix
        else:
            x_f, y_f, t_f = sample_collocation(
                n_f=n_f, device=device, t_value=t_value,
                xlim=xlim, ylim=ylim, radius=0.5,
            )
        if full_data_bfgs:
            t_d, x_d, y_d, u_d, v_d = t_d_fix, x_d_fix, y_d_fix, u_d_fix, v_d_fix
        else:
            t_d, x_d, y_d, u_d, v_d = dataset.sample(n_data, wake_frac=0.8, xlim=xlim, ylim=ylim)

        loss_pde_colloc = compute_pde_loss(model, x_f, y_f, t_f, Re=Re)
        if n_cfd_pde > 0:
            if frozen_colloc_bfgs:
                t_cfd_pde, x_cfd_pde, y_cfd_pde = t_cfd_fix, x_cfd_fix, y_cfd_fix
            else:
                t_cfd_pde, x_cfd_pde, y_cfd_pde = dataset.sample_domain_xy(n_cfd_pde, xlim=xlim, ylim=ylim, radius=0.5)
            loss_pde_cfd = compute_pde_loss(model, x_cfd_pde, y_cfd_pde, t_cfd_pde, Re=Re)
        else:
            loss_pde_cfd = torch.zeros((), device=device)

        loss_pde = loss_pde_colloc + lambda_pde_cfd * loss_pde_cfd
        bc = compute_bc_losses(model, device=device, t_value=t_value, xlim=xlim, ylim=ylim)
        loss_data = compute_data_loss(model, t_d, x_d, y_d, u_d, v_d)

        total = (
            fixed_weights["pde"] * loss_pde +
            fixed_weights["inlet"] * bc["inlet"] +
            fixed_weights["top_bottom"] * bc["top_bottom"] +
            fixed_weights["wall_uv"] * bc["wall_uv"] +
            fixed_weights["wall_psi"] * bc["wall_psi"] +
            fixed_weights["outlet"] * bc["outlet"] +
            fixed_weights["data"] * loss_data
        )
        total.backward()

        grad_flat = np.concatenate([
            p.grad.detach().cpu().numpy().ravel().astype(np.float64)
            if p.grad is not None
            else np.zeros(p.numel(), dtype=np.float64)
            for p in model.parameters()
        ])

        cur = float(total.detach().cpu().item())
        if cur < best["loss"]:
            best["loss"] = cur
            save_training_state(
                model, save_dir, epoch_now, best["loss"], "scipy_bfgs",
                extra_state={"frozen_weights": fixed_weights},
            )

        loss_and_gradient._last = {
            "total": cur,
            "pde": float(loss_pde.detach().cpu().item()),
            "pde_colloc": float(loss_pde_colloc.detach().cpu().item()),
            "pde_cfd": float(loss_pde_cfd.detach().cpu().item()),
            "inlet": float(bc["inlet"].detach().cpu().item()),
            "tb": float(bc["top_bottom"].detach().cpu().item()),
            "wall": float(bc["wall_uv"].detach().cpu().item()),
            "outlet": float(bc["outlet"].detach().cpu().item()),
            "data": float(loss_data.detach().cpu().item()),
        }
        return cur, grad_flat

    loss_and_gradient._last = {}

    def callback(xk):
        last = loss_and_gradient._last
        if not last:
            return
        print(
            f"[BFGS] batch={batch_no['i']:4d}  "
            f"call={total_calls['n']:6d}  "
            f"total={last['total']:.3e}  "
            f"PDE={last['pde']:.3e} [colloc={last['pde_colloc']:.3e}, cfd={last['pde_cfd']:.3e}]  "
            f"WALL={last['wall']:.3e}  "
            f"DATA={last['data']:.3e}"
        )

    if _CUSTOM_MIN_BFGS is None and method_bfgs not in ("BFGS", "BFGS_scipy"):
        warnings.warn(
            f"Requested method_bfgs='{method_bfgs}' but custom _optimize is not "
            f"available. Falling back to scipy standard BFGS — expect to plateau "
            f"around 1e-3 instead of reaching 1e-6.",
            RuntimeWarning,
        )

    use_custom = _CUSTOM_MIN_BFGS is not None
    bfgs_method = _CUSTOM_MIN_BFGS if use_custom else "BFGS"
    print(f"[BFGS] using {'CUSTOM _optimize._minimize_bfgs' if use_custom else 'scipy stock BFGS'} "
          f"(method_bfgs='{method_bfgs}')")

    while total_calls["n"] < maxiter:
        batch_no["i"] += 1

        result = minimize(
            loss_and_gradient,
            initial_weights,
            method=bfgs_method,
            jac=True,
            options={
                "maxiter": iters_per_batch,
                "gtol": gtol,
                "hess_inv0": H0,
                "disp": disp,
                "method_bfgs": method_bfgs,
            },
            tol=0,
            callback=callback,
        )

        initial_weights = result.x

        H0 = np.array(result.hess_inv, dtype=np.float64)
        H0 = 0.5 * (H0 + H0.T)
        try:
            cholesky(H0)
        except LinAlgError:
            print(f"[BFGS] batch {batch_no['i']}: H0 not PD — resetting to identity.")
            H0 = np.eye(initial_weights.size, dtype=np.float64)

        last = loss_and_gradient._last
        if last:
            print(
                f"[BFGS-END] batch={batch_no['i']:4d}  "
                f"last_total={last['total']:.3e}  "
                f"last_PDE={last['pde']:.3e} [colloc={last['pde_colloc']:.3e}, cfd={last['pde_cfd']:.3e}]  "
                f"last_WALL={last['wall']:.3e}  "
                f"last_DATA={last['data']:.3e}"
            )

        if total_calls["n"] >= maxiter:
            break

    w_tensor = torch.tensor(initial_weights, dtype=torch.float32, device=device)
    with torch.no_grad():
        vector_to_parameters(w_tensor, model.parameters())

    save_training_state(
        model, save_dir,
        start_epoch + total_calls["n"],
        best["loss"],
        "scipy_bfgs_final",
        extra_state={"frozen_weights": fixed_weights},
    )
    print(f"[BFGS] done. total_calls={total_calls['n']}  best_loss={best['loss']:.6e}")


# ==========================================================
# Top-level training entry point
# ==========================================================

def train_re40_single(model: nn.Module,
                      snapshot: Snapshot,
                      device: torch.device,
                      save_dir: str,
                      epochs_adam: int = 800,
                      maxiter_bfgs: int = 2000,
                      iters_per_batch: int = 100,
                      n_f: int = 6000,
                      n_data: int = 4000,
                      lr_adam: float = 1e-3,
                      gtol_bfgs: float = 0.0,
                      Re: float = 40.0,
                      t_value: float = 0.0,
                      xlim=(-3.0, 12.0),
                      ylim=(-4.0, 4.0),
                      method_bfgs: str = "SSBroyden1",
                      n_cfd_pde: int = 0,
                      lambda_pde_cfd: float = 1.0,
                      full_data_bfgs: bool = True,
                      frozen_colloc_bfgs: bool = True,
                      cfd_pde_wall_buffer: float = 0.0,
                      cfd_pde_edge_buffer: float = 0.0):

    os.makedirs(save_dir, exist_ok=True)
    dataset = SingleSnapshotDataset(snapshot, device=device)

    print("=" * 60)
    print(f"Phase 1 — Adam ({epochs_adam} epochs)")
    print("Goal: match CFD while improving PDE and enforcing wall no-slip exactly")
    print("=" * 60)
    _, frozen_weights = run_adam(
        model=model,
        dataset=dataset,
        device=device,
        save_dir=save_dir,
        epochs_adam=epochs_adam,
        n_f=n_f,
        n_data=n_data,
        lr_adam=lr_adam,
        Re=Re,
        t_value=t_value,
        xlim=xlim,
        ylim=ylim,
        n_cfd_pde=n_cfd_pde,
        lambda_pde_cfd=lambda_pde_cfd,
        cfd_pde_wall_buffer=cfd_pde_wall_buffer,
        cfd_pde_edge_buffer=cfd_pde_edge_buffer,
    )

    if maxiter_bfgs > 0:
        print("=" * 60)
        print(f"Phase 2 — BFGS ({maxiter_bfgs} total calls, {iters_per_batch} iters/batch)")
        print(f"Adaptive weights are frozen for BFGS stability | method={method_bfgs}")
        print("=" * 60)
        run_scipy_bfgs(
            model=model,
            dataset=dataset,
            device=device,
            save_dir=save_dir,
            start_epoch=epochs_adam,
            maxiter=maxiter_bfgs,
            n_f=n_f,
            n_data=n_data,
            Re=Re,
            t_value=t_value,
            fixed_weights=frozen_weights,
            iters_per_batch=iters_per_batch,
            gtol=gtol_bfgs,
            disp=False,
            xlim=xlim,
            ylim=ylim,
            method_bfgs=method_bfgs,
            n_cfd_pde=n_cfd_pde,
            lambda_pde_cfd=lambda_pde_cfd,
            full_data_bfgs=full_data_bfgs,
            frozen_colloc_bfgs=frozen_colloc_bfgs,
            cfd_pde_wall_buffer=cfd_pde_wall_buffer,
            cfd_pde_edge_buffer=cfd_pde_edge_buffer,
        )

    print(f"[train] Saved -> {os.path.join(save_dir, 'pinn_Re40_single.pt')}")


# ==========================================================
# Visualization
# ==========================================================

def load_model_for_viz(save_dir, device, width, depth):
    model = MLPStreamPressure(width=width, depth=depth).to(device)
    ckpt_main = os.path.join(save_dir, "pinn_Re40_single.pt")
    ckpt_fall = os.path.join(save_dir, "pinn_latest.pt")

    path = ckpt_main if (os.path.exists(ckpt_main) and os.path.getsize(ckpt_main) > 0) else ckpt_fall
    state = safe_load_checkpoint(path, device)
    if "model" not in state:
        raise KeyError(f"Checkpoint {path} has no 'model' key")

    model.load_state_dict(state["model"])
    model.eval()
    print(f"[viz] loaded {path}")
    return model


def evaluate_on_grid(model, device, t_val=0.0,
                     nx=320, ny=160,
                     xlim=(-3.0, 12.0), ylim=(-4.0, 4.0)):
    xs = np.linspace(xlim[0], xlim[1], nx, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    x = torch.tensor(X.reshape(-1, 1), device=device)
    y = torch.tensor(Y.reshape(-1, 1), device=device)
    t = torch.full_like(x, float(t_val))

    _, u, v, _ = model_uvp(model, x, y, t)
    omega = model_vorticity(model, x, y, t)

    U = u.detach().cpu().numpy().reshape(ny, nx)
    V = v.detach().cpu().numpy().reshape(ny, nx)
    W = omega.detach().cpu().numpy().reshape(ny, nx)

    mask = (X ** 2 + Y ** 2) <= 0.5 ** 2
    U[mask] = np.nan
    V[mask] = np.nan
    W[mask] = np.nan
    return X, Y, U, V, W


def evaluate_against_cfd(model, snapshot, device, xlim, ylim, radius=0.5):
    """
    Print ABSOLUTE error metrics on the CFD point cloud. Use these to decide
    if you have hit the mentor's 1e-4 ~ 1e-5 absolute-velocity-error target.
    Relative L2 of v can look bad even when absolute v is fine, because v is
    intrinsically small in this flow; trust MAE / max here.
    """
    sx, sy, su, sv = snapshot.x, snapshot.y, snapshot.u, snapshot.v
    in_box = (sx >= xlim[0]) & (sx <= xlim[1]) & (sy >= ylim[0]) & (sy <= ylim[1])
    out_cyl = (sx ** 2 + sy ** 2) >= radius ** 2
    m = in_box & out_cyl
    x = torch.tensor(sx[m], device=device, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(sy[m], device=device, dtype=torch.float32).view(-1, 1)
    t = torch.full_like(x, float(snapshot.t))
    _, u_p, v_p, _ = model_uvp(model, x, y, t)
    u_p = u_p.detach().cpu().numpy().ravel()
    v_p = v_p.detach().cpu().numpy().ravel()
    eu = u_p - su[m]
    ev = v_p - sv[m]
    speed_true = np.sqrt(su[m] ** 2 + sv[m] ** 2)
    print()
    print("=" * 60)
    print("ABSOLUTE error on CFD point cloud (target: ~1e-4 ~ 1e-5)")
    print("=" * 60)
    print(f"  points used           : {int(m.sum())}")
    print(f"  speed magnitude (mean): {float(speed_true.mean()):.4e}")
    print(f"  mae_u                 : {float(np.mean(np.abs(eu))):.4e}")
    print(f"  mae_v                 : {float(np.mean(np.abs(ev))):.4e}")
    print(f"  rmse_u                : {float(np.sqrt(np.mean(eu**2))):.4e}")
    print(f"  rmse_v                : {float(np.sqrt(np.mean(ev**2))):.4e}")
    print(f"  max |eu|              : {float(np.max(np.abs(eu))):.4e}")
    print(f"  max |ev|              : {float(np.max(np.abs(ev))):.4e}")
    print(f"  rel_l2_u              : {float(np.linalg.norm(eu)/np.linalg.norm(su[m])):.4e}")
    print(f"  rel_l2_v              : {float(np.linalg.norm(ev)/np.linalg.norm(sv[m])):.4e}")
    print()


def visualize_re40_single(save_dir, device, out_dir, width, depth,
                          xlim=(-3.0, 12.0), ylim=(-4.0, 4.0),
                          snapshot=None):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model_for_viz(save_dir, device, width=width, depth=depth)
    if snapshot is not None:
        evaluate_against_cfd(model, snapshot, device, xlim=xlim, ylim=ylim)
    X, Y, U, V, W = evaluate_on_grid(model, device=device, t_val=0.0, xlim=xlim, ylim=ylim)
    speed = np.sqrt(U ** 2 + V ** 2)

    for field, label, fname in [
        (W, "vorticity", "re40_single_vorticity.png"),
        (speed, "|u|", "re40_single_speed.png"),
    ]:
        fig = plt.figure(figsize=(7, 4))
        cmap = "coolwarm" if label == "vorticity" else None
        plt.pcolormesh(X, Y, field, shading="auto", cmap=cmap)
        plt.colorbar(label=label)
        plt.gca().add_patch(plt.Circle((0.0, 0.0), 0.5, color="k", fill=False))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Re=40 single snapshot: {label}")
        plt.tight_layout()
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"[viz] saved {out_path}")


# ==========================================================
# CLI
# ==========================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vtk-path", type=str, required=True)
    p.add_argument("--save-dir", type=str, default="checkpoints_re40_single")
    p.add_argument("--viz-dir", type=str, default="viz_re40_single")
    # Bigger network: 96/5 (~46k params) underfits the wake-pressure coupling
    # at this Re. 128/6 (~115k params) is still cheap and gives ~3x more
    # representational capacity, which is the difference between mae~1e-2 and
    # mae~1e-4.
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--epochs-adam", type=int, default=4000)
    p.add_argument("--maxiter-bfgs", type=int, default=12000,
                   help="Total BFGS loss+grad calls. BFGS is the only thing that "
                        "reliably gets you the last 2 orders of magnitude.")
    p.add_argument("--iters-per-batch", type=int, default=200,
                   help="maxiter per minimize() call")
    # Mentor's first point — 'use a lot more collocation points'.
    # 12000 -> 50000 (PDE colloc, random) and 2000 -> 12000 (PDE on CFD grid).
    p.add_argument("--n-f", type=int, default=50000)
    p.add_argument("--n-data", type=int, default=20000)
    p.add_argument("--lr-adam", type=float, default=1e-3)
    p.add_argument("--gtol-bfgs", type=float, default=0.0)
    p.add_argument("--method-bfgs", type=str, default="SSBroyden1",
                   choices=["BFGS", "BFGS_scipy", "SSBFGS_OL", "SSBFGS_AB", "SSBroyden1", "SSBroyden2"])
    p.add_argument("--n-cfd-pde", type=int, default=12000,
                   help="How many CFD grid points to also use for PDE residual each loss eval")
    p.add_argument("--lambda-pde-cfd", type=float, default=2.0,
                   help="Multiplier on CFD-grid PDE residual inside total PDE loss. "
                        "Mentor's second point — push the PDE to be satisfied right at "
                        "the points where we are matching data.")
    # Diagnostics on Re40.vtk show CFD has |vort transport| ~ 1~10 inside the
    # 0.5-thick band around the cylinder/box (boundary layer + far-field
    # under-resolution). Enforcing PDE residual on those CFD points is
    # numerically a contradiction. Defaults exclude that band.
    p.add_argument("--cfd-pde-wall-buffer", type=float, default=0.5,
                   help="Skip CFD points within this distance of the cylinder when "
                        "evaluating PDE residual on the CFD grid.")
    p.add_argument("--cfd-pde-edge-buffer", type=float, default=0.5,
                   help="Skip CFD points within this distance of the PDE box edges "
                        "when evaluating PDE residual on the CFD grid.")
    # Use the full CFD point cloud for data fitting in BFGS phase. Random
    # subsampling injects noise into the gradient that BFGS cannot tolerate
    # below ~1e-4. With this flag every BFGS evaluation sees ALL CFD points
    # and the data MSE can drop monotonically toward 1e-5 ~ 1e-6.
    p.add_argument("--full-data-bfgs", action="store_true", default=True,
                   help="In BFGS, fit ALL CFD points instead of random subset.")
    p.add_argument("--no-full-data-bfgs", dest="full_data_bfgs", action="store_false")
    # Likewise freeze the collocation set during BFGS. Random colloc each
    # call gives BFGS a stochastic objective and stalls it around 1e-3.
    p.add_argument("--frozen-colloc-bfgs", action="store_true", default=True,
                   help="Freeze (x_f, y_f, t_f) during BFGS so BFGS sees a deterministic loss.")
    p.add_argument("--no-frozen-colloc-bfgs", dest="frozen_colloc_bfgs", action="store_false")
    p.add_argument("--x-min", type=float, default=-3.0)
    p.add_argument("--x-max", type=float, default=12.0)
    p.add_argument("--y-min", type=float, default=-4.0)
    p.add_argument("--y-max", type=float, default=4.0)
    p.add_argument("--lift-delta", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--viz-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    snapshot = load_single_vtk(args.vtk_path, t_value=0.0)
    xlim = (args.x_min, args.x_max)
    ylim = (args.y_min, args.y_max)

    if not args.viz_only:
        model = MLPStreamPressure(
            width=args.width,
            depth=args.depth,
            radius=0.5,
            lift_delta=args.lift_delta,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: width={args.width}, depth={args.depth}, params={n_params}")
        print("Loss weights: adaptive during Adam, frozen during BFGS")
        print("Cylinder no-slip: enforced hard through stream-function lifting")
        print(f"CFD-aware PDE residual: n_cfd_pde={args.n_cfd_pde}, lambda_pde_cfd={args.lambda_pde_cfd}")

        train_re40_single(
            model=model,
            snapshot=snapshot,
            device=device,
            save_dir=args.save_dir,
            epochs_adam=args.epochs_adam,
            maxiter_bfgs=args.maxiter_bfgs,
            iters_per_batch=args.iters_per_batch,
            n_f=args.n_f,
            n_data=args.n_data,
            lr_adam=args.lr_adam,
            gtol_bfgs=args.gtol_bfgs,
            Re=40.0,
            t_value=0.0,
            xlim=xlim,
            ylim=ylim,
            method_bfgs=args.method_bfgs,
            n_cfd_pde=args.n_cfd_pde,
            lambda_pde_cfd=args.lambda_pde_cfd,
            full_data_bfgs=args.full_data_bfgs,
            frozen_colloc_bfgs=args.frozen_colloc_bfgs,
            cfd_pde_wall_buffer=args.cfd_pde_wall_buffer,
            cfd_pde_edge_buffer=args.cfd_pde_edge_buffer,
        )

    visualize_re40_single(
        save_dir=args.save_dir,
        device=device,
        out_dir=args.viz_dir,
        width=args.width,
        depth=args.depth,
        xlim=xlim,
        ylim=ylim,
        snapshot=snapshot,
    )


if __name__ == "__main__":
    main()
