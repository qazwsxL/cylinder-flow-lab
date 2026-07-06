
import os
import glob
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from scipy.optimize import fmin_bfgs

# Optional dependency for reading VTK files
try:
    import pyvista as pv
except Exception:
    pv = None


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
    Network: (x, y, t) -> (psi, p)
    Embedded input: x, y, t_norm, sin(2π t_norm), cos(2π t_norm)
    """

    def __init__(self, width: int = 32, depth: int = 3,
                 t_ref: float = 80.0, t_scale: float = 10.0):
        super().__init__()
        self.t_ref = t_ref
        self.t_scale = t_scale

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

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        inp = self.time_embed(x, y, t)
        return self.net(inp)


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
# Data containers
# ==========================================================

@dataclass
class Snapshot:
    t: float
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray


@dataclass
class TrainBatch:
    x_f: torch.Tensor
    y_f: torch.Tensor
    t_f: torch.Tensor

    t_d: torch.Tensor
    x_d: torch.Tensor
    y_d: torch.Tensor
    u_d: torch.Tensor
    v_d: torch.Tensor


# ==========================================================
# VTK loading
# ==========================================================

def infer_time_from_name(path: str) -> Optional[float]:
    base = os.path.basename(path)
    nums = []
    cur = ""
    for ch in base:
        if ch.isdigit() or ch == ".":
            cur += ch
        else:
            if cur:
                nums.append(cur)
                cur = ""
    if cur:
        nums.append(cur)
    if not nums:
        return None
    try:
        val = float(nums[-1])
        if val > 1000:
            return None
        return val
    except Exception:
        return None


def _pick_array_name(names, candidates):
    lower_map = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_vtk_series(vtk_dir: str, t_start: float = 80.0, dt: float = 0.2) -> List[Snapshot]:
    if pv is None:
        raise ImportError("pyvista is required to read VTK files. Install with: pip install pyvista")

    paths = sorted(glob.glob(os.path.join(vtk_dir, "*.vtk")))
    if not paths:
        raise FileNotFoundError(f"No .vtk files found under {vtk_dir}")

    snapshots: List[Snapshot] = []
    for i, path in enumerate(paths):
        mesh = pv.read(path)

        if mesh.n_cells > 0:
            centers = mesh.cell_centers().points
            data_src = mesh.cell_data
        else:
            centers = mesh.points
            data_src = mesh.point_data

        array_names = list(data_src.keys())
        vel_name = _pick_array_name(array_names, ["U", "u", "velocity", "Velocity", "vel"])
        if vel_name is None:
            raise KeyError(f"Could not find velocity array in {path}. Available arrays: {array_names}")

        vel = np.asarray(data_src[vel_name])
        if vel.ndim != 2 or vel.shape[1] < 2:
            raise ValueError(f"Velocity array {vel_name} in {path} has unexpected shape {vel.shape}")

        x = centers[:, 0].astype(np.float32)
        y = centers[:, 1].astype(np.float32)
        u = vel[:, 0].astype(np.float32)
        v = vel[:, 1].astype(np.float32)

        t = infer_time_from_name(path)
        if t is None:
            t = t_start + i * dt

        snapshots.append(Snapshot(t=t, x=x, y=y, u=u, v=v))
        print(f"Loaded {os.path.basename(path)}  t={t:.2f}s  cells={len(x)}")

    ts = [s.t for s in snapshots]
    print(f"[Series] {len(snapshots)} snapshots, t ∈ [{min(ts):.2f}, {max(ts):.2f}] s")
    return snapshots


# ==========================================================
# Dataset
# ==========================================================

class Re60SeriesDataset:
    def __init__(self, snapshots: List[Snapshot], device: torch.device):
        self.device = device
        self.snapshots = snapshots

        self.all_t = []
        self.all_x = []
        self.all_y = []
        self.all_u = []
        self.all_v = []

        for s in snapshots:
            n = len(s.x)
            self.all_t.append(np.full(n, s.t, dtype=np.float32))
            self.all_x.append(s.x.astype(np.float32))
            self.all_y.append(s.y.astype(np.float32))
            self.all_u.append(s.u.astype(np.float32))
            self.all_v.append(s.v.astype(np.float32))

        self.all_t = np.concatenate(self.all_t)
        self.all_x = np.concatenate(self.all_x)
        self.all_y = np.concatenate(self.all_y)
        self.all_u = np.concatenate(self.all_u)
        self.all_v = np.concatenate(self.all_v)

    def sample_single_snapshot(self, n_total: int, wake_frac: float = 0.8):
        s = self.snapshots[np.random.randint(len(self.snapshots))]
        x = s.x
        y = s.y

        wake_mask = (x >= 0.5) & (x <= 12.0) & (np.abs(y) <= 3.5)
        wake_ids = np.where(wake_mask)[0]
        all_ids = np.arange(len(x))

        n_wake = int(n_total * wake_frac)
        n_rest = n_total - n_wake

        ids_wake = np.random.choice(
            wake_ids if len(wake_ids) > 0 else all_ids,
            size=n_wake,
            replace=(len(wake_ids) < n_wake) if len(wake_ids) > 0 else (len(all_ids) < n_wake),
        )
        ids_rest = np.random.choice(all_ids, size=n_rest, replace=len(all_ids) < n_rest)
        ids = np.concatenate([ids_wake, ids_rest])
        np.random.shuffle(ids)

        t = torch.full((len(ids), 1), float(s.t), device=self.device)
        x = torch.tensor(s.x[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        y = torch.tensor(s.y[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        u = torch.tensor(s.u[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        v = torch.tensor(s.v[ids], dtype=torch.float32, device=self.device).view(-1, 1)
        return t, x, y, u, v

    def residual_resample_single_snapshot(self,
                                          model: nn.Module,
                                          n_total: int,
                                          wake_frac: float = 0.8,
                                          candidate_factor: int = 3):
        s = self.snapshots[np.random.randint(len(self.snapshots))]
        x = s.x
        y = s.y

        wake_mask = (x >= 0.5) & (x <= 12.0) & (np.abs(y) <= 3.5)
        wake_ids = np.where(wake_mask)[0]
        all_ids = np.arange(len(x))

        n_cand = min(len(all_ids), max(candidate_factor * n_total, n_total))
        n_wake = int(n_cand * wake_frac)
        n_rest = n_cand - n_wake

        ids_wake = np.random.choice(
            wake_ids if len(wake_ids) > 0 else all_ids,
            size=min(n_wake, len(wake_ids) if len(wake_ids) > 0 else len(all_ids)),
            replace=False if (len(wake_ids) >= n_wake if len(wake_ids) > 0 else len(all_ids) >= n_wake) else True,
        )
        ids_rest = np.random.choice(all_ids, size=n_rest, replace=len(all_ids) < n_rest)
        cand_ids = np.concatenate([ids_wake, ids_rest])
        np.random.shuffle(cand_ids)

        t_c = torch.full((len(cand_ids), 1), float(s.t), device=self.device)
        x_c = torch.tensor(s.x[cand_ids], dtype=torch.float32, device=self.device).view(-1, 1)
        y_c = torch.tensor(s.y[cand_ids], dtype=torch.float32, device=self.device).view(-1, 1)
        u_c = torch.tensor(s.u[cand_ids], dtype=torch.float32, device=self.device).view(-1, 1)
        v_c = torch.tensor(s.v[cand_ids], dtype=torch.float32, device=self.device).view(-1, 1)

        xg = x_c.detach().clone().requires_grad_(True)
        yg = y_c.detach().clone().requires_grad_(True)
        tg = t_c.detach().clone().requires_grad_(True)

        _, u_hat, v_hat, _ = model_uvp(model, xg, yg, tg)
        err = ((u_hat.detach() - u_c) ** 2 + (v_hat.detach() - v_c) ** 2).squeeze()

        probs = 0.9 * (err / (err.sum() + 1e-12)) + 0.1 * torch.full_like(err, 1.0 / err.numel())
        ids_local = torch.multinomial(probs, num_samples=n_total, replacement=(n_total > err.numel()))

        return t_c[ids_local], x_c[ids_local], y_c[ids_local], u_c[ids_local], v_c[ids_local]


# ==========================================================
# Sampling
# ==========================================================

def outside_cylinder_mask(x, y, xc=0.0, yc=0.0, radius=0.5):
    return (x - xc) ** 2 + (y - yc) ** 2 >= radius ** 2


def sample_collocation_re60(n_f: int,
                            device: torch.device,
                            xlim=(-5.0, 25.0),
                            ylim=(-10.0, 10.0),
                            tlim=(80.0, 90.0),
                            cylinder_center=(0.0, 0.0),
                            radius=0.5):
    xc, yc = cylinder_center
    n1 = int(0.5 * n_f)
    n2 = int(0.3 * n_f)
    n3 = n_f - n1 - n2

    x1 = np.random.uniform(xlim[0], xlim[1], size=n1)
    y1 = np.random.uniform(ylim[0], ylim[1], size=n1)
    t1 = np.random.uniform(tlim[0], tlim[1], size=n1)

    x2 = np.random.uniform(0.0, 12.0, size=n2)
    y2 = np.random.uniform(-3.0, 3.0, size=n2)
    t2 = np.random.uniform(tlim[0], tlim[1], size=n2)

    theta = np.random.uniform(0.0, 2 * np.pi, size=n3)
    rr = np.random.uniform(radius + 0.02, radius + 0.35, size=n3)
    x3 = xc + rr * np.cos(theta)
    y3 = yc + rr * np.sin(theta)
    t3 = np.random.uniform(tlim[0], tlim[1], size=n3)

    x = np.concatenate([x1, x2, x3])
    y = np.concatenate([y1, y2, y3])
    t = np.concatenate([t1, t2, t3])

    mask = outside_cylinder_mask(x, y, xc=xc, yc=yc, radius=radius)
    x, y, t = x[mask], y[mask], t[mask]

    return (
        torch.tensor(x, dtype=torch.float32, device=device).view(-1, 1),
        torch.tensor(y, dtype=torch.float32, device=device).view(-1, 1),
        torch.tensor(t, dtype=torch.float32, device=device).view(-1, 1),
    )


def sample_inlet(n: int, device: torch.device, x0=-5.0, ylim=(-10.0, 10.0), tlim=(80.0, 90.0)):
    x = torch.full((n, 1), x0, device=device)
    y = torch.empty((n, 1), device=device).uniform_(ylim[0], ylim[1])
    t = torch.empty((n, 1), device=device).uniform_(tlim[0], tlim[1])
    return x, y, t


def sample_top_bottom(n: int, device: torch.device, xlim=(-5.0, 25.0), yabs=10.0, tlim=(80.0, 90.0)):
    n2 = n // 2
    x_top = torch.empty((n2, 1), device=device).uniform_(xlim[0], xlim[1])
    y_top = torch.full((n2, 1), yabs, device=device)
    t_top = torch.empty((n2, 1), device=device).uniform_(tlim[0], tlim[1])

    x_bot = torch.empty((n - n2, 1), device=device).uniform_(xlim[0], xlim[1])
    y_bot = torch.full((n - n2, 1), -yabs, device=device)
    t_bot = torch.empty((n - n2, 1), device=device).uniform_(tlim[0], tlim[1])

    x = torch.cat([x_top, x_bot], dim=0)
    y = torch.cat([y_top, y_bot], dim=0)
    t = torch.cat([t_top, t_bot], dim=0)
    return x, y, t


def sample_outlet(n: int, device: torch.device, x0=25.0, ylim=(-10.0, 10.0), tlim=(80.0, 90.0)):
    x = torch.full((n, 1), x0, device=device)
    y = torch.empty((n, 1), device=device).uniform_(ylim[0], ylim[1])
    t = torch.empty((n, 1), device=device).uniform_(tlim[0], tlim[1])
    return x, y, t


def sample_cylinder_wall(n: int, device: torch.device, radius=0.5, center=(0.0, 0.0), tlim=(80.0, 90.0)):
    xc, yc = center
    theta = torch.empty((n, 1), device=device).uniform_(0.0, 2 * math.pi)
    x = xc + radius * torch.cos(theta)
    y = yc + radius * torch.sin(theta)
    t = torch.empty((n, 1), device=device).uniform_(tlim[0], tlim[1])
    return x, y, t


# ==========================================================
# Losses
# ==========================================================

def compute_pde_residuals(model: nn.Module,
                          x: torch.Tensor,
                          y: torch.Tensor,
                          t: torch.Tensor,
                          Re: float = 60.0):
    psi, u, v, p = model_uvp(model, x, y, t)

    u_t = grad(u, t)
    u_x = grad(u, x)
    u_y = grad(u, y)
    v_t = grad(v, t)
    v_x = grad(v, x)
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
    return f_u, f_v


def compute_pde_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, Re: float = 60.0):
    f_u, f_v = compute_pde_residuals(model, x, y, t, Re=Re)
    return (f_u ** 2).mean() + (f_v ** 2).mean()


def compute_bc_loss(model: nn.Module, device: torch.device, tmin=80.0, tmax=90.0,
                    xlim=(-5.0, 25.0), ylim=(-10.0, 10.0)):
    xi, yi, ti = sample_inlet(1024, device=device, x0=xlim[0], ylim=ylim, tlim=(tmin, tmax))
    _, u_i, v_i, _ = model_uvp(model, xi, yi, ti)
    v_target = 0.01 * torch.tanh(2.0 * yi)
    loss_inlet = ((u_i - 1.0) ** 2).mean() + ((v_i - v_target) ** 2).mean()

    xtb, ytb, ttb = sample_top_bottom(1024, device=device, xlim=xlim, yabs=ylim[1], tlim=(tmin, tmax))
    _, u_tb, v_tb, _ = model_uvp(model, xtb, ytb, ttb)
    loss_tb = ((u_tb - 1.0) ** 2).mean() + (v_tb ** 2).mean()

    xw, yw, tw = sample_cylinder_wall(2048, device=device, radius=0.5, center=(0.0, 0.0), tlim=(tmin, tmax))
    _, u_w, v_w, _ = model_uvp(model, xw, yw, tw)
    loss_wall = (u_w ** 2).mean() + (v_w ** 2).mean()

    xo, yo, to = sample_outlet(1024, device=device, x0=xlim[1], ylim=ylim, tlim=(tmin, tmax))
    _, u_o, v_o, p_o = model_uvp(model, xo, yo, to)
    u_x_o = grad(u_o, xo)
    v_x_o = grad(v_o, xo)
    loss_out = (p_o ** 2).mean() + (u_x_o ** 2).mean() + (v_x_o ** 2).mean()

    return loss_inlet + loss_tb + loss_wall + loss_out


def compute_ic_loss(model: nn.Module, device: torch.device, t0=80.0, n=2000,
                    xlim=(-5.0, 25.0), ylim=(-10.0, 10.0)):
    x = torch.empty((n, 1), device=device).uniform_(xlim[0], xlim[1])
    y = torch.empty((n, 1), device=device).uniform_(ylim[0], ylim[1])
    t = torch.full((n, 1), t0, device=device)

    mask = outside_cylinder_mask(x.detach().cpu().numpy().ravel(), y.detach().cpu().numpy().ravel())
    mask = torch.tensor(mask, device=device).view(-1, 1)
    x = x[mask[:, 0]].view(-1, 1)
    y = y[mask[:, 0]].view(-1, 1)
    t = t[mask[:, 0]].view(-1, 1)

    _, u, v, _ = model_uvp(model, x, y, t)

    wake_weight = torch.ones_like(x)
    wake_weight[(x > 0.0) & (x < 8.0) & (torch.abs(y) < 3.0)] = 0.1
    loss = (wake_weight * (u - 1.0) ** 2).mean() + (wake_weight * v ** 2).mean()
    return loss


def compute_data_loss(model: nn.Module,
                      t_d: torch.Tensor, x_d: torch.Tensor, y_d: torch.Tensor,
                      u_d: torch.Tensor, v_d: torch.Tensor):
    _, u, v, _ = model_uvp(model, x_d, y_d, t_d)
    return ((u - u_d) ** 2).mean() + ((v - v_d) ** 2).mean()


def build_probe_series_from_snapshots(snapshots: List[Snapshot], x0=3.0, y0=0.0):
    ts, vs = [], []
    for s in snapshots:
        idx = np.argmin((s.x - x0) ** 2 + (s.y - y0) ** 2)
        ts.append(s.t)
        vs.append(s.v[idx])
    return np.array(ts, dtype=np.float32), np.array(vs, dtype=np.float32)


def compute_probe_loss(model: nn.Module, probe_t, probe_v, device: torch.device, x0=3.0, y0=0.0):
    t = torch.tensor(probe_t, dtype=torch.float32, device=device).view(-1, 1)
    x = torch.full_like(t, x0, device=device)
    y = torch.full_like(t, y0, device=device)
    _, _, v, _ = model_uvp(model, x, y, t)
    v_tar = torch.tensor(probe_v, dtype=torch.float32, device=device).view(-1, 1)
    return ((v - v_tar) ** 2).mean()


def compute_symmetry_break_loss(model: nn.Module, device: torch.device, n=512, t0=80.0):
    x = torch.full((n, 1), -4.8, device=device)
    y = torch.empty((n, 1), device=device).uniform_(-2.5, 2.5)
    t = torch.empty((n, 1), device=device).uniform_(t0, t0 + 1.0)
    _, _, v, _ = model_uvp(model, x, y, t)
    v_tar = 0.01 * torch.tanh(2.0 * y)
    return ((v - v_tar) ** 2).mean()


# ==========================================================
# Residual-based resampling
# ==========================================================

def residual_resample_collocation(model: nn.Module,
                                  device: torch.device,
                                  n_f: int,
                                  candidate_factor: int = 4,
                                  xlim=(-5.0, 25.0),
                                  ylim=(-10.0, 10.0),
                                  tlim=(80.0, 90.0),
                                  Re: float = 60.0):
    n_cand = max(candidate_factor * n_f, n_f)

    x_c, y_c, t_c = sample_collocation_re60(
        n_f=n_cand,
        device=device,
        xlim=xlim,
        ylim=ylim,
        tlim=tlim,
    )

    xg = x_c.detach().clone().requires_grad_(True)
    yg = y_c.detach().clone().requires_grad_(True)
    tg = t_c.detach().clone().requires_grad_(True)

    f_u, f_v = compute_pde_residuals(model, xg, yg, tg, Re=Re)
    r = (f_u.detach().squeeze() ** 2 + f_v.detach().squeeze() ** 2)

    probs = 0.9 * (r / (r.sum() + 1e-12)) + 0.1 * torch.full_like(r, 1.0 / r.numel())
    ids = torch.multinomial(probs, num_samples=n_f, replacement=(n_f > r.numel()))

    return (
        x_c[ids].view(-1, 1),
        y_c[ids].view(-1, 1),
        t_c[ids].view(-1, 1),
    )


def build_resampled_train_batch(model: nn.Module,
                                series_ds: Re60SeriesDataset,
                                device: torch.device,
                                n_f: int,
                                n_data: int) -> TrainBatch:
    model.eval()

    x_f, y_f, t_f = residual_resample_collocation(
        model=model,
        device=device,
        n_f=n_f,
        candidate_factor=4,
        xlim=(-5.0, 25.0),
        ylim=(-10.0, 10.0),
        tlim=(80.0, 90.0),
        Re=60.0,
    )

    t_d, x_d, y_d, u_d, v_d = series_ds.residual_resample_single_snapshot(
        model=model,
        n_total=n_data,
        wake_frac=0.8,
        candidate_factor=3,
    )

    return TrainBatch(
        x_f=x_f, y_f=y_f, t_f=t_f,
        t_d=t_d, x_d=x_d, y_d=y_d, u_d=u_d, v_d=v_d
    )


# ==========================================================
# Training
# ==========================================================

def compute_total_loss(model: nn.Module,
                       probe_t,
                       probe_v,
                       device: torch.device,
                       epoch: int,
                       batch: TrainBatch):
    if epoch < 300:
        w_pde, w_bc, w_data, w_probe, w_sym, w_ic = 0.05, 8.0, 40.0, 120.0, 2.0, 0.5
    elif epoch < 1000:
        w_pde, w_bc, w_data, w_probe, w_sym, w_ic = 0.10, 10.0, 30.0, 100.0, 1.0, 0.2
    elif epoch < 2500:
        w_pde, w_bc, w_data, w_probe, w_sym, w_ic = 0.5, 10.0, 20.0, 80.0, 0.5, 0.0
    else:
        w_pde, w_bc, w_data, w_probe, w_sym, w_ic = 1.0, 10.0, 12.0, 50.0, 0.2, 0.0

    loss_pde = compute_pde_loss(model, batch.x_f, batch.y_f, batch.t_f, Re=60.0)
    loss_bc = compute_bc_loss(model, device, tmin=80.0, tmax=90.0)
    loss_ic = compute_ic_loss(model, device, t0=80.0)

    loss_data = compute_data_loss(
        model,
        batch.t_d, batch.x_d, batch.y_d,
        batch.u_d, batch.v_d
    )

    loss_probe = compute_probe_loss(model, probe_t, probe_v, device, x0=3.0, y0=0.0)
    loss_sym = compute_symmetry_break_loss(model, device, n=512, t0=80.0)

    total = (
        w_pde * loss_pde
        + w_bc * loss_bc
        + w_data * loss_data
        + w_probe * loss_probe
        + w_sym * loss_sym
        + w_ic * loss_ic
    )

    logs = {
        "L": total,
        "PDE": loss_pde,
        "BC": loss_bc,
        "IC": loss_ic,
        "DATA": loss_data,
        "PROBE": loss_probe,
        "SYM": loss_sym,
    }
    return total, logs


def get_flat_params(model: nn.Module) -> np.ndarray:
    return np.concatenate([
        p.detach().cpu().numpy().ravel() for p in model.parameters()
    ]).astype(np.float64)


def set_flat_params(model: nn.Module, flat_params: np.ndarray, device: torch.device):
    flat_params = np.asarray(flat_params, dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            numel = p.numel()
            vals = flat_params[offset:offset + numel].reshape(p.shape)
            p.copy_(torch.from_numpy(vals).to(device=device, dtype=p.dtype))
            offset += numel
    if offset != flat_params.size:
        raise ValueError(f"Parameter size mismatch: consumed {offset}, got {flat_params.size}")


def save_training_state(model: nn.Module, save_dir: str, epoch: int, best_loss: float, stage: str):
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "best_loss": best_loss,
        "stage": stage,
    }
    atomic_torch_save(state, os.path.join(save_dir, "pinn_latest.pt"))
    atomic_torch_save(state, os.path.join(save_dir, "pinn_Re60_only.pt"))


def run_scipy_bfgs_epoch(model: nn.Module,
                         probe_t,
                         probe_v,
                         device: torch.device,
                         save_dir: str,
                         epoch: int,
                         batch: TrainBatch,
                         maxiter: int,
                         gtol: float = 1e-5,
                         disp: bool = False):
    x0 = get_flat_params(model)
    best = {"loss": float("inf")}
    cache = {}

    def eval_fg(x_flat: np.ndarray):
        key = x_flat.tobytes()
        if key in cache:
            return cache[key]

        set_flat_params(model, x_flat, device)
        model.train()

        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        loss, logs = compute_total_loss(
            model=model,
            probe_t=probe_t,
            probe_v=probe_v,
            device=device,
            epoch=epoch,
            batch=batch,
        )
        loss.backward()

        grad_flat = []
        for p in model.parameters():
            if p.grad is None:
                grad_flat.append(np.zeros(p.numel(), dtype=np.float64))
            else:
                grad_flat.append(p.grad.detach().cpu().numpy().ravel().astype(np.float64))
        grad_flat = np.concatenate(grad_flat)

        cur = float(loss.detach().cpu().item())
        cache[key] = (cur, grad_flat, logs)

        if cur < best["loss"]:
            best["loss"] = cur
            save_training_state(
                model=model,
                save_dir=save_dir,
                epoch=epoch,
                best_loss=best["loss"],
                stage="scipy_bfgs",
            )
        return cache[key]

    result = fmin_bfgs(
        f=lambda x: eval_fg(x)[0],
        x0=x0,
        fprime=lambda x: eval_fg(x)[1],
        gtol=gtol,
        maxiter=maxiter,
        disp=disp,
        full_output=True,
        retall=False,
    )

    xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = result
    set_flat_params(model, xopt, device)

    final_loss, final_logs = compute_total_loss(
        model=model,
        probe_t=probe_t,
        probe_v=probe_v,
        device=device,
        epoch=epoch,
        batch=batch,
    )

    save_training_state(
        model=model,
        save_dir=save_dir,
        epoch=epoch,
        best_loss=min(best["loss"], float(fopt)),
        stage="scipy_bfgs_epoch_final",
    )

    print(
        f"[BFGS  Re=60.0] ep={epoch:5d} "
        f"L={final_logs['L'].item():.3e} "
        f"PDE={final_logs['PDE'].item():.3e} "
        f"BC={final_logs['BC'].item():.3e} "
        f"IC={final_logs['IC'].item():.3e} "
        f"DATA={final_logs['DATA'].item():.3e} "
        f"PROBE={final_logs['PROBE'].item():.3e} "
        f"SYM={final_logs['SYM'].item():.3e} "
        f"| func_calls={func_calls} grad_calls={grad_calls} warnflag={warnflag}"
    )

    return result, final_logs


def train_re60_only(model: nn.Module,
                    snapshots: List[Snapshot],
                    device: torch.device,
                    save_dir: str,
                    epochs_adam: int = 100,
                    epochs_bfgs: int = 200,
                    bfgs_steps_per_epoch: int = 3,
                    n_f: int = 8000,
                    n_data: int = 4000,
                    lr_adam: float = 1e-3,
                    gtol_bfgs: float = 1e-5):
    os.makedirs(save_dir, exist_ok=True)
    series_ds = Re60SeriesDataset(snapshots, device=device)
    probe_t, probe_v = build_probe_series_from_snapshots(snapshots, x0=3.0, y0=0.0)

    best_loss = float("inf")
    adam = torch.optim.Adam(model.parameters(), lr=lr_adam)

    # Adam warmup with epoch-level residual resampling
    for epoch in range(1, epochs_adam + 1):
        model.train()
        batch = build_resampled_train_batch(
            model=model,
            series_ds=series_ds,
            device=device,
            n_f=n_f,
            n_data=n_data,
        )

        adam.zero_grad()
        loss, logs = compute_total_loss(
            model=model,
            probe_t=probe_t,
            probe_v=probe_v,
            device=device,
            epoch=epoch,
            batch=batch,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        adam.step()

        cur = float(loss.detach().cpu().item())
        if cur < best_loss:
            best_loss = cur
            save_training_state(model, save_dir, epoch, best_loss, stage="adam_warmup")
        else:
            atomic_torch_save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "stage": "adam_warmup",
            }, os.path.join(save_dir, "pinn_latest.pt"))

        if epoch % 10 == 0:
            print(
                f"[Adam  Re=60.0] ep={epoch:5d} "
                f"L={logs['L'].item():.3e} "
                f"PDE={logs['PDE'].item():.3e} "
                f"BC={logs['BC'].item():.3e} "
                f"IC={logs['IC'].item():.3e} "
                f"DATA={logs['DATA'].item():.3e} "
                f"PROBE={logs['PROBE'].item():.3e} "
                f"SYM={logs['SYM'].item():.3e}"
            )

    # BFGS main stage: resample every epoch, fixed batch within each BFGS call
    for k in range(1, epochs_bfgs + 1):
        epoch = epochs_adam + k

        batch = build_resampled_train_batch(
            model=model,
            series_ds=series_ds,
            device=device,
            n_f=n_f,
            n_data=n_data,
        )

        run_scipy_bfgs_epoch(
            model=model,
            probe_t=probe_t,
            probe_v=probe_v,
            device=device,
            save_dir=save_dir,
            epoch=epoch,
            batch=batch,
            maxiter=bfgs_steps_per_epoch,
            gtol=gtol_bfgs,
            disp=False,
        )

    print(f"[Re60-only] Saved -> {os.path.join(save_dir, 'pinn_Re60_only.pt')}")


# ==========================================================
# Visualization
# ==========================================================

def load_model_for_viz(save_dir: str, device: torch.device, width: int, depth: int):
    model = MLPStreamPressure(width=width, depth=depth).to(device)
    ckpt_main = os.path.join(save_dir, "pinn_Re60_only.pt")
    ckpt_fallback = os.path.join(save_dir, "pinn_latest.pt")

    if os.path.exists(ckpt_main) and os.path.getsize(ckpt_main) > 0:
        state = safe_load_checkpoint(ckpt_main, device)
        ckpt_path = ckpt_main
    else:
        state = safe_load_checkpoint(ckpt_fallback, device)
        ckpt_path = ckpt_fallback

    if "model" not in state:
        raise KeyError(f"Checkpoint {ckpt_path} does not contain key 'model'")

    model.load_state_dict(state["model"])
    model.eval()
    print(f"[viz] loading checkpoint: {ckpt_path}")
    return model


def evaluate_on_grid(model: nn.Module,
                     device: torch.device,
                     t_val: float,
                     nx: int = 320,
                     ny: int = 160,
                     xlim=(-0.5, 15.0),
                     ylim=(-4.0, 4.0)):
    xs = np.linspace(xlim[0], xlim[1], nx, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    x = torch.tensor(X.reshape(-1, 1), device=device)
    y = torch.tensor(Y.reshape(-1, 1), device=device)
    t = torch.full_like(x, float(t_val), device=device)

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


def visualize_saved_re60(save_dir: str,
                         device: torch.device,
                         out_dir: str,
                         width: int,
                         depth: int,
                         times: Optional[List[float]] = None):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model_for_viz(save_dir, device, width=width, depth=depth)

    if times is None:
        times = [84.2, 84.8, 85.4, 86.0, 86.6, 87.2, 87.8, 88.4]

    for t_val in times:
        X, Y, U, V, W = evaluate_on_grid(model, device, t_val)
        speed = np.sqrt(U ** 2 + V ** 2)

        fig = plt.figure(figsize=(7, 4))
        plt.pcolormesh(X, Y, speed, shading="auto")
        plt.colorbar(label="|u|")
        plt.contour(X, Y, W, levels=20, linewidths=0.5)
        circle = plt.Circle((0.0, 0.0), 0.5, color="k", fill=False)
        plt.gca().add_patch(circle)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Speed / vorticity at t={t_val:.2f}")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"re60_t{t_val:.2f}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[viz] saved {out_path}")


# ==========================================================
# Main
# ==========================================================

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk-dir", type=str, required=True, help="Directory containing Re60 *.vtk files")
    ap.add_argument("--save-dir", type=str, default="checkpoints_re60_only")
    ap.add_argument("--viz-dir", type=str, default="viz_re60")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)

    ap.add_argument("--epochs-adam", type=int, default=100)
    ap.add_argument("--epochs-bfgs", type=int, default=200)
    ap.add_argument("--bfgs-steps-per-epoch", type=int, default=3)
    ap.add_argument("--n-f", type=int, default=8000)
    ap.add_argument("--n-data", type=int, default=4000)
    ap.add_argument("--lr-adam", type=float, default=1e-3)
    ap.add_argument("--gtol-bfgs", type=float, default=1e-5)

    ap.add_argument("--viz-only", action="store_true")
    return ap


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    snapshots = load_vtk_series(args.vtk_dir, t_start=80.0, dt=0.2)

    if not args.viz_only:
        model = MLPStreamPressure(width=args.width, depth=args.depth).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: width={args.width}, depth={args.depth} (params={n_params})")

        train_re60_only(
            model=model,
            snapshots=snapshots,
            device=device,
            save_dir=args.save_dir,
            epochs_adam=args.epochs_adam,
            epochs_bfgs=args.epochs_bfgs,
            bfgs_steps_per_epoch=args.bfgs_steps_per_epoch,
            n_f=args.n_f,
            n_data=args.n_data,
            lr_adam=args.lr_adam,
            gtol_bfgs=args.gtol_bfgs,
        )

    visualize_saved_re60(
        save_dir=args.save_dir,
        device=device,
        out_dir=args.viz_dir,
        width=args.width,
        depth=args.depth,
    )


if __name__ == "__main__":
    main()
