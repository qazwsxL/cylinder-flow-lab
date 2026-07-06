# cylinder-flow-lab

PINN study of 2D cylinder flow at Re=40, with Re=60 reference. Working toward
"a PINN setup that REPRODUCES the cylinder flow field" rather than just
fitting the CFD.

## Layout

```
cylinder-flow-lab/
├── README.md
├── Re40.vtk                         primary CFD reference (UMean / pMean)
├── Re60/                            Re=60 vtks + v_t.csv + README
│
├── src/                             Python source files
│   ├── cfp40.py                     Re=40 trainer — PRIMARY (--use-pt flag)
│   ├── cfp_pt.py                    PT reference impl for Re=60 curriculum
│   ├── cfp.py                       Re=5→60 curriculum trainer
│   ├── cfp40_v2.py                  Re=40 v2: hard BCs + Fourier features
│   ├── cfp60.py                     Re=60 variant
│   ├── _optimize.py                 custom SSBroyden / SSBFGS BFGS variants
│   ├── plot_mentor_style.py         mentor's 6-panel figure style
│   ├── diagnose_cfd_pde.py          CFD-vs-PDE residual diagnostic
│   ├── run_analysis_re40.py         post-training analysis
│   └── sanity_check_re40.py         pre-training CFD-PINN consistency check
│
├── scripts/                         SLURM / run scripts
│   ├── run_pt_oscar.sh              ← current: PT job on Oscar
│   ├── run_pt_local.sh              ← current: PT job on local Linux
│   ├── train_two_phase.sh
│   ├── train_v2_two_stage.sh
│   └── ...
│
├── notebooks/
│   ├── analysis_updated_re40_clean.ipynb
│   └── re40_domain_residual_check.ipynb
│
├── runs/                            experiment outputs
│   ├── pt/                          pseudo-time stepping runs (checkpoints + viz)
│   ├── re40_single/
│   ├── phase1/, phase2/
│   ├── v2_two_stage/
│   └── ...
│
├── reports/                         pptx slides + figs
├── logs/                            SLURM .out / .err
└── archive/                         retired checkpoints / old code
```

## Current focus — pseudo-time stepping (Wang et al. 2025)

`src/cfp40.py` supports `--use-pt` to avoid spurious solutions.
Outputs land in `runs/pt/`.

```bash
# Oscar
sbatch scripts/run_pt_oscar.sh

# Local Linux
bash scripts/run_pt_local.sh
```

The PT weight `w` is logged every 10 BFGS batches as `[PT] batch=... w=...`.

## Running from src/

All Python files live in `src/`; run them from there so `_optimize.py` is importable:

```bash
cd src/
python sanity_check_re40.py --vtk-path ../Re40.vtk ...
python cfp40.py --vtk-path ../Re40.vtk --save-dir ../runs/... ...
```

## Key v2 design choices (vs cfp40.py)

- **Hard BCs**: no-slip on cylinder + free-stream (u=1, v=0) on
  inlet/top/bottom encoded structurally via stream-function lifting. Their
  losses are numerically zero by construction (~1e-13).
- **Fourier features**: 32 random spatial features at σ=2 to fight spectral
  bias on the wake / shear layer.
- **Pressure anchored** at outlet (p=0).
- **VTK loader reads UMean / pMean** by default (not the snapshot `U`, which
  carries phase / fluctuation that a steady PINN cannot fit).
- **Box (-8,12)×(-8,8)** so the hard inlet is in CFD's actual free-stream
  region (sanity_check_v2 confirmed rmse(u-1) = 1.4e-3 there vs 3.1e-2 at
  the old x=-3 inlet).
- **Auto-normalized BFGS weights**: each loss term enters BFGS at unit
  magnitude (1 / Adam-EMA), so SSBroyden self-preconditions — no hand-tuning.
