# cylinder-flow-lab

PINN study of 2D cylinder flow at Re=40, with Re=60 reference. Working toward
"a PINN setup that REPRODUCES the cylinder flow field" rather than just
fitting the CFD.

## Layout

```
cylinder-flow-lab/
├── README.md
│
├── Source code (root)
│   ├── cfp.py                       earlier general PINN
│   ├── cfp40.py                     Re=40 trainer (instantaneous-U loader, soft BCs)
│   ├── cfp40_v2.py                  Re=40 v2: hard BCs + Fourier features +
│   │                                auto-normalized BFGS weights, reads UMean
│   ├── cfp60.py                     Re=60 variant
│   ├── _optimize.py                 custom SSBroyden / SSBFGS BFGS variants
│   ├── diagnose_cfd_pde.py          CFD-vs-PDE residual diagnostic
│   ├── run_analysis_re40.py         post-training analysis (used by phase1/phase2)
│   └── sanity_check_re40.py         pre-training CFD-PINN consistency check
│
├── Notebooks (root)
│   ├── analysis_updated_re40_clean.ipynb
│   └── re40_domain_residual_check.ipynb
│
├── Scripts (root)
│   ├── run_pinn.sh
│   ├── train_two_phase.sh           original 2-phase: data-only -> data+PDE
│   └── train_v2_two_stage.sh        v2 2-stage: data-anchored -> data-free refine
│
├── Data (root)
│   ├── Re40.vtk                     primary reference (UMean / pMean inside)
│   └── Re60/                        Re=60 vtks + summary CSV
│
├── runs/                            all experiment outputs, organized by experiment
│   ├── re_curriculum/checkpoints/   Re10..Re45 curriculum ckpts
│   ├── re40_single/                 single-snapshot Re=40 baseline (ckpt + viz)
│   ├── cfd_pde/                     CFD-aware PDE residual experiments
│   ├── phase1/                      data-only fit (interpolator)
│   │   ├── checkpoints_phase1/
│   │   ├── analysis_phase1/
│   │   └── analysis_runA/
│   ├── phase2/                      phase1 + PDE turned back on
│   ├── sanity_checks/               sanity_check_re40.py outputs
│   │   ├── sanity_check_re40/       (old box -3..12 / -4..4)
│   │   └── sanity_check_re40_v2/    (new box -8..12 / -8..8)
│   ├── v2_smoke/                    200-epoch v2 smoke test (May 10)
│   └── v2_two_stage/                v2 full two-stage run (created by sbatch)
│       ├── A1/{checkpoints,viz}/    data-anchored
│       └── A2/{checkpoints,viz}/    data-free refine
│
├── logs/                            slurm .out / .err / .log
├── reports/                         pptx + 汇报稿
├── archive/                         retired smoke / sanity / wake / duplicate _optimize
└── .git/, .gitignore
```

## Sequence of work

1. `python sanity_check_re40.py --vtk-path Re40.vtk --x-min -8 --x-max 12 --y-min -8 --y-max 8`
   — confirm CFD ↔ PINN setup consistency. Outputs go to
   `runs/sanity_checks/sanity_check_re40_v2/`.
2. `sbatch train_v2_two_stage.sh` — run A1 (data-anchored) then A2 (data-free
   refine). Outputs in `runs/v2_two_stage/`.
3. Inspect `runs/v2_two_stage/A2/viz/v2_vorticity.png` — wake should show two
   parallel vorticity bands at peak ~±8 (matching CFD's `UMean`).

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
