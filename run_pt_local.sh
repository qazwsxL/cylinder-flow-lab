#!/bin/bash
# Run cfp40.py with pseudo-time stepping on a local Linux desktop.
# Usage: bash run_pt_local.sh

set -e

# ---- Adjust these to your setup ----
CONDA_ENV="pinn"
WORK_DIR="$HOME/cylinder-flow-lab"   # or wherever the repo lives
VTK_PATH="Re40.vtk"                  # relative to WORK_DIR
# -------------------------------------

# Activate conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$WORK_DIR"

python -u cfp40.py \
    --vtk-path "$VTK_PATH" \
    --save-dir checkpoints_pt \
    --viz-dir  viz_pt \
    --width 96 --depth 5 \
    --epochs-adam 2000 \
    --maxiter-bfgs 6000 --iters-per-batch 150 \
    --n-f 30000 --n-data 10000 \
    --n-cfd-pde 8000 --lambda-pde-cfd 1.0 \
    --cfd-pde-wall-buffer 0.5 --cfd-pde-edge-buffer 0.5 \
    --method-bfgs SSBroyden1 \
    --use-pt --pt-w-init 1.0 --pt-ema 0.9 \
    2>&1 | tee logs_pt_$(date +%Y%m%d_%H%M%S).txt
