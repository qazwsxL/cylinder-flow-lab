#!/bin/bash
#SBATCH -J pinn_re40_2phase
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -o pinn_re40_2phase_%j.out
#SBATCH -e pinn_re40_2phase_%j.err

# ================================================================
# Mentor's two-phase strategy for Re=40 cylinder PINN.
#
# Phase 1  (data-only)
#   Skip the PDE residual entirely. Train the network as a smooth
#   *interpolator* of every CFD point. This isolates "how well can we fit
#   the CFD field if we don't care about NS at all".
#
# Phase 2  (resume + turn PDE on)
#   Load Phase-1 weights and now require the PDE to also be satisfied.
#   The DROP in PDE residual during Phase 2 measures how badly the data-fit
#   field violates Navier-Stokes — i.e. how *consistent* the CFD is with
#   our NS setup. If PDE residual stays small, CFD ↔ setup are consistent;
#   if it has to fight the data hard, they are not.
#
# Mentor's iter recipe: keep iters_per_batch around 100-200, then resample.
# That's what `--no-frozen-colloc-bfgs` + small iters_per_batch buys you.
# ================================================================

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn

cd "/oscar/home/jchen790/cylinder flow lab"

WIDTH=96
DEPTH=5
ITERS_PER_BATCH=150          # mentor: 100-200 then resample

# ---------------------------------------------------------------- Phase 1
echo "============================================================"
echo " Phase 1 — data-only fit on ALL CFD points (no PDE)"
echo "============================================================"

python -u cfp40.py --vtk-path Re40.vtk \
    --save-dir checkpoints_phase1 \
    --viz-dir  viz_phase1 \
    --width $WIDTH --depth $DEPTH \
    --data-only \
    --use-all-cfd-data \
    --epochs-adam 1000 \
    --maxiter-bfgs 1500 \
    --iters-per-batch $ITERS_PER_BATCH \
    --no-frozen-colloc-bfgs \
    --method-bfgs SSBroyden1

# Run the analysis ipynb's checks on the Phase-1 checkpoint. This produces
# both the velocity error map (how well does it fit data) AND the PDE
# residual map (how badly does the data-fit field violate NS — that's the
# CFD-consistency answer).
python -u run_analysis_re40.py \
    --vtk-path Re40.vtk \
    --save-dir checkpoints_phase1 \
    --out-dir  analysis_phase1 \
    --width $WIDTH --depth $DEPTH

# ---------------------------------------------------------------- Phase 2
echo "============================================================"
echo " Phase 2 — turn PDE back on, see if it agrees with the data fit"
echo "============================================================"

python -u cfp40.py --vtk-path Re40.vtk \
    --save-dir checkpoints_phase2 \
    --viz-dir  viz_phase2 \
    --width $WIDTH --depth $DEPTH \
    --resume-from checkpoints_phase1/pinn_Re40_single.pt \
    --use-all-cfd-data \
    --epochs-adam 200 \
    --maxiter-bfgs 1500 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 50000 \
    --n-cfd-pde 12000 --lambda-pde-cfd 1.0 \
    --cfd-pde-wall-buffer 0.5 --cfd-pde-edge-buffer 0.5 \
    --no-frozen-colloc-bfgs \
    --method-bfgs SSBroyden1

python -u run_analysis_re40.py \
    --vtk-path Re40.vtk \
    --save-dir checkpoints_phase2 \
    --out-dir  analysis_phase2 \
    --width $WIDTH --depth $DEPTH

# ---------------------------------------------------------------- Summary
echo "============================================================"
echo " Done.  Compare:"
echo "   analysis_phase1/summary.txt   (data-only fit, PDE residual = consistency)"
echo "   analysis_phase2/summary.txt   (with PDE on, full PINN final)"
echo "============================================================"
