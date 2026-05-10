#!/bin/bash
#SBATCH -J pinn_re40_v2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -o pinn_re40_v2_%j.out
#SBATCH -e pinn_re40_v2_%j.err

# ================================================================
# v2 setup, two-stage training.
#
#   Hard BCs  : no-slip on cylinder + free-stream on inlet/top/bottom
#               are encoded in the network (see cfp40_v2.MLPStreamPressureHardBC).
#               Their losses are numerically zero by construction.
#   Loader    : reads UMean / pMean (steady time-average), NOT 'U' snapshot.
#   Box       : (-8,12) × (-8,8), where the CFD's free stream actually holds
#               (sanity_check_re40.py confirmed rmse(u-1) ≈ 1.4e-3 at x=-8).
#   Weights   : Adam = AdaptiveLossWeightsV2 ; BFGS = auto-normalized from
#               the Adam EMA so SSBroyden sees ~unit-magnitude objectives
#               and self-preconditions (mentor's "二阶法不需要调权重").
#
# Stage A1 (data-anchored)
#   200-epoch sanity showed the network falls into the trivial
#   u=1 / v=0 attractor (wake never develops). A small CFD anchor
#   (--use-data --n-data 5000) gives it a soft pull out of that basin.
#
# Stage A2 (data-free refine)
#   Warm-start from A1 weights, drop --use-data, let pure physics refine.
#   This is the "PINN reproduces the field on its own" test.
# ================================================================

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn

cd "/oscar/home/jchen790/cylinder flow lab"

# ---------- model size ----------
# Width=96, depth=5 -> ~38k params -> H0 (full-matrix BFGS) ~12 GB.
# This comfortably fits 64 GB even with SSBroyden's 6-7x transient peaks.
# Bump to 128 / 6 only if you also bump --mem to ≥128G in the SBATCH header.
WIDTH=96
DEPTH=5

# ---------- shared training args ----------
ITERS_PER_BATCH=150
N_F=50000
N_DATA=5000

# ---------- Fourier features ----------
FOURIER_F=32
FOURIER_SIGMA=2.0

# ---------- box (matches the sanity-check verdict) ----------
XMIN=-8.0
XMAX=12.0
YMIN=-8.0
YMAX=8.0

# =================================================================
# Stage A1 — Adam + BFGS  WITH a small CFD anchor
# =================================================================
echo "============================================================"
echo " Stage A1 — data-anchored (--use-data --n-data $N_DATA)"
echo "   goal: break the trivial-solution (u=1,v=0) attractor"
echo "============================================================"

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir checkpoints_v2_A1 \
    --viz-dir  viz_v2_A1 \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 4000 \
    --maxiter-bfgs 8000 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f $N_F \
    --n-data $N_DATA \
    --use-data \
    --fourier-features $FOURIER_F --fourier-sigma $FOURIER_SIGMA \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1

# =================================================================
# Stage A2 — resume A1, drop the data anchor, refine on PURE PHYSICS
# =================================================================
echo "============================================================"
echo " Stage A2 — data-free refinement (warm-start from A1)"
echo "   goal: the PINN should HOLD the field with no CFD nudging"
echo "============================================================"

if [[ ! -s checkpoints_v2_A1/pinn_Re40_single.pt ]]; then
    echo "[A2] A1 checkpoint missing or empty — aborting A2."
    exit 1
fi

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir checkpoints_v2_A2 \
    --viz-dir  viz_v2_A2 \
    --resume-from checkpoints_v2_A1/pinn_Re40_single.pt \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 1000 \
    --maxiter-bfgs 8000 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f $N_F \
    --fourier-features $FOURIER_F --fourier-sigma $FOURIER_SIGMA \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1
    # NOTE: NO --use-data here.

# =================================================================
# Summary
# =================================================================
echo "============================================================"
echo " Done. Compare:"
echo "   viz_v2_A1/  (with data anchor — should already show wake)"
echo "   viz_v2_A2/  (data-free refine — wake should survive)"
echo ""
echo " Look for:"
echo "   v2_speed.png      : recirculation bubble behind cylinder"
echo "   v2_v.png          : antisymmetric ± lobes in the wake"
echo "   v2_vorticity.png  : two parallel vorticity bands, peak ~±8"
echo "============================================================"
