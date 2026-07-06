#!/bin/bash
#SBATCH -J pinn_re40_v2
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH -t 24:00:00
# NOTE: bumped 64G -> 128G after job 2485074 OOM-killed during SSBroyden.
# At width=96 depth=5 (~38k params), H0 is 11 GB but the SSBroyden update
# allocates 6-7 transient N×N matrices in one numpy expression, peaking
# at ~70-80 GB. 64 GB was insufficient. 128G has comfortable headroom.
#SBATCH -o logs/pinn_re40_v2_%j.out
#SBATCH -e logs/pinn_re40_v2_%j.err

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
# Width=96, depth=5 -> ~38k params -> H0 (full-matrix BFGS) ~11 GB,
# SSBroyden transient peak ~70-80 GB. With --mem=128G we have comfortable
# headroom. Bump to 128/6 (~91k params, H0 ~62 GB) only if you also bump
# --mem to ≥256G.
WIDTH=96
DEPTH=5

# ---------- shared training args ----------
ITERS_PER_BATCH=150
N_F=50000
# n_data bumped 5000 -> 10000. Combined with the new AdaptiveLossWeightsV2
# data-base of 5.0 + auto_normalize_from_ema's data_priority of 5x, the
# effective data weight is now ~25x what the previous run had. That's what
# the previous run was missing — it stayed in the trivial u=1 attractor
# because the data anchor couldn't outpull PDE.
N_DATA=10000

# ---------- Fourier features (multi-scale, low-freq biased) ----------
# Run 2533795 with σ ∈ {0.5,1,2,4} produced a wake (good!) but with heavy
# high-freq noise from σ=4 and a wake too short (only to x≈2 vs CFD's x≈4).
# Shift the band centers DOWN: drop σ=4, add σ=0.25.
#   σ=0.25 → λ≈4   wake extent  (NEW)
#   σ=0.5  → λ≈2   recirculation bubble
#   σ=1    → λ≈1   medium structure
#   σ=2    → λ≈0.5 boundary-layer, cylinder curvature
# This gives the network a basis biased toward the actual flow scales and
# removes the σ=4 source of background noise.
FOURIER_F=32
FOURIER_SIGMAS="0.25 0.5 1.0 2.0"

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

mkdir -p runs/v2_two_stage/A1 runs/v2_two_stage/A2 logs

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir runs/v2_two_stage/A1/checkpoints \
    --viz-dir  runs/v2_two_stage/A1/viz \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 3000 \
    --maxiter-bfgs 2500 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f $N_F \
    --n-data $N_DATA \
    --use-data \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1

# =================================================================
# Stage A2 — resume A1, drop the data anchor, refine on PURE PHYSICS
# =================================================================
echo "============================================================"
echo " Stage A2 — data-free refinement (warm-start from A1)"
echo "   goal: the PINN should HOLD the field with no CFD nudging"
echo "============================================================"

A1_CKPT=runs/v2_two_stage/A1/checkpoints/pinn_Re40_single.pt
if [[ ! -s "$A1_CKPT" ]]; then
    echo "[A2] A1 checkpoint missing or empty ($A1_CKPT) — aborting A2."
    exit 1
fi

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir runs/v2_two_stage/A2/checkpoints \
    --viz-dir  runs/v2_two_stage/A2/viz \
    --resume-from "$A1_CKPT" \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 1000 \
    --maxiter-bfgs 4000 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f $N_F \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1
    # NOTE: NO --use-data here.

# =================================================================
# Summary
# =================================================================
echo "============================================================"
echo " Done. Compare:"
echo "   runs/v2_two_stage/A1/viz/  (with data anchor — should already show wake)"
echo "   runs/v2_two_stage/A2/viz/  (data-free refine — wake should survive)"
echo ""
echo " Look for:"
echo "   v2_speed.png      : recirculation bubble behind cylinder"
echo "   v2_v.png          : antisymmetric ± lobes in the wake"
echo "   v2_vorticity.png  : two parallel vorticity bands, peak ~±8"
echo "============================================================"
