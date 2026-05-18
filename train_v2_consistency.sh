#!/bin/bash
#SBATCH -J pinn_re40_consist
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -o logs/pinn_re40_consist_%j.out
#SBATCH -e logs/pinn_re40_consist_%j.err

# ================================================================
# Mentor's strict consistency protocol
# ================================================================
#
# Goal: TEST whether the CFD field is consistent with our v2 PINN's
#       NS + hard-BC + multi-scale-Fourier formulation, using interpretation
#       (Y) — i.e. PDE loss should drop AND velocity field should stay close
#       to CFD when data is removed.
#
# Phase 1  (--data-only --use-all-cfd-data)
#   Use ALL 7456 CFD cell-centres as data points. PDE loss is OFF. Outlet
#   loss is OFF. The network is trained as a pure interpolator of CFD.
#   Output checkpoint = "best possible CFD reproduction with the v2 model class".
#
# Phase 2  (resume + NO data, only PDE with n_f=20000)
#   Warm-start from Phase 1. Drop the data anchor entirely. Drive only the
#   PDE residual on a fresh 20000-point collocation set.
#
#   Success criterion (mentor's "Y"):
#     - PDE loss DROPS substantially from its Phase-1-endpoint value, AND
#     - mae_u / mae_v stay within a factor of ~2 of Phase-1 values.
#
#   Failure mode (what our previous A1→A2 run showed):
#     - PDE drops AND mae_u/mae_v double or worse → network slid to a
#       PDE-minimizing-but-physically-wrong local minimum. The CFD field
#       is NOT consistent with the v2 NS setup (within this model class).
#
# Architecture: identical for both phases — v2 hard-BC + multi-scale Fourier
# (σ ∈ {0.25, 0.5, 1.0, 2.0}), box (-8,12)×(-8,8), UMean / pMean from VTK.
# ================================================================

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn

cd "/oscar/home/jchen790/cylinder flow lab"

WIDTH=96
DEPTH=5
ITERS_PER_BATCH=150
FOURIER_F=32
FOURIER_SIGMAS="0.25 0.5 1.0 2.0"
XMIN=-8.0
XMAX=12.0
YMIN=-8.0
YMAX=8.0

mkdir -p runs/v2_consistency/P1 runs/v2_consistency/P2 logs

# =================================================================
# Phase 1 — pure data fit (interpolator)
# =================================================================
echo "============================================================"
echo " Phase 1 — DATA-ONLY (all CFD pts, no PDE, no outlet)"
echo "   goal: best possible CFD reproduction by v2 model class"
echo "============================================================"

python -u cfp40_v2_consistency.py --vtk-path Re40.vtk \
    --save-dir runs/v2_consistency/P1/checkpoints \
    --viz-dir  runs/v2_consistency/P1/viz \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 3000 \
    --maxiter-bfgs 2500 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 0 \
    --data-only \
    --use-all-cfd-data \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1 \
    --cfd-monitor-every 100

# =================================================================
# Phase 2 — warm-start, drop data, drive PDE only
# =================================================================
echo "============================================================"
echo " Phase 2 — PDE-ONLY with n_f=20000 (NO data anchor)"
echo "   warm-start from Phase 1; this IS the consistency test"
echo "============================================================"

P1_CKPT=runs/v2_consistency/P1/checkpoints/pinn_Re40_single.pt
if [[ ! -s "$P1_CKPT" ]]; then
    echo "[P2] Phase-1 checkpoint missing or empty ($P1_CKPT) — aborting."
    exit 1
fi

python -u cfp40_v2_consistency.py --vtk-path Re40.vtk \
    --save-dir runs/v2_consistency/P2/checkpoints \
    --viz-dir  runs/v2_consistency/P2/viz \
    --resume-from "$P1_CKPT" \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 1000 \
    --maxiter-bfgs 4000 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 20000 \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1 \
    --cfd-monitor-every 50
    # NOTE: NO --use-data, NO --data-only.
    # The default is data weight = 0, PDE + outlet only → mentor's Phase 2.

echo "============================================================"
echo " Consistency test done. The decisive numbers are:"
echo ""
echo "   runs/v2_consistency/P1/  ABSOLUTE error  =  data-fit endpoint"
echo "   runs/v2_consistency/P2/  ABSOLUTE error  =  after dropping data"
echo ""
echo " Look in the .out for ABSOLUTE error blocks and compare:"
echo "   - PDE loss at end of P1 (it will be evaluated by P2's first call)"
echo "   - PDE loss at end of P2 (should be MUCH lower)"
echo "   - mae_u / mae_v (should stay close to P1 values to count as consistent)"
echo "============================================================"
