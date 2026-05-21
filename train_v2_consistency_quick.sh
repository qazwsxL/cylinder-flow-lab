#!/bin/bash
#SBATCH -J pinn_re40_consist_quick
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -t 01:00:00
#SBATCH -o logs/pinn_re40_consist_quick_%j.out
#SBATCH -e logs/pinn_re40_consist_quick_%j.err

# ================================================================
# QUICK smoke test (~20 min total) of mentor's consistency protocol
# ================================================================
#
# Protocol identical to train_v2_consistency.sh:
#   Phase 1: pure data fit on ALL CFD points (no PDE, no outlet)
#   Phase 2: warm-start, drop data entirely, PDE only with n_f=20000
#
# Aggressively shrunk for fast turnaround:
#   width=32 depth=3      → ~3k params (H0 ~75 MB)
#   F=8 Fourier (4 bands × 2 rows)
#   ~300+200 Adam epochs, ~200+300 BFGS iters
#   1h walltime, 16G mem
#
# Use this to *quickly* decide whether the consistency conclusion is
# architecture-robust before committing to multi-hour runs. If P2 mae_u
# still blows up here (likely), the trivial-attractor finding is real.
# If P2 holds, scale up: train_v2_consistency_smoke.sh or full version.
# ================================================================

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn

cd "/oscar/home/jchen790/cylinder flow lab"

WIDTH=32
DEPTH=3
ITERS_PER_BATCH=50
FOURIER_F=8
FOURIER_SIGMAS="0.25 0.5 1.0 2.0"
XMIN=-8.0
XMAX=12.0
YMIN=-8.0
YMAX=8.0

mkdir -p runs/v2_consistency_quick/P1 runs/v2_consistency_quick/P2 logs

# =================================================================
# Phase 1 — pure data fit on ALL CFD pts
# =================================================================
echo "============================================================"
echo " [QUICK] Phase 1 — DATA-ONLY (all CFD pts, no PDE, no outlet)"
echo "============================================================"
T0=$(date +%s)

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir runs/v2_consistency_quick/P1/checkpoints \
    --viz-dir  runs/v2_consistency_quick/P1/viz \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 300 \
    --maxiter-bfgs 200 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 0 \
    --data-only \
    --use-all-cfd-data \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1

T1=$(date +%s); echo "[timing] Phase 1 took $((T1-T0)) s"

# =================================================================
# Phase 2 — warm-start, drop data, PDE only with n_f=20000
# =================================================================
echo "============================================================"
echo " [QUICK] Phase 2 — PDE-ONLY  (n_f=20000, NO data)"
echo "============================================================"

P1_CKPT=runs/v2_consistency_quick/P1/checkpoints/pinn_Re40_single.pt
if [[ ! -s "$P1_CKPT" ]]; then
    echo "[P2] Phase-1 checkpoint missing or empty ($P1_CKPT) — aborting."
    exit 1
fi

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir runs/v2_consistency_quick/P2/checkpoints \
    --viz-dir  runs/v2_consistency_quick/P2/viz \
    --resume-from "$P1_CKPT" \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 200 \
    --maxiter-bfgs 300 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 20000 \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1
    # NO --use-data, NO --data-only

T2=$(date +%s); echo "[timing] Phase 2 took $((T2-T1)) s    (total $((T2-T0)) s)"

echo "============================================================"
echo " QUICK SMOKE DONE."
echo "   runs/v2_consistency_quick/P1/  — data-fit endpoint"
echo "   runs/v2_consistency_quick/P2/  — after dropping data"
echo " Compare the two ABSOLUTE error blocks above to judge consistency."
echo "============================================================"
