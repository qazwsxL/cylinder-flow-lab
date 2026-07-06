#!/bin/bash
#SBATCH -J pinn_re40_consist_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 04:00:00
#SBATCH -o logs/pinn_re40_consist_smoke_%j.out
#SBATCH -e logs/pinn_re40_consist_smoke_%j.err

# ================================================================
# Smoke test of mentor's strict consistency protocol
# ================================================================
#
# PROTOCOL — identical to train_v2_consistency.sh:
#   Phase 1: pure data fit on ALL CFD points (no PDE, no outlet)
#   Phase 2: warm-start, drop data entirely, PDE only with n_f=20000
#
# WHAT'S DIFFERENT (smoke knobs):
#   - smaller network (width=64, depth=4 → ~17k params, H0 ~2.3 GB)
#   - fewer Adam epochs + BFGS iters, but enough that BFGS plateaus
#   - 4 h walltime instead of 24 h
#   - 32 GB memory instead of 128 GB (small network → small Hessian)
#
# DECISION RULE:
#   - If smoke says CONSISTENT (PDE drops AND mae_u stays within ~2× of P1)
#     → re-run with full setup (width=96, depth=5, longer training) to
#       confirm with publication-quality numbers.
#   - If smoke says NOT consistent → matches our earlier full-size finding
#     and confirms the result is architecture-robust, not a P1-undertraining
#     artifact.
# ================================================================

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn

cd "/oscar/home/jchen790/cylinder flow lab"

# ----- shrunk network -----
WIDTH=64
DEPTH=4

# ----- shrunk training -----
ITERS_PER_BATCH=80
FOURIER_F=16
FOURIER_SIGMAS="0.25 0.5 1.0 2.0"

# ----- box (unchanged — uses CFD's actual free-stream region) -----
XMIN=-8.0
XMAX=12.0
YMIN=-8.0
YMAX=8.0

mkdir -p runs/v2_consistency_smoke/P1 runs/v2_consistency_smoke/P2 logs

# =================================================================
# Phase 1 — pure data fit (interpolator), all CFD pts
# =================================================================
echo "============================================================"
echo " [SMOKE] Phase 1 — DATA-ONLY (all CFD pts, no PDE, no outlet)"
echo "============================================================"

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir runs/v2_consistency_smoke/P1/checkpoints \
    --viz-dir  runs/v2_consistency_smoke/P1/viz \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 1500 \
    --maxiter-bfgs 1200 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 0 \
    --data-only \
    --use-all-cfd-data \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1

# =================================================================
# Phase 2 — warm-start, drop data, PDE only with n_f=20000
# =================================================================
echo "============================================================"
echo " [SMOKE] Phase 2 — PDE-ONLY  (n_f=20000, NO data)"
echo "============================================================"

P1_CKPT=runs/v2_consistency_smoke/P1/checkpoints/pinn_Re40_single.pt
if [[ ! -s "$P1_CKPT" ]]; then
    echo "[P2] Phase-1 checkpoint missing or empty ($P1_CKPT) — aborting."
    exit 1
fi

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir runs/v2_consistency_smoke/P2/checkpoints \
    --viz-dir  runs/v2_consistency_smoke/P2/viz \
    --resume-from "$P1_CKPT" \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 500 \
    --maxiter-bfgs 1500 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 20000 \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1
    # NOTE: NO --use-data, NO --data-only.

echo "============================================================"
echo " SMOKE TEST DONE. Look at:"
echo ""
echo "   runs/v2_consistency_smoke/P1/  ABSOLUTE error"
echo "   runs/v2_consistency_smoke/P2/  ABSOLUTE error"
echo ""
echo " Then:"
echo "   - If P2 mae_u within ~2× of P1 → CONSISTENT (smoke positive)"
echo "       → re-run with full setup (sbatch train_v2_consistency.sh after"
echo "         fixing the cfp40_v2_consistency.py path), expect same trend"
echo "         at higher precision."
echo "   - If P2 mae_u >> P1 mae_u  → NOT CONSISTENT (matches prior result)"
echo "       → conclude the v2 setup has a trivial-attractor problem"
echo "         independent of model capacity. Run train_v2_weakanchor.sh"
echo "         to test whether a weak data prior is enough to fix it."
echo "============================================================"
