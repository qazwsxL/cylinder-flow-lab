#!/bin/bash
#SBATCH -J pinn_re40_weak
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -o logs/pinn_re40_weak_%j.out
#SBATCH -e logs/pinn_re40_weak_%j.err

# ================================================================
# CONTROL EXPERIMENT — Weak data anchor in Phase 2
# ================================================================
#
# Companion to train_v2_consistency.sh.
#
# The strict (Y) test (data completely off in P2) showed:
#   - P1 data fit was essentially perfect (mae_u ~ 5e-4)
#   - P2 with PDE-only collapsed: PDE dropped 5900× but DATA exploded to ~0.32
#   - Wake disappeared in P2 viz
#
# Mentor's verdict on (Y): NOT consistent. But there are two confounded
# explanations for that:
#   (a) the CFD field really is incompatible with our PDE solution space
#       (trivial attractor)
#   (b) the network's P1 fit was a noisy interpolation (vorticity ±50 vs
#       CFD ±8) so P2 was smoothing AWAY a bad starting state
#
# This control disambiguates (a) vs (b):
#   Phase 1: identical to consistency P1 (pure data fit, all CFD pts).
#   Phase 2: warm-start, n_f=20000 PDE, but ALSO keep n_data=1000 with
#            --data-priority 0.5. PDE dominates the BFGS objective; data
#            is a weak regularizer just to prevent drift.
#
# Decision rule:
#   - If P2 mae_u stays within ~2× of P1 (≲ 1e-3) AND PDE drops a lot:
#       trivial attractor is overcome by ANY small data prior; the v2 setup
#       IS practically consistent given a whisper of data.
#   - If P2 still collapses (mae_u > 1e-2):
#       trivial attractor is fundamental to this PDE solution space; need
#       stronger physical priors (larger model, Re curriculum, etc.).
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

# weak-anchor knobs (P2 only)
P2_N_DATA=1000           # ~13% of full CFD point count
P2_DATA_PRIORITY=0.5     # << 1 so PDE dominates; data just regularises

mkdir -p runs/v2_weakanchor/P1 runs/v2_weakanchor/P2 logs

# =================================================================
# Phase 1 — pure data fit  (same protocol as consistency P1)
# =================================================================
echo "============================================================"
echo " Phase 1 — DATA-ONLY (all CFD pts, no PDE, no outlet)"
echo "============================================================"

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir runs/v2_weakanchor/P1/checkpoints \
    --viz-dir  runs/v2_weakanchor/P1/viz \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 3000 \
    --maxiter-bfgs 2500 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 0 \
    --data-only \
    --use-all-cfd-data \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1

# =================================================================
# Phase 2 — warm-start + PDE primary + WEAK data anchor
# =================================================================
echo "============================================================"
echo " Phase 2 — PDE-PRIMARY with WEAK data anchor"
echo "   n_f=20000  +  n_data=$P2_N_DATA  +  data_priority=$P2_DATA_PRIORITY"
echo "============================================================"

P1_CKPT=runs/v2_weakanchor/P1/checkpoints/pinn_Re40_single.pt
if [[ ! -s "$P1_CKPT" ]]; then
    echo "[P2] Phase-1 checkpoint missing or empty ($P1_CKPT) — aborting."
    exit 1
fi

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir runs/v2_weakanchor/P2/checkpoints \
    --viz-dir  runs/v2_weakanchor/P2/viz \
    --resume-from "$P1_CKPT" \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 1000 \
    --maxiter-bfgs 4000 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 20000 \
    --use-data \
    --n-data $P2_N_DATA \
    --data-priority $P2_DATA_PRIORITY \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1

echo "============================================================"
echo " Weak-anchor control done. Compare against consistency run:"
echo ""
echo "   runs/v2_consistency/P2/  (data=0,       PDE-only)"
echo "   runs/v2_weakanchor/P2/   (n_data=1000,  data_priority=0.5)"
echo ""
echo " Decision:"
echo "   - mae_u in weakanchor P2 ≲ 1e-3 → trivial attractor is fragile,"
echo "     v2 is practically consistent given a tiny data prior."
echo "   - mae_u still >> 1e-3        → trivial attractor is fundamental,"
echo "     need bigger model / Re curriculum / stronger physical priors."
echo "============================================================"
