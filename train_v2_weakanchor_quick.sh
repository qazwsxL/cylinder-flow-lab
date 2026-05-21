#!/bin/bash
#SBATCH -J pinn_re40_weak_quick
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -t 01:00:00
#SBATCH -o logs/pinn_re40_weak_quick_%j.out
#SBATCH -e logs/pinn_re40_weak_quick_%j.err

# ================================================================
# QUICK weak-anchor control (~10 min) — companion to consistency_quick
# ================================================================
#
# Same shrunk setup as train_v2_consistency_quick.sh, but P2 keeps a
# WEAK data prior. This isolates: did the consistency_quick failure come
# from (a) the trivial-attractor problem, or (b) zero-data underconstraint?
#
#   Phase 1 (unchanged from consistency_quick):
#     pure data fit, all CFD pts, no PDE
#
#   Phase 2 (weak anchor):
#     warm-start P1, n_f=20000 PDE, n_data=500 (~7% of CFD), data_priority=0.5
#     PDE dominates the BFGS objective; data is a whisper-level regulariser.
#
# Decision rule:
#   - P2 mae_u ≲ P1 mae_u (~0.1) → weak anchor IS enough; collapse was about
#     zero-data, not fundamental trivial-attractor.
#   - P2 mae_u >> P1 mae_u        → even a weak anchor can't pull the
#     optimizer out of the wrong PDE minimum; trivial attractor is real,
#     the v2 setup needs more (bigger model / Re curriculum / different
#     physical prior) to be consistent at Re=40.
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

# weak-anchor knobs (P2 only)
P2_N_DATA=500
P2_DATA_PRIORITY=0.5

mkdir -p runs/v2_weakanchor_quick/P1 runs/v2_weakanchor_quick/P2 logs

# =================================================================
# Phase 1 — pure data fit (identical to consistency_quick P1)
# =================================================================
echo "============================================================"
echo " [WEAK-QUICK] Phase 1 — DATA-ONLY (all CFD pts)"
echo "============================================================"
T0=$(date +%s)

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir runs/v2_weakanchor_quick/P1/checkpoints \
    --viz-dir  runs/v2_weakanchor_quick/P1/viz \
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
# Phase 2 — PDE main + WEAK data prior
# =================================================================
echo "============================================================"
echo " [WEAK-QUICK] Phase 2 — PDE-PRIMARY  (n_f=20000, n_data=$P2_N_DATA, data_priority=$P2_DATA_PRIORITY)"
echo "============================================================"

P1_CKPT=runs/v2_weakanchor_quick/P1/checkpoints/pinn_Re40_single.pt
if [[ ! -s "$P1_CKPT" ]]; then
    echo "[P2] Phase-1 checkpoint missing or empty ($P1_CKPT) — aborting."
    exit 1
fi

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir runs/v2_weakanchor_quick/P2/checkpoints \
    --viz-dir  runs/v2_weakanchor_quick/P2/viz \
    --resume-from "$P1_CKPT" \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 200 \
    --maxiter-bfgs 300 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 20000 \
    --use-data \
    --n-data $P2_N_DATA \
    --data-priority $P2_DATA_PRIORITY \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1

T2=$(date +%s); echo "[timing] Phase 2 took $((T2-T1)) s    (total $((T2-T0)) s)"

echo "============================================================"
echo " WEAK-ANCHOR QUICK DONE. Compare three quick runs:"
echo ""
echo "   consistency_quick P2 (no data)            : trivial-attractor unmasked"
echo "   weakanchor_quick  P2 (500 pts, prior 0.5) : THIS RUN"
echo "   weakanchor_quick  P1 (pure data fit)      : sets the floor"
echo ""
echo " If P2 mae_u in this run is close to P1 → weak anchor cures it."
echo " If P2 mae_u still blows up           → trivial attractor is real."
echo "============================================================"
