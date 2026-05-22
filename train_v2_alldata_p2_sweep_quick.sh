#!/bin/bash
#SBATCH -J pinn_re40_alldata_p2_quick
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -t 01:30:00
#SBATCH -o logs/pinn_re40_alldata_p2_quick_%j.out
#SBATCH -e logs/pinn_re40_alldata_p2_quick_%j.err

# ================================================================
# QUICK ALL-CFD-IN-P2 SWEEP  (~20 min)
# ================================================================
#
# This is the "don't turn CFD off in Phase 2" experiment.
# Phase 2 KEEPS all CFD points as a data anchor; we sweep the
# data_priority from very weak (0.5) up to strong (5) to map the
# full data-anchor-strength curve.
#
# Companion to train_v2_pde_weight_sweep_quick.sh — that one focused
# on {1,2,5}. This one adds priority=0.5 to map the weakest endpoint
# and gives a finer-grained picture of the trade-off.
#
# Protocol:
#   Phase 1 (shared baseline):
#     pure data fit on ALL CFD points (no PDE).
#
#   Phase 2 (sweep, 4 runs from the same P1 ckpt):
#     --use-all-cfd-data + n_f=20000, data_priority ∈ {0.5, 1, 2, 5}.
#
#   Comparison to existing weakanchor_quick:
#     weakanchor_quick P2 = priority=0.5 + n_data=500  → mae_u 0.174
#     here  prio=0.5 row  = priority=0.5 + ALL CFD     → tests whether
#       the "more data points at same weak weight" alone fixes drift.
#
# Decision rule:
#   - If priority≥1 with all CFD pts gives mae_u(P2)/mae_u(P1) ≤ 2
#     AND PDE drops a lot → strong anchor + PDE coexist; the v2 setup
#     IS practically consistent given a dense data anchor.
#   - If even priority=5 with all data still pulls mae_u badly → the
#     trivial attractor is so deep that BFGS escapes data even with
#     this anchor; need bigger model / Re curriculum / extra priors.
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

ROOT=runs/v2_alldata_p2_sweep_quick
mkdir -p $ROOT/P1 logs

# =================================================================
# Phase 1 — pure data fit on ALL CFD pts (shared baseline)
# =================================================================
echo "============================================================"
echo " [ALLDATA-P2] Phase 1 — DATA-ONLY (all CFD pts, no PDE)"
echo "============================================================"
T0=$(date +%s)

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir $ROOT/P1/checkpoints \
    --viz-dir  $ROOT/P1/viz \
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

P1_CKPT=$ROOT/P1/checkpoints/pinn_Re40_single.pt
if [[ ! -s "$P1_CKPT" ]]; then
    echo "[P2] Phase-1 checkpoint missing or empty ($P1_CKPT) — aborting."
    exit 1
fi

# =================================================================
# Phase 2 — sweep data_priority ∈ {0.5, 1, 2, 5} (all CFD pts kept on)
# =================================================================
# Each P2 trimmed a bit so 4 runs still fit ~20 min total.
# =================================================================
for PRIO in 0.5 1 2 5; do
    TAG=$(echo $PRIO | tr '.' 'p')           # 0.5 → 0p5
    OUT=$ROOT/P2_prio${TAG}
    mkdir -p $OUT/checkpoints $OUT/viz
    echo "============================================================"
    echo " [ALLDATA-P2] Phase 2 — priority=${PRIO}  (n_f=20000, all CFD pts)"
    echo "============================================================"
    TA=$(date +%s)

    python -u cfp40_v2.py --vtk-path Re40.vtk \
        --save-dir $OUT/checkpoints \
        --viz-dir  $OUT/viz \
        --resume-from "$P1_CKPT" \
        --width $WIDTH --depth $DEPTH \
        --epochs-adam 100 \
        --maxiter-bfgs 150 \
        --iters-per-batch $ITERS_PER_BATCH \
        --n-f 20000 \
        --use-data \
        --use-all-cfd-data \
        --data-priority $PRIO \
        --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
        --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
        --method-bfgs SSBroyden1

    TB=$(date +%s); echo "[timing] P2 prio=${PRIO} took $((TB-TA)) s"
done

T2=$(date +%s); echo "[timing] Total $((T2-T0)) s"

echo "============================================================"
echo " ALL-DATA-IN-P2 SWEEP DONE."
echo "   $ROOT/P1/          baseline data fit"
echo "   $ROOT/P2_prio0p5/  priority=0.5"
echo "   $ROOT/P2_prio1/    priority=1"
echo "   $ROOT/P2_prio2/    priority=2"
echo "   $ROOT/P2_prio5/    priority=5"
echo ""
echo " Tabulate: priority | PDE drop ratio | mae_u(P2)/mae_u(P1)"
echo " Lowest priority that keeps mae ratio ≤ 2 = answer to '场也贴合'."
echo "============================================================"
