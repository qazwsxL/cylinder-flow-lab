#!/bin/bash
#SBATCH -J pinn_re40_pdew_sweep_quick
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -t 01:30:00
#SBATCH -o logs/pinn_re40_pdew_sweep_quick_%j.out
#SBATCH -e logs/pinn_re40_pdew_sweep_quick_%j.err

# ================================================================
# QUICK PDE-WEIGHT SWEEP  (~20 min)
# ================================================================
#
# Goal: pinpoint the data_priority value at which P2 BOTH:
#   (a) PDE residual drops substantially vs P1, AND
#   (b) mae_u stays within ~2× of P1 (field still fits CFD).
#
# Protocol:
#   Phase 1 (shared baseline):
#     pure data fit on ALL CFD points (no PDE), same as consistency_quick P1.
#
#   Phase 2 (sweep, 3 runs from the same P1 ckpt):
#     keep ALL CFD points as data anchor (--use-all-cfd-data),
#     turn PDE on at n_f=20000, sweep data_priority in {1, 2, 5}.
#     - priority=1  → data and PDE both at auto-norm baseline
#     - priority=2  → data 2× stronger than PDE (mild anchor)
#     - priority=5  → data 5× stronger than PDE (strong anchor, default)
#
# Decision rule:
#   Look at PDE drop ratio (P1→P2) AND mae_u (P2/P1):
#     priority where PDE drops ≥ 10× AND mae_u(P2)/mae_u(P1) ≤ 2
#     → that's the sweet spot, promote to full-size run.
#   If ALL three priorities pull mae_u badly, anchor strength alone
#   won't fix it → need bigger model or Re curriculum.
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

ROOT=runs/v2_pde_weight_sweep_quick
mkdir -p $ROOT/P1 logs

# =================================================================
# Phase 1 — pure data fit on ALL CFD pts (shared baseline)
# =================================================================
echo "============================================================"
echo " [PDEW-SWEEP] Phase 1 — DATA-ONLY (all CFD pts, no PDE)"
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
# Phase 2 — sweep data_priority ∈ {1, 2, 5} (all CFD pts kept on)
# =================================================================
for PRIO in 1 2 5; do
    OUT=$ROOT/P2_prio${PRIO}
    mkdir -p $OUT/checkpoints $OUT/viz
    echo "============================================================"
    echo " [PDEW-SWEEP] Phase 2 — priority=${PRIO}  (n_f=20000, all CFD pts)"
    echo "============================================================"
    TA=$(date +%s)

    python -u cfp40_v2.py --vtk-path Re40.vtk \
        --save-dir $OUT/checkpoints \
        --viz-dir  $OUT/viz \
        --resume-from "$P1_CKPT" \
        --width $WIDTH --depth $DEPTH \
        --epochs-adam 150 \
        --maxiter-bfgs 200 \
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
echo " PDE WEIGHT SWEEP DONE."
echo "   $ROOT/P1/         baseline data fit"
echo "   $ROOT/P2_prio1/   priority=1"
echo "   $ROOT/P2_prio2/   priority=2"
echo "   $ROOT/P2_prio5/   priority=5"
echo ""
echo " Compare:  PDE drop ratio (P1→P2) AND mae_u(P2)/mae_u(P1)"
echo " Sweet spot = PDE drops ≥10× AND mae_u stays within 2× of P1."
echo "============================================================"
