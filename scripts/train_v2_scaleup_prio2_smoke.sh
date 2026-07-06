#!/bin/bash
#SBATCH -J pinn_re40_scaleup_prio2_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -o logs/pinn_re40_scaleup_prio2_smoke_%j.out
#SBATCH -e logs/pinn_re40_scaleup_prio2_smoke_%j.err

# ================================================================
# SCALE-UP SMOKE — does bigger capacity close the PDE-vs-data gap?
# ================================================================
#
# Motivation
# ----------
# In the quick sweep (width=32, depth=3, ~3k params), all 8 P2 runs
# showed a strict monotonic trade-off:
#   priority ↑ → mae drift ↓ but PDE drop ↓
# The best balance was prio=2 with all CFD pts anchored:
#   mae_u  = 0.128  (1.21× P1)        ← "场贴合" ✓
#   PDE    = 0.545 → 0.467  (~1.2× drop)   ← "PDE 降" weak
#
# Hypothesis: the small net can't represent the CFD field AND have
# low PDE residual at the same time — the two manifolds don't overlap
# at this capacity. With more parameters BOTH may be achievable, which
# would meet mentor's strict (Y) criterion.
#
# Setup (smoke)
# -------------
#   width=64 depth=4  (~17k params, ~6× bigger than quick)
#   F=16 Fourier features (4 σ × 4 freqs per σ)
#   P1: pure data fit on ALL CFD pts, 500 Adam + 400 BFGS
#   P2: --use-all-cfd-data + n_f=20000 + data_priority=2
#       300 Adam + 500 BFGS  ← give BFGS extra room for PDE to drop
#
# Decision rule
# -------------
#   - If mae_u(P2)/mae_u(P1) ≤ 1.5  AND  PDE drops ≥ 5×
#     → capacity was the bottleneck; both halves of (Y) can coexist.
#       Promote to full-size (width=96 depth=5, long BFGS).
#   - If mae stays good but PDE still only ~1-2× drop
#     → capacity helped little; the trade-off is intrinsic to this PDE
#       residual landscape at Re=40. Try Re curriculum (option 2) or
#       accept prio=2 as the practical (Y) endpoint.
#   - If mae explodes  (>2× P1)
#     → bigger net + dense anchor still loses to the trivial attractor;
#       unlikely but would be a strong negative result.
# ================================================================

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn

cd "/oscar/home/jchen790/cylinder flow lab"

# ----- scaled-up network -----
WIDTH=64
DEPTH=4
ITERS_PER_BATCH=100
FOURIER_F=16
FOURIER_SIGMAS="0.25 0.5 1.0 2.0"
XMIN=-8.0
XMAX=12.0
YMIN=-8.0
YMAX=8.0

# ----- prio=2 = best-balanced point from quick sweep -----
P2_DATA_PRIORITY=2.0

ROOT=runs/v2_scaleup_prio2_smoke
mkdir -p $ROOT/P1 $ROOT/P2 logs

# =================================================================
# Phase 1 — pure data fit on ALL CFD pts (no PDE)
# =================================================================
echo "============================================================"
echo " [SCALEUP] Phase 1 — DATA-ONLY (all CFD pts, width=$WIDTH depth=$DEPTH)"
echo "============================================================"
T0=$(date +%s)

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir $ROOT/P1/checkpoints \
    --viz-dir  $ROOT/P1/viz \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 500 \
    --maxiter-bfgs 400 \
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
# Phase 2 — PDE-on + all CFD points kept, priority=2 (best balance)
# =================================================================
echo "============================================================"
echo " [SCALEUP] Phase 2 — PDE+all-CFD (priority=$P2_DATA_PRIORITY, n_f=20000)"
echo "  give BFGS extra room so PDE actually has a chance to drop"
echo "============================================================"
TA=$(date +%s)

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir $ROOT/P2/checkpoints \
    --viz-dir  $ROOT/P2/viz \
    --resume-from "$P1_CKPT" \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 300 \
    --maxiter-bfgs 500 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 20000 \
    --use-data \
    --use-all-cfd-data \
    --data-priority $P2_DATA_PRIORITY \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1

TB=$(date +%s); echo "[timing] Phase 2 took $((TB-TA)) s    (total $((TB-T0)) s)"

echo "============================================================"
echo " SCALE-UP SMOKE DONE."
echo "   $ROOT/P1/  baseline data fit (width=$WIDTH depth=$DEPTH)"
echo "   $ROOT/P2/  PDE + all-CFD, priority=$P2_DATA_PRIORITY"
echo ""
echo " Compare against quick-sweep (width=32) prio=2 row:"
echo "   quick   mae_u(P2)=0.128 (1.21× P1=0.106)  PDE 0.545→0.467 (~1.2× drop)"
echo "   scaleup mae_u(P2)=?     (?× P1=?)         PDE ?→? (?× drop)"
echo ""
echo " Decision rule:"
echo "   mae ratio ≤ 1.5 AND PDE drop ≥ 5×  →  promote to full size (Y achieved)"
echo "   mae stays good, PDE drop still ~1-2×  →  intrinsic trade-off, try Re curriculum"
echo "   mae > 2×  →  big net still loses to trivial attractor (unlikely)"
echo "============================================================"
