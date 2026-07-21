#!/bin/bash
#SBATCH -J pinn_re40_abl
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --array=0-3
#SBATCH -o pinn_re40_ablation_%A_%a.out
#SBATCH -e pinn_re40_ablation_%A_%a.err
# ============================================================================
# 2x2 ablation (PT x CFD anchor) as a JOB ARRAY: each of the 4 cells runs as an
# independent task, in parallel, each with its own 24h wall clock and 100G — so
# the four cells no longer share one wall-time budget.
#
# MEMORY: 96/5 SSBroyden keeps a dense NxN float64 inverse Hessian (H0 ~ 11.6 GB)
# and the update peaks at ~6-8x H0. 64G OOM-killed every cell at BFGS batch 2;
# 100G clears the pre-flight soft minimum (~83 GB). If a cell still OOMs, raise
# to 128G (sbatch --mem=128G ...).
#
#   task 0: baseline       (no PT, no anchor)
#   task 1: pt             (PT,    no anchor)
#   task 2: cfd_anchor     (no PT, anchor)
#   task 3: pt_cfd_anchor  (PT,    anchor)
#
# PT cells use the faithful stepper (component-wise tau + gamma) + --pt-resample.
# ============================================================================
set -eo pipefail

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn

cd "/oscar/home/jchen790/cylinder flow lab/src"

# ---- shared config (identical across all four cells) -----------------------
VTK=../Re40.vtk
WIDTH=96
DEPTH=5
EPOCHS_ADAM=2000
MAXITER_BFGS=6000
ITERS_PER_BATCH=150
COMMON="--vtk-path $VTK --width $WIDTH --depth $DEPTH \
    --epochs-adam $EPOCHS_ADAM --maxiter-bfgs $MAXITER_BFGS \
    --iters-per-batch $ITERS_PER_BATCH \
    --method-bfgs SSBroyden1 \
    --n-f 30000 --n-data 10000 \
    --n-cfd-pde 8000 --lambda-pde-cfd 1.0 \
    --cfd-pde-wall-buffer 0.5 --cfd-pde-edge-buffer 0.5 \
    --seed 42"

PT="--use-pt --pt-w-init 1.0 --pt-ema 0.9 --pt-resample"
ANCHOR_ON="--use-all-cfd-data"
ANCHOR_OFF="--no-data-anchor"

# indices 0-3: original 2x2.  4: velocity+pressure anchor.  5: pure PT, widened w.
KEYS=(baseline pt cfd_anchor pt_cfd_anchor cfd_anchor_p pt_wide)
KEY=${KEYS[$SLURM_ARRAY_TASK_ID]}

PT_WIDE="--use-pt --pt-w-init 1.0 --pt-ema 0.9 --pt-resample --pt-w-min 1e-4 --pt-w-max 1e4"

case "$KEY" in
  baseline)      EXTRA="$ANCHOR_OFF" ;;
  pt)            EXTRA="$PT $ANCHOR_OFF" ;;
  cfd_anchor)    EXTRA="$ANCHOR_ON" ;;
  pt_cfd_anchor) EXTRA="$PT $ANCHOR_ON" ;;
  cfd_anchor_p)  EXTRA="$ANCHOR_ON --anchor-pressure --data-p-weight 1.0" ;;
  pt_wide)       EXTRA="$PT_WIDE $ANCHOR_OFF" ;;
  *) echo "unknown SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

ROOT=../runs/ablation/$KEY
mkdir -p "$ROOT/checkpoints" "$ROOT/viz"

echo "############################################################"
echo "# array task $SLURM_ARRAY_TASK_ID  ->  cell '$KEY'"
echo "#   flags: $EXTRA"
echo "############################################################"

python -u cfp40.py $COMMON \
    --save-dir "$ROOT/checkpoints" --viz-dir "$ROOT/viz" $EXTRA \
    2>&1 | tee "$ROOT/train.log"

echo "[done] cell '$KEY' finished -> $ROOT/checkpoints/pinn_Re40_single.pt"
