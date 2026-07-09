#!/bin/bash
#SBATCH -J pinn_re40_abl
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --array=0-3
#SBATCH -o pinn_re40_ablation_%A_%a.out
#SBATCH -e pinn_re40_ablation_%A_%a.err
# ============================================================================
# 2x2 ablation (PT x CFD anchor) as a JOB ARRAY: each of the 4 cells runs as an
# independent task, in parallel, each with its own 24h wall clock and 64G — so
# the four cells no longer share one wall-time budget (the sequential version
# was getting cut off at the time limit).
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

KEYS=(baseline pt cfd_anchor pt_cfd_anchor)
KEY=${KEYS[$SLURM_ARRAY_TASK_ID]}

case "$KEY" in
  baseline)      EXTRA="$ANCHOR_OFF" ;;
  pt)            EXTRA="$PT $ANCHOR_OFF" ;;
  cfd_anchor)    EXTRA="$ANCHOR_ON" ;;
  pt_cfd_anchor) EXTRA="$PT $ANCHOR_ON" ;;
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
