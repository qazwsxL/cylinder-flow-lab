#!/bin/bash
#SBATCH -J pinn_re40_sweep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --array=0-5
#SBATCH -o pinn_re40_sweep_%A_%a.out
#SBATCH -e pinn_re40_sweep_%A_%a.err
# ============================================================================
# Re=40 "from data to no data" sweep: how few CFD data points can the PINN use
# and still recover the correct field, and does pseudo-time (PT) push that floor
# toward less data?
#
#   data amount N in {2000, 500, 100}  x  {no PT, PT}
#
# N = a number of RANDOM CFD points (resampled each Adam epoch / BFGS batch),
# an intermediate rung between all-data (runs/ablation/cfd_anchor) and no-data
# (runs/ablation/baseline for no-PT, runs/ablation/pt for PT). Combine this
# sweep with those two endpoints to get the full rel-L2-vs-N curve.
# ============================================================================
set -eo pipefail

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn
cd "/oscar/home/jchen790/cylinder flow lab/src"

VTK=../Re40.vtk
COMMON="--vtk-path $VTK --width 96 --depth 5 \
    --epochs-adam 2000 --maxiter-bfgs 6000 --iters-per-batch 150 \
    --method-bfgs SSBroyden1 \
    --n-f 30000 \
    --n-cfd-pde 8000 --lambda-pde-cfd 1.0 \
    --cfd-pde-wall-buffer 0.5 --cfd-pde-edge-buffer 0.5 \
    --seed 42"
# PT with widened bounds. NO --pt-resample: PT uses the same collocation strategy
# as the no-PT cells (Adam resamples each epoch; BFGS frozen), so the comparison
# isolates PT alone rather than PT+resampling.
PT="--use-pt --pt-w-init 1.0 --pt-ema 0.9 --pt-w-min 1e-4 --pt-w-max 1e4"

# index -> (N, PT?)
NS=(2000 2000 500 500 100 100)
PTS=(0    1    0   1   0   1)
N=${NS[$SLURM_ARRAY_TASK_ID]}
USEPT=${PTS[$SLURM_ARRAY_TASK_ID]}

if [ "$USEPT" = "1" ]; then TAG="data${N}_PT"; EXTRA="--n-data-fixed $N $PT";
else                        TAG="data${N}_noPT"; EXTRA="--n-data-fixed $N"; fi

ROOT=../runs/sweep/$TAG
mkdir -p "$ROOT/checkpoints" "$ROOT/viz"
echo "############################################################"
echo "# sweep cell '$TAG'   N=$N  PT=$USEPT"
echo "#   flags: $EXTRA"
echo "############################################################"
python -u cfp40.py $COMMON --save-dir "$ROOT/checkpoints" --viz-dir "$ROOT/viz" $EXTRA \
    2>&1 | tee "$ROOT/train.log"
echo "[done] $TAG -> $ROOT/checkpoints/pinn_Re40_single.pt"
