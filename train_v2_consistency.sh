#!/bin/bash
set -e

#SBATCH -J pinn_re40_consist
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH -t 03:00:00
#SBATCH -o logs/pinn_re40_consist_%j.out
#SBATCH -e logs/pinn_re40_consist_%j.err

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn

cd "/oscar/home/jchen790/cylinder flow lab"

WIDTH=48
DEPTH=4
ITERS_PER_BATCH=50
FOURIER_F=32
FOURIER_SIGMAS="0.25 0.5 1.0 2.0"
XMIN=-8.0
XMAX=12.0
YMIN=-8.0
YMAX=8.0

mkdir -p runs/v2_consistency/P1 runs/v2_consistency/P2 logs

echo "============================================================"
echo " Phase 1 — DATA-ONLY short test"
echo "============================================================"

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir runs/v2_consistency/P1/checkpoints \
    --viz-dir  runs/v2_consistency/P1/viz \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 150 \
    --maxiter-bfgs 150 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 0 \
    --data-only \
    --use-all-cfd-data \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1

echo "============================================================"
echo " Phase 2 — PDE-ONLY short consistency test"
echo "============================================================"

P1_CKPT=runs/v2_consistency/P1/checkpoints/pinn_Re40_single.pt
if [[ ! -s "$P1_CKPT" ]]; then
    echo "[ERROR] Phase-1 checkpoint missing: $P1_CKPT"
    exit 1
fi

python -u cfp40_v2.py --vtk-path Re40.vtk \
    --save-dir runs/v2_consistency/P2/checkpoints \
    --viz-dir  runs/v2_consistency/P2/viz \
    --resume-from "$P1_CKPT" \
    --width $WIDTH --depth $DEPTH \
    --epochs-adam 100 \
    --maxiter-bfgs 200 \
    --iters-per-batch $ITERS_PER_BATCH \
    --n-f 5000 \
    --fourier-features $FOURIER_F --fourier-sigmas "$FOURIER_SIGMAS" \
    --x-min $XMIN --x-max $XMAX --y-min $YMIN --y-max $YMAX \
    --method-bfgs SSBroyden1

echo "============================================================"
echo " Short consistency test done."
echo " Compare P1 vs P2 ABSOLUTE error blocks."
echo " Success = PDE drops and CFD error does not drift much."
echo "============================================================"
