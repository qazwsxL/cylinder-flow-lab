#!/bin/bash
#SBATCH -J pinn_re40_pt
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -o pinn_re40_pt_%j.out
#SBATCH -e pinn_re40_pt_%j.err

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn

cd "/oscar/home/jchen790/cylinder flow lab/src"

python -u cfp40.py --vtk-path ../Re40.vtk \
    --save-dir ../runs/pt/checkpoints \
    --viz-dir  ../runs/pt/viz \
    --width 96 --depth 5 \
    --epochs-adam 2000 \
    --maxiter-bfgs 6000 --iters-per-batch 150 \
    --n-f 30000 --n-data 10000 \
    --n-cfd-pde 8000 --lambda-pde-cfd 1.0 \
    --cfd-pde-wall-buffer 0.5 --cfd-pde-edge-buffer 0.5 \
    --method-bfgs SSBroyden1 \
    --use-pt --pt-w-init 1.0 --pt-ema 0.9
