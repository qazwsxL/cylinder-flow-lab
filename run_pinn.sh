#!/bin/bash
#SBATCH -J pinn_re40
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -o pinn_re40_%j.out
#SBATCH -e pinn_re40_%j.err

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn

cd "/oscar/home/jchen790/cylinder flow lab"

python -u cfp40.py --vtk-path Re40.vtk \
    --save-dir checkpoints_re40_v2 --viz-dir viz_re40_v2 \
    --width 128 --depth 6 \
    --epochs-adam 4000 --maxiter-bfgs 12000 --iters-per-batch 200 \
    --n-f 50000 --n-data 20000 \
    --n-cfd-pde 12000 --lambda-pde-cfd 2.0 \
    --cfd-pde-wall-buffer 0.5 --cfd-pde-edge-buffer 0.5 \
    --method-bfgs SSBroyden1