#!/bin/bash
#SBATCH -J pinn_rerender_vort
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 2
#SBATCH -t 00:15:00
#SBATCH -o logs/pinn_rerender_vort_%j.out
#SBATCH -e logs/pinn_rerender_vort_%j.err

# Re-render the three small-net (32/3) PINN vorticity panels at a matched
# colour scale (vmin/vmax = ±8) so slide 31 has apples-to-apples PINN ↔ CFD
# comparison.  Runs in <15 min on a single GPU (inference only, no training).

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn

cd "/oscar/home/jchen790/cylinder flow lab"

python -u reports/rerender_vorticity_matched.py
