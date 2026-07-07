#!/bin/bash
#SBATCH -J pinn_re40_ablation
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -o pinn_re40_ablation_%j.out
#SBATCH -e pinn_re40_ablation_%j.err
# ============================================================================
# 2x2 ablation for Re=40 (single snapshot): pseudo-time (PT) x CFD data anchor.
#
#                    no anchor            anchor (--use-all-cfd-data)
#   no PT      baseline                   cfd_anchor
#   PT         pt                         pt_cfd_anchor
#
# "CFD anchor" = CFD velocity data supervision (the DATA loss term).
#   anchor OFF -> --no-data-anchor   (data-free PINN: PDE + BC only)
#   anchor ON  -> --use-all-cfd-data (all CFD velocity points supervised)
#
# ALL four cells share the SAME net (96/5), the SAME architecture (cfp40.py
# stream-function MLP), the SAME domain/sampling and the SAME optimizer budget,
# so the only things that vary are PT and the data anchor.
#
# Note: the CFD-aware PDE *residual* (--n-cfd-pde, PDE evaluated at CFD point
# locations, NO velocity labels) is kept ON for all four cells — it is physics,
# not a data anchor, so keeping it uniform isolates the two factors cleanly.
# ============================================================================
set -e

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

ROOT=../runs/ablation
mkdir -p "$ROOT"

run () {  # $1=key  $2...=extra flags
    local key="$1"; shift
    local sd="$ROOT/$key/checkpoints"
    local vd="$ROOT/$key/viz"
    mkdir -p "$sd" "$vd"
    echo "############################################################"
    echo "# 2x2 cell: $key   flags: $*"
    echo "############################################################"
    python -u cfp40.py $COMMON --save-dir "$sd" --viz-dir "$vd" "$@" \
        2>&1 | tee "$ROOT/$key/train.log"
}

PT="--use-pt --pt-w-init 1.0 --pt-ema 0.9"
ANCHOR_ON="--use-all-cfd-data"
ANCHOR_OFF="--no-data-anchor"

# (no PT, no anchor)
run baseline       $ANCHOR_OFF
# (PT, no anchor)
run pt             $PT $ANCHOR_OFF
# (no PT, anchor)
run cfd_anchor     $ANCHOR_ON
# (PT, anchor)
run pt_cfd_anchor  $PT $ANCHOR_ON

echo "============================================================"
echo " 2x2 done. Build the figure:"
echo "   cd src && python ablation_plot.py --vtk-path ../Re40.vtk \\"
echo "        --runs-root ../runs/ablation --out-dir ../runs/ablation/_summary \\"
echo "        --width $WIDTH --depth $DEPTH"
echo "============================================================"
