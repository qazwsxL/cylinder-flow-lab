#!/bin/bash
#SBATCH -J pinn_re40_anneal
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 4
#SBATCH -t 48:00:00
#SBATCH -o pinn_re40_anneal_%j.out
#SBATCH -e pinn_re40_anneal_%j.err
# ============================================================================
# Re=40 DATA-ANNEALING curriculum: solve with data, then remove it step by step
# while WARM-STARTING each stage from the previous one, so the network stays in
# the correct solution basin instead of cold-starting into a spurious wake.
#
#   N = 2000 -> 1000 -> 500 -> 250 -> 100 -> 50 -> 0
#   each stage --resume-from the previous stage's checkpoint (except the first)
#   NO pseudo-time while data is present (the sweep showed PT hurts with data).
#   At N=0 we run TWO variants: warm-started plain, and warm-started faithful PT.
#
# Sequential + idempotent: a stage whose checkpoint already exists & finished is
# skipped, so if the 48h wall clock is hit you can just re-submit to continue.
# ============================================================================
set -eo pipefail

module load anaconda3/2023.09-0-aqbc
module load cuda/11.8.0-kuhf
source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate pinn
cd "/oscar/home/jchen790/cylinder flow lab/src"

COMMON="--vtk-path ../Re40.vtk --width 96 --depth 5 --iters-per-batch 150 \
    --method-bfgs SSBroyden1 --n-f 30000 \
    --n-cfd-pde 8000 --lambda-pde-cfd 1.0 \
    --cfd-pde-wall-buffer 0.5 --cfd-pde-edge-buffer 0.5 --seed 42"
ROOT=../runs/anneal
mkdir -p "$ROOT"

done_already () {  # $1 = stage dir
    [ -f "$1/checkpoints/pinn_Re40_single.pt" ] && \
    grep -q "\[train\] Saved" "$1/train.log" 2>/dev/null
}

run_stage () {  # $1=out  $2=epochs_adam  $3=maxiter_bfgs  $4...=extra flags
    local out="$1"; local ea="$2"; local mb="$3"; shift 3
    if done_already "$out"; then echo "[skip] $out already done"; return; fi
    mkdir -p "$out/checkpoints" "$out/viz"
    echo "############################################################"
    echo "# stage $out   epochs_adam=$ea maxiter_bfgs=$mb   flags: $*"
    echo "############################################################"
    python -u cfp40.py $COMMON --epochs-adam "$ea" --maxiter-bfgs "$mb" \
        --save-dir "$out/checkpoints" --viz-dir "$out/viz" "$@" \
        2>&1 | tee "$out/train.log"
}

# ---- data-annealing stages (velocity anchor, N shrinking, warm-started) ----
LEVELS=(2000 1000 500 250 100 50)
prev=""
for i in "${!LEVELS[@]}"; do
    N=${LEVELS[$i]}
    out="$ROOT/N${N}"
    if [ -z "$prev" ]; then
        run_stage "$out" 1500 3000 --n-data-fixed "$N"                       # cold start, solid budget
    else
        run_stage "$out" 200 1200 --n-data-fixed "$N" \
                  --resume-from "$prev/checkpoints/pinn_Re40_single.pt"       # warm-started, light budget
    fi
    prev="$out"
done

# ---- N=0 (data-free), warm-started from the smallest-data solution ----
run_stage "$ROOT/N0_noPT" 400 2500 --no-data-anchor \
          --resume-from "$prev/checkpoints/pinn_Re40_single.pt"
run_stage "$ROOT/N0_PT"   400 2500 --no-data-anchor \
          --resume-from "$prev/checkpoints/pinn_Re40_single.pt" \
          --use-pt --pt-resample --pt-w-init 1.0 --pt-ema 0.9 --pt-w-min 1e-4 --pt-w-max 1e4

echo "============================================================"
echo " Anneal done. Evaluate the whole curriculum:"
echo "   python export_fields.py --vtk-path ../Re40.vtk \\"
echo "        --runs-root ../runs/anneal --out ../runs/anneal/_export/fields.npz"
echo "============================================================"
