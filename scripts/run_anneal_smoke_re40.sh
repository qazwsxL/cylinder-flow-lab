#!/bin/bash
#SBATCH -J pinn_re40_anneal_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH -o pinn_re40_anneal_smoke_%j.out
#SBATCH -e pinn_re40_anneal_smoke_%j.err
# ============================================================================
# SMOKE test of the data-annealing idea (cheap, ~a few hours). One question:
# does warm-starting N=0 from a correct low-data solution KEEP the field correct
# (<< 0.56, the cold-start no-data error), or does it still collapse to the
# spurious wake?
#
#   A: N=500  (cold, moderate budget)  -> a correct low-data reference
#   B: N=0    warm-started from A, no PT
#   C: N=0    warm-started from A, faithful PT (resample + widened bounds)
#
# If B (and/or C) land near A's error -> annealing works, run the full ladder.
# If they bounce back to ~0.5 -> warm-start alone isn't enough; rethink.
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
ROOT=../runs/anneal_smoke
mkdir -p "$ROOT"

done_already () { [ -f "$1/checkpoints/pinn_Re40_single.pt" ] && grep -q "\[train\] Saved" "$1/train.log" 2>/dev/null; }
run_stage () {
    local out="$1"; local ea="$2"; local mb="$3"; shift 3
    if done_already "$out"; then echo "[skip] $out done"; return; fi
    mkdir -p "$out/checkpoints" "$out/viz"
    echo "###### stage $out  adam=$ea bfgs=$mb  flags: $* ######"
    python -u cfp40.py $COMMON --epochs-adam "$ea" --maxiter-bfgs "$mb" \
        --save-dir "$out/checkpoints" --viz-dir "$out/viz" "$@" 2>&1 | tee "$out/train.log"
}

# A: correct low-data reference (cold)
run_stage "$ROOT/A_N500" 800 1200 --n-data-fixed 500
A="$ROOT/A_N500/checkpoints/pinn_Re40_single.pt"

# B: N=0 warm-started, no PT  (the key test)
run_stage "$ROOT/B_N0_noPT" 300 1200 --no-data-anchor --resume-from "$A"

# C: N=0 warm-started, faithful PT
run_stage "$ROOT/C_N0_PT"   300 1200 --no-data-anchor --resume-from "$A" \
          --use-pt --pt-resample --pt-w-init 1.0 --pt-ema 0.9 --pt-w-min 1e-4 --pt-w-max 1e4

echo "=== smoke done. Evaluate: ==="
echo "  python export_fields.py --vtk-path ../Re40.vtk --runs-root ../runs/anneal_smoke --out ../runs/anneal_smoke/_export/fields.npz"
