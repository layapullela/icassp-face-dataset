#!/bin/bash
#SBATCH --job-name=ballet_bcmp
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --account=minjilab99
#SBATCH --output=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset/logs/ballet_bcmp_%j.out
#SBATCH --error=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset/logs/ballet_bcmp_%j.err
#
# SSC-Block-TV vs OSC, TKSS, BDOSC, Gram-NCut (trivial DP). Known k.
# Block-TV: optional 3rd arg pins block_size; otherwise search {2,...,12}.
# Args: K MODE [BLOCK_SIZE]   MODE is clean | hetero
#   sbatch submit_block_compare.sh 8 clean
#   sbatch submit_block_compare.sh 12 clean
#   sbatch submit_block_compare.sh 8 clean 5
#   sbatch submit_block_compare.sh 5 hetero

set -euo pipefail

K="${1:-5}"
MODE="${2:-clean}"
BLOCK_SIZE="${3:-}"
N_TRIALS="${N_TRIALS:-50}"
METHODS=(SSC-Block-TV OSC TKSS BDOSC Gram-NCut)
BLOCK_ARGS=()
if [ -n "$BLOCK_SIZE" ]; then
    BLOCK_ARGS=(--block-size "$BLOCK_SIZE")
    TAG_PREFIX="block${BLOCK_SIZE}"
else
    TAG_PREFIX="bsearch"
fi

ROOT=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset
mkdir -p "$ROOT/logs"

if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; fi
if [ -f /sw/lmod/lmod/init/bash ]; then . /sw/lmod/lmod/init/bash; fi

source /home/lpullela/miniconda3/etc/profile.d/conda.sh
conda activate ssc_559

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1

cd "$ROOT"

echo "host=$(hostname)  python=$(which python)  start=$(date)  k=$K  mode=$MODE  block_size=${BLOCK_SIZE:-search}"
if [ "$MODE" = "hetero" ]; then
    python -u "$ROOT/cluster_experiment.py" --k "$K" --known-k --hetero-noise \
        --n-train-combos 24 --n-test-combos 18 --n-trials "$N_TRIALS" \
        --out-tag "${TAG_PREFIX}het" \
        --methods "${METHODS[@]}" \
        "${BLOCK_ARGS[@]}"
elif [ "$MODE" = "clean" ]; then
    python -u "$ROOT/cluster_experiment.py" --k "$K" --known-k \
        --n-train-combos 24 --n-test-combos 18 --n-trials "$N_TRIALS" \
        --out-tag "${TAG_PREFIX}k${K}" \
        --methods "${METHODS[@]}" \
        "${BLOCK_ARGS[@]}"
else
    echo "unknown MODE=$MODE (use clean or hetero)" >&2
    exit 1
fi
echo "end=$(date)"
