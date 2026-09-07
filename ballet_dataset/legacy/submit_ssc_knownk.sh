#!/bin/bash
#SBATCH --job-name=ballet_ssck
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --account=minjilab99
#SBATCH --output=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset/logs/ballet_ssck_%j.out
#SBATCH --error=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset/logs/ballet_ssck_%j.err
#
# SSC-TV-L21 + SSC-TV-L21-col only, constrained search, known k.
# Results: ballet_cluster_knownk_scaled_results_ssc2.csv
#
# Submit from anywhere:
#   sbatch /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset/submit_ssc_knownk.sh

set -euo pipefail

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

echo "host=$(hostname)  python=$(which python)  start=$(date)"
python -u "$ROOT/cluster_experiment.py" --k 5 --known-k \
    --n-train-combos 24 --n-test-combos 18 --n-trials 50 \
    --out-tag fix \
    --methods SSC-TV-L21 SSC-TV-L21-col
echo "end=$(date)"
