#!/bin/bash
#SBATCH --job-name=surv_block
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --account=minjilab99
#SBATCH --output=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset/logs/cluster_block_%j.out
#SBATCH --error=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset/logs/cluster_block_%j.err
#
# SSC-Block-TV only, clean data (σ=0), known k=5. Block size k=5 (fixed).
# Results: split_k5_clean_min30_trials50_knownk_scaled_blocktv.csv
#
# Submit from anywhere:
#   sbatch /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset/submit_block_tv.sh

set -euo pipefail

ROOT=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset
mkdir -p "$ROOT/logs"

if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; fi
if [ -f /sw/lmod/lmod/init/bash ]; then . /sw/lmod/lmod/init/bash; fi

source /home/lpullela/miniconda3/etc/profile.d/conda.sh
conda activate ssc_559

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

cd "$ROOT"

echo "host=$(hostname)  python=$(which python)  start=$(date)"
python -u "$ROOT/cluster_experiment.py" \
    --sigmas 0.0 --n-trials 50 --known-k --out-tag blocktv \
    --methods SSC-Block-TV
echo "end=$(date)"
