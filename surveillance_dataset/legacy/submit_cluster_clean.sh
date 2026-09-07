#!/bin/bash
#SBATCH --job-name=surv_scale
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --account=minjilab99
#SBATCH --output=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset/logs/cluster_scale_%j.out
#SBATCH --error=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset/logs/cluster_scale_%j.err
#
# Noiseless clustering: tune and evaluate at σ=0 only.
# k inferred by eigengap then DP NCut; includes Gram-NCut baseline.
# Optuna budget scales with n_params: OSC=50, TKSS/SSC-col=75, SSC-TV-L21=75 (3 searched).
# BDOSC and Gram-NCut are fixed.
# Results: split_k5_clean_min30_trials50_khat_scaled.csv
#
# Submit from anywhere:
#   sbatch /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset/submit_cluster_clean.sh

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
    --sigmas 0.0 --n-trials 50 \
    --methods OSC SSC-TV-L21 SSC-TV-L21-col TKSS BDOSC Gram-NCut
echo "end=$(date)"
