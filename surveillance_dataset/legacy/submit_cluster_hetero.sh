#!/bin/bash
#SBATCH --job-name=surv_hetero
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --account=minjilab99
#SBATCH --output=/nfs/turbo/umms-minjilab/lpullela/surveillance_dataset/logs/cluster_hetero_%j.out
#SBATCH --error=/nfs/turbo/umms-minjilab/lpullela/surveillance_dataset/logs/cluster_hetero_%j.err
#
# Mixed per-column noise: each frame draws σ ~ Unif[0, 1].
# Tunes SSC-TV-L21, SSC-TV-L21-col, OSC, TKSS (40 trials) and evaluates BDOSC with
# fixed params. Results: split_k5_hetero_trials40.csv
#
# Submit from anywhere:
#   sbatch /nfs/turbo/umms-minjilab/lpullela/surveillance_dataset/submit_cluster_hetero.sh

set -euo pipefail

ROOT=/nfs/turbo/umms-minjilab/lpullela/surveillance_dataset
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
python -u "$ROOT/cluster_experiment.py" --hetero-noise \
    --methods OSC SSC-TV-L21 SSC-TV-L21-col TKSS BDOSC Gram-NCut
echo "end=$(date)"
