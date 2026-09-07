#!/bin/bash
#SBATCH --job-name=surv_osc_eval
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --account=minjilab99
#SBATCH --output=/nfs/turbo/umms-minjilab/lpullela/surveillance_dataset/logs/cluster_osc_eval_%j.out
#SBATCH --error=/nfs/turbo/umms-minjilab/lpullela/surveillance_dataset/logs/cluster_osc_eval_%j.err
#
# OSC eval only: job 59281670 hit the walltime after 39/40 Optuna trials.
# Uses the best trial (20: lambda_1=0.003017, lambda_2=8.42, mean ARI=0.7538)
# and appends train/test rows to split_k5_sigmas_trials40.csv.
#
# Submit:
#   sbatch /nfs/turbo/umms-minjilab/lpullela/surveillance_dataset/submit_cluster_osc_eval.sh

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
python -u "$ROOT/cluster_experiment.py" --methods OSC --append --no-tune
echo "end=$(date)"
