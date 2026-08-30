#!/bin/bash
#SBATCH --job-name=surv_cluster
#SBATCH --time=15:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --account=minjilab99
#SBATCH --output=/nfs/turbo/umms-minjilab/lpullela/surveillance_dataset/logs/cluster_experiment_%j.out
#SBATCH --error=/nfs/turbo/umms-minjilab/lpullela/surveillance_dataset/logs/cluster_experiment_%j.err
#
# Submit from anywhere:
#   sbatch /nfs/turbo/umms-minjilab/lpullela/surveillance_dataset/submit_cluster_experiment.sh
#
# Check progress:
#   squeue -u $USER
#   tail -f /nfs/turbo/umms-minjilab/lpullela/surveillance_dataset/logs/cluster_experiment_<jobid>.out

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
python -u "$ROOT/cluster_experiment.py"
echo "end=$(date)"
