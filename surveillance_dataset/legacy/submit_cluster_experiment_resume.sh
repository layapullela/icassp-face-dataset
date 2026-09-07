#!/bin/bash
#SBATCH --job-name=surv_cluster_resume
#SBATCH --time=1-12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard
#SBATCH --account=minjilab99
#SBATCH --output=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset/logs/cluster_experiment_resume_%j.out
#SBATCH --error=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset/logs/cluster_experiment_resume_%j.err
#
# Resume after OSC tuning crash: load saved OSC params, tune SSC-TV-L21 and TKSS,
# then run full train/test eval for all methods.
#
# Submit from anywhere:
#   sbatch /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset/submit_cluster_experiment_resume.sh

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
    --skip-tune OSC \
    --methods OSC SSC-TV-L21 TKSS BDOSC
echo "end=$(date)"
