#!/bin/bash
#SBATCH --job-name=cv_tkss_k8_s0p25
#SBATCH --account=minjilab0
#SBATCH --partition=standard
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/block_ssc_benchmark/cv/logs/surveillance/cv_tkss_k8_s0p25_%j.out
#SBATCH --error=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/block_ssc_benchmark/cv/logs/surveillance/cv_tkss_k8_s0p25_%j.err

set -euo pipefail

if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; fi
if [ -f /sw/lmod/lmod/init/bash ]; then . /sw/lmod/lmod/init/bash; fi

source /home/lpullela/miniconda3/etc/profile.d/conda.sh
conda activate ssc_559

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

mkdir -p /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/block_ssc_benchmark/cv/logs/surveillance

echo "========================================"
echo "Job: cv_tkss_k8_s0p25"
echo "Method: TKSS  k: 8  sigma: 0.25  tune_pool_groups: 20"
echo "========================================"
echo "host=$(hostname)  python=$(which python)  start=$(date)"

cd /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset

python -u cluster_experiment.py \
    --k 8 \
    --known-k \
    --cv \
    --tune-pool-groups 20 \
    --methods TKSS \
    --sigmas 0.25 \
    --n-trials 20 \
    --out-dir /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/block_ssc_benchmark/cv \
    --out-tag cv_tkss_k8_sigma0p25

echo "end=$(date)"
echo "Experiment completed!"
