#!/bin/bash
#SBATCH --job-name=ssc_block_surveillance_k5
#SBATCH --account=minjilab99
#SBATCH --partition=standard
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset/logs/ssc_block_surveillance_k5_%j.out
#SBATCH --error=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset/logs/ssc_block_surveillance_k5_%j.err

set -euo pipefail

# Load lmod if available
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; fi
if [ -f /sw/lmod/lmod/init/bash ]; then . /sw/lmod/lmod/init/bash; fi

# Activate conda environment
source /home/lpullela/miniconda3/etc/profile.d/conda.sh
conda activate ssc_559

# Set threading environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job: ssc_block_surveillance_k5"
echo "Dataset: surveillance"
echo "K: 5"
echo "known_k: True"
echo "extra_args: --sigmas 0.0"
echo "Methods: SSC-Block-TV SSC-Block-TV-Col21 SSC-Block-TV-L1"
echo "========================================"
echo "host=$(hostname)  python=$(which python)  start=$(date)"

cd /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset

# Create logs directory if it doesn't exist
mkdir -p logs

python -u cluster_experiment.py \
    --k 5 \
    --known-k \
    --sigmas \
    0.0 \
    --methods SSC-Block-TV SSC-Block-TV-Col21 SSC-Block-TV-L1 \
    --n-trials 10 \
    --out-tag k5_variants

echo "end=$(date)"
echo "Experiment completed!"
