#!/bin/bash
# Launch benchmark experiments for SSC-Block-TV variants
# For k = 5, 8, 12 on both surveillance and ballet datasets

set -euo pipefail

# Load lmod if available
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; fi
if [ -f /sw/lmod/lmod/init/bash ]; then . /sw/lmod/lmod/init/bash; fi

# Activate conda environment
source /home/lpullela/miniconda3/etc/profile.d/conda.sh
conda activate ssc_559

# Set threading environment variables
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Define methods to run - focusing on the block TV variants
METHODS="SSC-Block-TV SSC-Block-TV-Col21 SSC-Block-TV-L1"

echo "========================================"
echo "Starting Benchmark Experiments"
echo "Methods: $METHODS"
echo "K values: 5, 8, 12"
echo "known_k: True"
echo "Python: $(which python)"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "========================================"

# Surveillance dataset experiments
echo ""
echo "===== SURVEILLANCE DATASET ====="
cd /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset

for K in 5 8 12; do
    echo ""
    echo "--- Running surveillance k=$K ---"
    python -u cluster_experiment.py \
        --k $K \
        --known-k \
        --sigmas 0.0 \
        --methods $METHODS \
        --n-trials 10 \
        --out-tag "k${K}_variants"
done

# Ballet dataset experiments
echo ""
echo "===== BALLET DATASET ====="
cd /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset

for K in 5 8 12; do
    echo ""
    echo "--- Running ballet k=$K ---"
    python -u cluster_experiment.py \
        --k $K \
        --known-k \
        --methods $METHODS \
        --n-trials 100 \
        --out-tag "k${K}_variants"
done

echo ""
echo "========================================"
echo "All experiments completed!"
echo "End: $(date)"
echo "========================================"
