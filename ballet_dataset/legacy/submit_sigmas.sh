#!/bin/bash
#SBATCH --job-name=ballet_sigma
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --account=minjilab99
#SBATCH --output=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset/logs/ballet_sigma_%j.out
#SBATCH --error=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset/logs/ballet_sigma_%j.err
#
# Homogeneous noise: one SLURM job per σ. Each job tunes its own hyperparams
# on that σ only (10 Optuna trials), then evaluates train+test at that σ.
# Hold out 15 sequences; 24 train + 18 test k=5 combos; 30x30 downsample.
#
# Results / params (per σ):
#   ballet_cluster_sigmas_{σ}_results.csv
#   ballet_cluster_sigmas_{σ}_params.json
#
# Submit all σ levels (from anywhere):
#   bash /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset/submit_sigmas.sh
#
# Or one level:
#   SIGMA=0.5 sbatch .../submit_sigmas.sh

set -euo pipefail

ROOT=/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset
SIGMAS=(0 0.25 0.5 0.75)

# No SIGMA → submit one job per σ (run via bash, not sbatch).
if [ -z "${SIGMA:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        echo "ERROR: SIGMA must be set inside a Slurm job (use: bash $0)" >&2
        exit 1
    fi
    mkdir -p "$ROOT/logs"
    echo "Submitting one job per σ: ${SIGMAS[*]}"
    for s in "${SIGMAS[@]}"; do
        jid=$(sbatch --parsable --job-name="ballet_s${s}" --export=ALL,SIGMA="$s" "$0")
        echo "  σ=${s}  job=${jid}"
    done
    exit 0
fi

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

echo "host=$(hostname)  python=$(which python)  start=$(date)  SIGMA=${SIGMA}"
python -u "$ROOT/cluster_experiment.py" --k 5 \
    --n-train-combos 24 --n-test-combos 18 --n-trials 10 \
    --sigmas "${SIGMA}" \
    --methods OSC TKSS SSC-TV-L21 SSC-TV-L21-col BDOSC Gram-NCut
echo "end=$(date)"
