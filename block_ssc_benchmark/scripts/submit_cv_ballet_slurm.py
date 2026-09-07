#!/usr/bin/env python3
"""Submit SLURM jobs for the ballet hold-out CV benchmark.

Mirrors submit_cv_surveillance_slurm.py: rotates every group of k sequences
through as the held-out test set (k-fold CV) instead of the single
train/test pool + sampled combos used by the non-CV ballet runs.
Hyperparameters are retuned per fold on that fold's train groups only,
using --tune-pool-groups (see below).

44 ballet sequences total. Fold counts: k=3 -> 14 folds (2 unused),
k=5 -> 8 folds (4 unused), k=8 -> 5 folds (4 unused). Unlike surveillance
(25 people), every held-in pool here (30-41 sequences) comfortably supports
20 distinct k-sized tuning subsets, so k=12 does not need to be dropped --
same K_VALUES = [3, 5, 8] are used for both datasets anyway.

--tune-pool-groups TUNE_POOL_GROUPS makes every fold tune on that many
randomly drawn, DISTINCT, possibly-overlapping k-sized subsets of that
fold's held-in sequences (see sample_combos in cluster_experiment.py),
instead of just the fold's few fixed partition groups. Reported train/test
eval rows still come from the original fixed partition groups.

One job per (k, method, sigma) combination:
  K_VALUES x METHODS x SIGMAS = 3 x 5 x 3 = 45 jobs.

Everything lives under one common folder, block_ssc_benchmark/cv/:
  cv/results/ballet/*_cv_{method}_k{k}_sigma{sigma}.csv
  cv/results/ballet/*_cv_{method}_k{k}_sigma{sigma}_timing.json
  cv/params/ballet/*_cv_..._sigma{s}_fold{i}.json
  cv/logs/ballet/cv_{method}_k{k}_sigma{sigma}_%j.{out,err}
"""

import subprocess
import sys
from pathlib import Path

BALLET_DIR = "/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset"
BENCHMARK_DIR = Path("/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/block_ssc_benchmark")
CV_DIR = BENCHMARK_DIR / "cv"

K_VALUES = [3, 5, 8]
METHODS = ["SSC-Block-TV-Col21", "OSC", "BDOSC", "TKSS", "Gram-NCut"]
SIGMAS = [0.0, 0.25, 0.5]
N_TRIALS = 20
TUNE_POOL_GROUPS = 20

SLURM_CONFIG = {
    "account": "minjilab0",
    "partition": "standard",
    "time": "1-00:00:00",
    "mem": "32G",
    "cpus": "4",
}


def sigma_tag(sigma):
    return f"{sigma:g}".replace(".", "p")


def method_tag(method):
    return method.lower().replace("-", "_")


def create_slurm_script(k, method, sigma):
    job_name = f"cvb_{method_tag(method)}_k{k}_s{sigma_tag(sigma)}"
    out_tag = f"cv_{method_tag(method)}_k{k}_sigma{sigma_tag(sigma)}"
    log_dir = CV_DIR / "logs" / "ballet"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={SLURM_CONFIG['account']}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem={SLURM_CONFIG['mem']}
#SBATCH --cpus-per-task={SLURM_CONFIG['cpus']}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err

set -euo pipefail

if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; fi
if [ -f /sw/lmod/lmod/init/bash ]; then . /sw/lmod/lmod/init/bash; fi

source /home/lpullela/miniconda3/etc/profile.d/conda.sh
conda activate ssc_559

export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-4}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-4}}
export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-4}}
export NUMEXPR_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-4}}
export PYTHONUNBUFFERED=1

mkdir -p {log_dir}

echo "========================================"
echo "Job: {job_name}"
echo "Method: {method}  k: {k}  sigma: {sigma}  tune_pool_groups: {TUNE_POOL_GROUPS}"
echo "========================================"
echo "host=$(hostname)  python=$(which python)  start=$(date)"

cd {BALLET_DIR}

python -u cluster_experiment.py \\
    --k {k} \\
    --known-k \\
    --cv \\
    --tune-pool-groups {TUNE_POOL_GROUPS} \\
    --methods {method} \\
    --sigmas {sigma} \\
    --n-trials {N_TRIALS} \\
    --out-dir {CV_DIR} \\
    --out-tag {out_tag}

echo "end=$(date)"
echo "Experiment completed!"
"""
    return script, job_name


def main():
    dry_run = "--dry-run" in sys.argv[1:]

    scripts_dir = BENCHMARK_DIR / "scripts" / "cv_slurm_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (CV_DIR / "logs" / "ballet").mkdir(parents=True, exist_ok=True)
    (CV_DIR / "results" / "ballet").mkdir(parents=True, exist_ok=True)
    (CV_DIR / "params" / "ballet").mkdir(parents=True, exist_ok=True)

    print("Ballet Hold-Out CV Benchmark - SLURM Job Submission")
    print("=" * 60)
    print(f"k values: {K_VALUES}")
    print(f"methods:  {METHODS}")
    print(f"sigmas:   {SIGMAS}")
    print(f"n_trials: {N_TRIALS}  tune_pool_groups: {TUNE_POOL_GROUPS}")
    print(f"total jobs: {len(K_VALUES) * len(METHODS) * len(SIGMAS)}")
    print(f"dry_run: {dry_run}")

    submitted_jobs = []
    for k in K_VALUES:
        for method in METHODS:
            for sigma in SIGMAS:
                script_content, job_name = create_slurm_script(k, method, sigma)
                script_path = scripts_dir / f"run_{job_name}.sh"
                script_path.write_text(script_content)
                script_path.chmod(0o755)
                print(f"\nCreated: {script_path}")

                if dry_run:
                    continue

                try:
                    result = subprocess.run(
                        ["sbatch", str(script_path)],
                        capture_output=True, text=True, check=True,
                    )
                    job_id = result.stdout.strip().split()[-1]
                    submitted_jobs.append((method, k, sigma, job_id))
                    print(f"  Submitted job: {job_id}")
                except subprocess.CalledProcessError as e:
                    print(f"  ERROR submitting job: {e}")
                    print(f"  stderr: {e.stderr}")

    print("\n" + "=" * 60)
    if dry_run:
        print(f"Dry run: wrote {len(K_VALUES) * len(METHODS) * len(SIGMAS)} scripts, submitted none.")
    else:
        print("Summary of Submitted Jobs:")
        for method, k, sigma, job_id in submitted_jobs:
            print(f"  {method:20s} k={k:2d}  sigma={sigma:<4g}  Job ID: {job_id}")
        print(f"\nTotal jobs submitted: {len(submitted_jobs)}")
    print(f"SLURM scripts saved in: {scripts_dir}")
    print("\nTo check job status: squeue -u $USER")
    print("To cancel all jobs: scancel -u $USER")


if __name__ == "__main__":
    main()
