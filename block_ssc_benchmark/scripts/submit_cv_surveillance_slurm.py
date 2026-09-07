#!/usr/bin/env python3
"""Submit SLURM jobs for the surveillance hold-out CV benchmark.

Rotates every group of k people through as the held-out test set (k-fold
CV) instead of the single fixed split used by submit_block_variants_slurm.py.
Hyperparameters are retuned per fold on that fold's train groups only,
using --tune-pool-groups (see below).

k=12 was dropped: a 25-person pool only supports 2 disjoint groups of 12
(1 person unused), and only C(13,12)=13 *distinct* size-12 subsets exist
in the 13 held-in people once one is tested -- not enough combinatorial
room for a real tuning pool (see cv/deprecated/ for that run and the
overlapping-pool ablation that surfaced this). k=3 was added instead.
Fold counts from 25 people: k=3 -> 8 folds, k=5 -> 5 folds, k=8 -> 3 folds.

--tune-pool-groups TUNE_POOL_GROUPS makes every fold tune on that many
randomly drawn, DISTINCT, possibly-overlapping k-sized subsets of that
fold's held-in people (see sample_overlapping_groups in
cluster_experiment.py), instead of just the fold's 1-7 fixed partition
groups. This is now the standard tuning setup for every k here (not an
ablation) -- with TUNE_POOL_GROUPS=20, every (method, k, sigma, fold)
tunes on >=20 distinct group compositions x 24 sequences. Reported
train/test eval rows still come from the original fixed partition
groups. BDOSC and Gram-NCut aren't tuned, so the flag is a no-op for them.

One job per (k, method, sigma) combination:
  K_VALUES x METHODS x SIGMAS = 3 x 5 x 3 = 45 jobs.

Everything lives under one common folder, block_ssc_benchmark/cv/:
  cv/results/surveillance/*_cv_{method}_k{k}_sigma{sigma}.csv
  cv/results/surveillance/*_cv_{method}_k{k}_sigma{sigma}_timing.json
  cv/params/surveillance/*_cv_{method}_k{k}_sigma{sigma}_sigma{s}_fold{i}.json
  cv/logs/surveillance/cv_{method}_k{k}_sigma{sigma}_%j.{out,err}
"""

import subprocess
import sys
from pathlib import Path

SURVEILLANCE_DIR = "/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset"
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
    job_name = f"cv_{method_tag(method)}_k{k}_s{sigma_tag(sigma)}"
    out_tag = f"cv_{method_tag(method)}_k{k}_sigma{sigma_tag(sigma)}"
    log_dir = CV_DIR / "logs" / "surveillance"

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

cd {SURVEILLANCE_DIR}

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
    (CV_DIR / "logs" / "surveillance").mkdir(parents=True, exist_ok=True)
    (CV_DIR / "results" / "surveillance").mkdir(parents=True, exist_ok=True)
    (CV_DIR / "params" / "surveillance").mkdir(parents=True, exist_ok=True)

    print("Surveillance Hold-Out CV Benchmark - SLURM Job Submission")
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
