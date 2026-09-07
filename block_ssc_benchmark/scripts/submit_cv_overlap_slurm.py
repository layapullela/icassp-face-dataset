#!/usr/bin/env python3
"""Ablation: does an enlarged, overlapping tuning pool fix SSC-Block-TV-Col21's
noise-robustness gap at k=8/k=12?

Hypothesis (from the completed cv/ run): at k=8 and k=12, each fold's
tuning pool is only 2 and 1 fixed partition group(s) x 24 sequences (48
and 24 matrices) versus k=5's 4 groups (96 matrices). SSC-Block-TV-Col21
has more hyperparameters (4) than OSC (2) or TKSS (3), so it may be
tuning on too little data at k=8/12 -- consistent with it degrading
fastest under noise there (k=12, sigma=0.5: ARI=0.575, worst of all 5
methods; see cv/analysis/summary_table.md).

This resubmits ONLY the tuned methods (SSC-Block-TV-Col21, OSC, TKSS) at
k=8 and k=12, with --tune-pool-groups 4: each fold tunes on 4 randomly
drawn (possibly overlapping) k-sized subsets of that fold's held-in
people instead of the 1-2 fixed partition groups. Reported train/test
eval rows still come from the original fixed partition groups, so this
is directly comparable to the existing cv/ results.

k=5 is skipped (already has 4 groups/fold, nothing to fix). BDOSC and
Gram-NCut are skipped (not tuned, so there's no tuning pool to enlarge).

3 methods x 2 k x 3 sigma = 18 jobs. Output goes to a separate
"cv_overlap_" out-tag so it sits alongside (never overwrites) the
original cv/ results for direct before/after comparison.
"""

import subprocess
import sys
from pathlib import Path

SURVEILLANCE_DIR = "/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset"
BENCHMARK_DIR = Path("/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/block_ssc_benchmark")
CV_DIR = BENCHMARK_DIR / "cv"

K_VALUES = [8, 12]
METHODS = ["SSC-Block-TV-Col21", "OSC", "TKSS"]
SIGMAS = [0.0, 0.25, 0.5]
N_TRIALS = 20
TUNE_POOL_GROUPS = 4

SLURM_CONFIG = {
    "account": "minjilab0",
    "partition": "standard",
    "time": "9:00:00",
    "mem": "32G",
    "cpus": "4",
}


def sigma_tag(sigma):
    return f"{sigma:g}".replace(".", "p")


def method_tag(method):
    return method.lower().replace("-", "_")


def create_slurm_script(k, method, sigma):
    job_name = f"cvov_{method_tag(method)}_k{k}_s{sigma_tag(sigma)}"
    out_tag = f"cv_overlap_{method_tag(method)}_k{k}_sigma{sigma_tag(sigma)}"
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

    print("Surveillance CV Benchmark - Overlapping Tuning Pool Ablation")
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
