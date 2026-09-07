#!/usr/bin/env python3
"""
Submit SLURM jobs for SSC-Block-TV variant experiments.
Creates separate jobs for each (dataset, k) combination.

Defaults to --known-k (true cluster count, no eigengap), matching
submit_block_compare.sh. Pass --khat to estimate k instead.
"""

import subprocess
import sys
from pathlib import Path

# Configuration
DATASETS = {
    "surveillance": {
        "path": "/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset",
        "n_trials": 10,
        # Match submit_block_compare.sh clean mode (no 0.25/0.5/0.75 sweep).
        "extra_args": ["--sigmas", "0.0"],
    },
    "ballet": {
        "path": "/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset",
        "n_trials": 100,
        "extra_args": [],
    },
}

K_VALUES = [5, 8, 12]
METHODS = ["SSC-Block-TV", "SSC-Block-TV-Col21", "SSC-Block-TV-L1"]
BASELINE_METHODS = ["OSC", "TKSS", "BDOSC"]
KNOWN_K = True

# SLURM configuration
SLURM_CONFIG = {
    "account": "minjilab99",
    "partition": "standard",
    "time": "1-00:00:00",  # 1 day
    "mem": "32G",
    "cpus": "4",
}


def create_slurm_script(
    dataset_name, k, dataset_info, known_k=KNOWN_K, methods=None, out_tag=None, job_prefix="ssc_block",
):
    """Create SLURM submission script for a specific experiment."""
    methods = list(methods or METHODS)
    job_name = f"{job_prefix}_{dataset_name}_k{k}"
    out_tag = out_tag or f"k{k}_variants"
    known_k_echo = "True" if known_k else "False"
    extra_args = list(dataset_info.get("extra_args") or [])
    cmd_flags = [f"--k {k}"]
    if known_k:
        cmd_flags.append("--known-k")
    if extra_args:
        cmd_flags.append(" ".join(extra_args))
    cmd_flags.append(f"--methods {' '.join(methods)}")
    cmd_flags.append(f"--n-trials {dataset_info['n_trials']}")
    cmd_flags.append(f"--out-tag {out_tag}")
    cmd_joined = " \\\n    ".join(cmd_flags)
    extra_echo = " ".join(extra_args) if extra_args else "(none)"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={SLURM_CONFIG['account']}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem={SLURM_CONFIG['mem']}
#SBATCH --cpus-per-task={SLURM_CONFIG['cpus']}
#SBATCH --output={dataset_info['path']}/logs/{job_name}_%j.out
#SBATCH --error={dataset_info['path']}/logs/{job_name}_%j.err

set -euo pipefail

# Load lmod if available
if [ -f /etc/profile.d/lmod.sh ]; then . /etc/profile.d/lmod.sh; fi
if [ -f /sw/lmod/lmod/init/bash ]; then . /sw/lmod/lmod/init/bash; fi

# Activate conda environment
source /home/lpullela/miniconda3/etc/profile.d/conda.sh
conda activate ssc_559

# Set threading environment variables
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-4}}
export MKL_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-4}}
export OPENBLAS_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-4}}
export NUMEXPR_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-4}}
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job: {job_name}"
echo "Dataset: {dataset_name}"
echo "K: {k}"
echo "known_k: {known_k_echo}"
echo "extra_args: {extra_echo}"
echo "Methods: {' '.join(methods)}"
echo "========================================"
echo "host=$(hostname)  python=$(which python)  start=$(date)"

cd {dataset_info['path']}

# Create logs directory if it doesn't exist
mkdir -p logs

python -u cluster_experiment.py \\
    {cmd_joined}

echo "end=$(date)"
echo "Experiment completed!"
"""
    return script


def main():
    known_k = KNOWN_K
    argv = sys.argv[1:]
    if "--khat" in argv:
        known_k = False
        argv = [a for a in argv if a != "--khat"]

    baselines = False
    if "--baselines" in argv:
        baselines = True
        argv = [a for a in argv if a != "--baselines"]
    methods = BASELINE_METHODS if baselines else METHODS
    out_tag_suffix = "baselines" if baselines else "variants"
    job_prefix = "ssc_base" if baselines else "ssc_block"

    dataset_filter = None
    if "--datasets" in argv:
        i = argv.index("--datasets")
        dataset_filter = argv[i + 1].split(",")
        argv = argv[:i] + argv[i + 2:]
    datasets = DATASETS
    if dataset_filter:
        unknown = [d for d in dataset_filter if d not in DATASETS]
        if unknown:
            raise SystemExit(f"unknown dataset(s): {unknown}; choose from {list(DATASETS)}")
        datasets = {k: DATASETS[k] for k in dataset_filter}

    print("SSC-Block-TV Variants Benchmark - SLURM Job Submission")
    print("=" * 60)
    print(f"known_k: {known_k}  (pass --khat to estimate k with eigengap)")
    print(f"datasets: {', '.join(datasets)}")
    print(f"methods: {', '.join(methods)}")
    print(f"out_tag: k{{k}}_{out_tag_suffix}")

    # Create slurm_scripts directory
    scripts_dir = Path("/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/slurm_scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    submitted_jobs = []
    
    for dataset_name, dataset_info in datasets.items():
        for k in K_VALUES:
            # Create SLURM script
            script_content = create_slurm_script(
                dataset_name, k, dataset_info, known_k=known_k,
                methods=methods, out_tag=f"k{k}_{out_tag_suffix}",
                job_prefix=job_prefix,
            )
            script_path = scripts_dir / f"run_{job_prefix}_{dataset_name}_k{k}.sh"
            
            # Write script
            script_path.write_text(script_content)
            script_path.chmod(0o755)
            
            print(f"\nCreated: {script_path}")
            print(f"  Dataset: {dataset_name}, K: {k}, known_k={known_k}")
            
            # Submit job
            try:
                result = subprocess.run(
                    ["sbatch", str(script_path)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                job_id = result.stdout.strip().split()[-1]
                submitted_jobs.append((dataset_name, k, job_id))
                print(f"  Submitted job: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"  ERROR submitting job: {e}")
                print(f"  stderr: {e.stderr}")
    
    print("\n" + "=" * 60)
    print("Summary of Submitted Jobs:")
    print("=" * 60)
    for dataset_name, k, job_id in submitted_jobs:
        print(f"  {dataset_name:15s} k={k:2d}  Job ID: {job_id}")
    
    print(f"\nTotal jobs submitted: {len(submitted_jobs)}")
    print(f"SLURM scripts saved in: {scripts_dir}")
    print("\nTo check job status: squeue -u $USER")
    print("To cancel all jobs: scancel -u $USER")


if __name__ == "__main__":
    main()
