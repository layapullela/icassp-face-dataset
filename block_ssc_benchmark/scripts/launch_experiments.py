#!/usr/bin/env python3
"""
Interactive launcher for SSC-Block-TV variant experiments.
Allows you to select which experiments to run.
"""

import subprocess
import sys
from pathlib import Path

def run_experiment(dataset, k, methods, n_trials):
    """Run a single experiment."""
    if dataset == "surveillance":
        exp_dir = "/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/surveillance_dataset"
    else:
        exp_dir = "/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/ballet_dataset"
    
    out_tag = f"k{k}_variants"
    
    cmd = [
        "python", "cluster_experiment.py",
        "--k", str(k),
        "--known-k",
    ]
    if dataset == "surveillance":
        cmd.extend(["--sigmas", "0.0"])
    cmd.extend(["--methods"] + methods + [
        "--n-trials", str(n_trials),
        "--out-tag", out_tag
    ])
    
    print(f"\n{'='*60}")
    print(f"Running: {dataset} dataset, k={k}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {exp_dir}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run(cmd, cwd=exp_dir, check=True)
        print(f"\n✓ Completed: {dataset} k={k}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {dataset} k={k}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ Interrupted by user")
        return False


def main():
    print("="*60)
    print("SSC-Block-TV Variants - Interactive Launcher")
    print("="*60)
    
    all_methods = ["SSC-Block-TV", "SSC-Block-TV-Col21", "SSC-Block-TV-L1"]
    datasets = ["surveillance", "ballet"]
    k_values = [5, 8, 12]
    
    print("\nAvailable methods:")
    for i, method in enumerate(all_methods, 1):
        print(f"  {i}. {method}")
    
    print("\nAvailable datasets:")
    for i, ds in enumerate(datasets, 1):
        print(f"  {i}. {ds}")
    
    print("\nAvailable k values:")
    print(f"  {', '.join(map(str, k_values))}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("Choose what to run:")
    print("  1. Run all experiments (all datasets, all k values)")
    print("  2. Run single dataset, all k values")
    print("  3. Run single dataset, single k value")
    print("  4. Custom selection")
    print("  q. Quit")
    
    choice = input("\nYour choice: ").strip()
    
    experiments = []
    
    if choice == "1":
        # Run all
        for dataset in datasets:
            n_trials = 10 if dataset == "surveillance" else 100
            for k in k_values:
                experiments.append((dataset, k, all_methods, n_trials))
    
    elif choice == "2":
        # Single dataset, all k
        print("\nSelect dataset:")
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds}")
        ds_choice = input("Dataset: ").strip()
        try:
            dataset = datasets[int(ds_choice) - 1]
            n_trials = 10 if dataset == "surveillance" else 100
            for k in k_values:
                experiments.append((dataset, k, all_methods, n_trials))
        except (ValueError, IndexError):
            print("Invalid choice!")
            return
    
    elif choice == "3":
        # Single dataset, single k
        print("\nSelect dataset:")
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds}")
        ds_choice = input("Dataset: ").strip()
        
        k_choice = input(f"K value ({', '.join(map(str, k_values))}): ").strip()
        
        try:
            dataset = datasets[int(ds_choice) - 1]
            k = int(k_choice)
            if k not in k_values:
                print(f"Invalid k value! Choose from {k_values}")
                return
            n_trials = 10 if dataset == "surveillance" else 100
            experiments.append((dataset, k, all_methods, n_trials))
        except (ValueError, IndexError):
            print("Invalid choice!")
            return
    
    elif choice == "4":
        # Custom
        print("\nSelect dataset:")
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds}")
        ds_choice = input("Dataset: ").strip()
        
        k_choice = input(f"K value ({', '.join(map(str, k_values))}): ").strip()
        
        print("\nSelect methods (comma-separated, e.g., 1,2,3):")
        for i, method in enumerate(all_methods, 1):
            print(f"  {i}. {method}")
        method_choices = input("Methods: ").strip()
        
        try:
            dataset = datasets[int(ds_choice) - 1]
            k = int(k_choice)
            if k not in k_values:
                print(f"Warning: k={k} is not standard ({k_values})")
            
            method_indices = [int(x.strip()) for x in method_choices.split(",")]
            selected_methods = [all_methods[i-1] for i in method_indices]
            
            n_trials = 10 if dataset == "surveillance" else 100
            trials_input = input(f"Number of trials (default: {n_trials}): ").strip()
            if trials_input:
                n_trials = int(trials_input)
            
            experiments.append((dataset, k, selected_methods, n_trials))
        except (ValueError, IndexError):
            print("Invalid input!")
            return
    
    elif choice.lower() == "q":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice!")
        return
    
    # Confirm and run
    print("\n" + "="*60)
    print("Experiments to run:")
    for i, (dataset, k, methods, n_trials) in enumerate(experiments, 1):
        print(f"\n{i}. {dataset} (k={k}, trials={n_trials})")
        print(f"   Methods: {', '.join(methods)}")
    
    print("\n" + "="*60)
    confirm = input("\nProceed? [y/N]: ").strip().lower()
    
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Run experiments
    results = []
    for dataset, k, methods, n_trials in experiments:
        success = run_experiment(dataset, k, methods, n_trials)
        results.append((dataset, k, success))
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    for dataset, k, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {dataset} k={k}")
    
    n_success = sum(1 for _, _, s in results if s)
    n_total = len(results)
    print(f"\nCompleted: {n_success}/{n_total}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(1)
