# SSC-Block-TV Variants Benchmark

This directory contains variants of the SSC-Block-TV method for temporal subspace clustering experiments.

Jobs **default to `--known-k`**: the true number of clusters is passed to DP NCut (no eigengap). This matches `submit_block_compare.sh`. To estimate k instead, pass `--khat` to the SLURM submitter.

## New Variants Created

### 1. **SSC-Block-TV** (original)
   - File: `ssc_block_tv.py`
   - Error term: **||E||_{2,1}^{block}** (block-wise segment sparsity)
   - TV term: **||C Db^T||_1** (block finite-difference on C)

### 2. **SSC-Block-TV-Col21** (NEW)
   - File: `ssc_block_tv_col21.py`
   - Error term: **||E||_{2,1}** (standard column-wise sparsity)
   - TV term: **||C Db^T||_1** (block finite-difference on C)
   - Each column of E is shrunk independently by its L2 norm

### 3. **SSC-Block-TV-L1** (NEW)
   - File: `ssc_block_tv_l1.py`
   - Error term: **||E||_1** (element-wise sparsity)
   - TV term: **||C Db^T||_1** (block finite-difference on C)
   - Each element of E is soft-thresholded independently

## Files Created

### Algorithm Implementations
Canonical copies live in `surveillance_dataset/`. Ballet imports them via `sys.path` (do not maintain a second copy).

- `surveillance_dataset/ssc_block_tv.py` - Original block L2,1 on E
- `surveillance_dataset/ssc_block_tv_col21.py` - Column-wise L2,1 variant
- `surveillance_dataset/ssc_block_tv_l1.py` - Element-wise L1 variant

### Experiment Scripts
- `run_block_variants_benchmark.sh` - Bash script to run all experiments locally (`--known-k`)
- `submit_block_variants_slurm.py` - Python script to submit SLURM jobs (`--known-k` by default)

### Updated Files
- `surveillance_dataset/cluster_experiment.py` - Added new variants to benchmarking
- `ballet_dataset/cluster_experiment.py` - Added new variants to benchmarking

## Running the Experiments

### Option 1: Local Sequential Execution (Bash)

```bash
cd /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2
chmod +x run_block_variants_benchmark.sh
./run_block_variants_benchmark.sh
```

This will run all experiments sequentially for k = 5, 8, 12 on both datasets, with `--known-k`.

### Option 2: SLURM Parallel Submission (Recommended)

```bash
cd /nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2
python submit_block_variants_slurm.py
```

This will:
1. Create SLURM submission scripts in `slurm_scripts/`
2. Submit 6 jobs (2 datasets × 3 k values), each with `--known-k`
3. Each job runs independently in parallel

Estimate k with eigengap instead:

```bash
python submit_block_variants_slurm.py --khat
```

**Before submitting**, edit `submit_block_variants_slurm.py` to adjust:
- SLURM account name
- Partition name
- Time/memory/CPU requirements

### Option 3: Manual Execution (Individual Jobs)

#### Surveillance Dataset
```bash
cd surveillance_dataset

# k=5
python cluster_experiment.py --k 5 --known-k --sigmas 0.0 --methods SSC-Block-TV SSC-Block-TV-Col21 SSC-Block-TV-L1 \
    --n-trials 10 --out-tag k5_variants

# k=8
python cluster_experiment.py --k 8 --known-k --sigmas 0.0 --methods SSC-Block-TV SSC-Block-TV-Col21 SSC-Block-TV-L1 \
    --n-trials 10 --out-tag k8_variants

# k=12
python cluster_experiment.py --k 12 --known-k --sigmas 0.0 --methods SSC-Block-TV SSC-Block-TV-Col21 SSC-Block-TV-L1 \
    --n-trials 10 --out-tag k12_variants
```

#### Ballet Dataset
```bash
cd ballet_dataset

# k=5
python cluster_experiment.py --k 5 --known-k --methods SSC-Block-TV SSC-Block-TV-Col21 SSC-Block-TV-L1 \
    --n-trials 100 --out-tag k5_variants

# k=8
python cluster_experiment.py --k 8 --known-k --methods SSC-Block-TV SSC-Block-TV-Col21 SSC-Block-TV-L1 \
    --n-trials 100 --out-tag k8_variants

# k=12
python cluster_experiment.py --k 12 --known-k --methods SSC-Block-TV SSC-Block-TV-Col21 SSC-Block-TV-L1 \
    --n-trials 100 --out-tag k12_variants
```

## Output Files

Results will be saved with the `--out-tag` suffix. `--known-k` rewrites `khat` → `knownk` in the stem:

### Surveillance Dataset
- `split_k5_clean_min30_trials10_knownk_k5_variants.csv` - Results for k=5
- `split_k5_clean_min30_trials10_knownk_k5_variants_params.json` - Hyperparameters for k=5
- (Similar for k=8, k=12)

### Ballet Dataset
- `ballet_cluster_knownk_scaled_results_k5_variants.csv` - Results for k=5
- `ballet_cluster_knownk_scaled_params_k5_variants.json` - Hyperparameters for k=5
- (Similar for k=8, k=12)

## Experiment Configuration

### Both datasets
- **known k**: True (pass `--khat` to the SLURM submitter to estimate k)

### Surveillance Dataset
- **Trials**: 10 (Optuna hyperparameter search)
- **Noise levels**: σ = 0.0 only (`--sigmas 0.0`)
- **Frame sampling**: 30-100 frames per person
- **Train/test split**: 4 groups train, 1 group test

### Ballet Dataset
- **Trials**: 100 (Optuna hyperparameter search)
- **Noise levels**: Clean (σ = 0.0)
- **Frame sampling**: 10-100 frames per sequence
- **Train/test split**: 24 train combos, 18 test combos

## Monitoring Jobs

Check SLURM job status:
```bash
squeue -u $USER
```

View job output:
```bash
tail -f surveillance_dataset/logs/ssc_block_surveillance_k5_*.out
tail -f ballet_dataset/logs/ssc_block_ballet_k5_*.out
```

Cancel jobs if needed:
```bash
scancel <job_id>
# or cancel all your jobs:
scancel -u $USER
```

## Expected Runtime

- **Surveillance k=5**: ~2-4 hours
- **Surveillance k=8**: ~3-6 hours
- **Surveillance k=12**: ~4-8 hours
- **Ballet k=5**: ~6-12 hours
- **Ballet k=8**: ~8-16 hours
- **Ballet k=12**: ~10-20 hours

*Note: Actual runtime depends on cluster load and hardware.*
