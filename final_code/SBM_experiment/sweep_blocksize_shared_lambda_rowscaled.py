#!/usr/bin/env python3
"""Test-set ARI vs block_size for WTV-SSC (spectral-norm D, the default
solver), with lambda_2 RESCALED per block_size to hold the effective
row-equivalent TV penalty fixed, instead of holding the raw lambda_2 fixed.

Motivation
----------
sweep_blocksize_shared_lambda.py holds lambda_2 literally constant across
block_size while using the spectral-norm-normalized D (||D||_2 = 1). But
||D_raw||_2 itself drifts only mildly with block_size (~1.54 -> ~1.04 over
the grid here), while a single row's norm (every row has the same norm by
construction) shrinks much faster (~1.0 -> ~0.26). So a literal fixed
lambda_2 under spectral norm corresponds to a progressively WEAKER
row-equivalent penalty as block_size grows -- part of the earlier ARI decay
at large block_size may be under-regularization, not genuine over-smoothing.

Fix: for each block_size, compute

    D_raw, _, _ = sliding_block_diff_matrix(N, block_size, normalize=False)
    f = ||D_raw||_2 / ||D_raw[len(D_raw)//2]||   (spectral norm / row norm)
    lambda_2_spec = f * LAMBDA_2_ROW_REF

so that lambda_2_spec * D_spec == LAMBDA_2_ROW_REF * D_row exactly (the same
physical penalty on C @ D_raw^T), while still running the spectral-norm
solver (different ADMM eigs/scale than norm_mode="row", so not numerically
identical to the row-norm sweep, but comparable in penalty magnitude).

LAMBDA_2_ROW_REF = 10**-0.5 matches the shared lambda_2 used in both
sweep_blocksize_shared_lambda.py and
sweep_blocksize_shared_lambda_rownorm.py, so this sweep is the row-equivalent
apples-to-apples spectral counterpart.

No-outlier variant: OUTLIER_FRAC = 0.0 -- no burst-outlier columns (the
original default is sbm.SEQ_OUTLIER_FRAC = 0.20).

N=90, K=3 equal-size blocks variant: BLOCK_SIZES = [30, 30, 30] (was the
unequal [22, 33, 45], N=100, shared with benchmark_sbm.py).
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
FINAL_CODE = HERE.parent
sys.path.insert(0, str(FINAL_CODE))
sys.path.insert(0, str(HERE))

from sbm import make_sequential_instance  # noqa: E402
from sscbench.methods import run_ssc_sparse_block_tv  # noqa: E402
from ssc_block_tv_l1_sliding import sliding_block_diff_matrix  # noqa: E402

BLOCK_SIZES =[15,20,30,40,45]  #[24, 30, 36, 32, 28]  # K=3, N=90, equal-size clusters
K = len(BLOCK_SIZES)
N = sum(BLOCK_SIZES)
TEST_SEEDS = range(20, 40)
#lambda_1=0.1 lambda_2=2.1544346900318843 
WALK_STDS =  (0.4,) #(0, 0.05, 0.10, 0.15, 0.2, 0.25, 0.30)
LAMBDA_1 = 0.1 #10 ** -0.5
LAMBDA_2_ROW_REF = 1 #2.1544346900318843 #10 ** -0.5
SIGMA = 0.5
MIX = 0.25 #old: 0.25

# No burst-outlier columns for this rerun (sbm.SEQ_OUTLIER_FRAC = 0.20 is the
# original default).
OUTLIER_FRAC = 0.0

OUT_CSV = (HERE / "results" / "n90_k3_equal_hard_sparse_l1" /
           "blocksize_sweep_shared_lambda_rowscaled_no_outlier.csv")
BLOCK_SIZE_GRID = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30]


def row_to_spectral_scale(n, block_size):
    d_raw, _centers, _k_eff = sliding_block_diff_matrix(n, block_size, normalize=False)
    row_norm = float(np.linalg.norm(d_raw[len(d_raw) // 2]))
    spectral_norm = float(np.linalg.norm(d_raw, 2))
    return spectral_norm / row_norm


def main():
    fieldnames = ["walk_std", "lambda_1", "lambda_2_row_ref", "f_scale",
                  "lambda_2_spec", "block_size", "test_ari_mean",
                  "test_ari_std", "seconds"]
    scale = {bs: row_to_spectral_scale(N, bs) for bs in BLOCK_SIZE_GRID}
    print("block_size -> f = spectral/row scale:", flush=True)
    for bs in BLOCK_SIZE_GRID:
        print(f"  bs={bs:<3} f={scale[bs]:.4f}  "
              f"lambda_2_spec={scale[bs] * LAMBDA_2_ROW_REF:.4f}", flush=True)

    rows = []
    for walk_std in WALK_STDS:
        graphs = [make_sequential_instance(seed, walk_std, block_sizes=BLOCK_SIZES,
                                           outlier_frac=OUTLIER_FRAC, mix=MIX, sigma=SIGMA)
                  for seed in TEST_SEEDS]
        print(f"\n=== walk_std={walk_std:g}  lambda_1={LAMBDA_1:.4g}  "
              f"lambda_2_row_ref={LAMBDA_2_ROW_REF:.4g} (row-equivalent, "
              f"spectral D) ===", flush=True)
        for block_size in BLOCK_SIZE_GRID:
            f_scale = scale[block_size]
            lambda_2_spec = f_scale * LAMBDA_2_ROW_REF
            t0 = time.time()
            
            scores = [
                adjusted_rand_score(
                    labels,
                    run_ssc_sparse_block_tv(
                        Y, K, lambda_1=LAMBDA_1, lambda_2=lambda_2_spec,
                        block_size=block_size, max_iter=100
                    ),
                )
                for Y, labels in graphs
            ]
            elapsed = time.time() - t0
            mean_ari = float(np.mean(scores))
            std_ari = float(np.std(scores))
            rows.append({
                "walk_std": walk_std, "lambda_1": LAMBDA_1,
                "lambda_2_row_ref": LAMBDA_2_ROW_REF, "f_scale": f_scale,
                "lambda_2_spec": lambda_2_spec, "block_size": block_size,
                "test_ari_mean": mean_ari, "test_ari_std": std_ari,
                "seconds": elapsed,
            })
            print(f"  block_size={block_size:<3}  lambda_2_spec={lambda_2_spec:.3f}  "
                  f"test ARI={mean_ari:.3f} +/- {std_ari:.3f}  ({elapsed:.1f}s)",
                  flush=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
