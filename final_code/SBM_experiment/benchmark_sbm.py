"""Sequential-subspace benchmark: SSC-Sparse-Block-TV vs OSC, TKSS, DP-NCuts.

No-outlier, mix=0, sigma=1.5 variant (N=150, K=5, max_iter=100, matching
the block-size sweep; walk_std restricted to {0, 0.05, 0.10, 0.15, 0.20,
0.25, 0.30})
------------------------------------------------------------------------
Deltas from the base benchmark_sbm.py protocol:
  1. OUTLIER_FRAC = 0.0 -- no burst-outlier columns (the original had 20%
     of columns replaced by isotropic Gaussian in contiguous bursts).
  2. SEQ_SUBSPACE_MIX overridden to 0.0 -- identical subspaces across
     clusters (was 0.25, partially independent).
  3. SEQ_SIGMA overridden to 1.5 -- much higher cluster noise (was 0.3).
  4. BLOCK_SIZES = [24,30,36,32,28] (K=5, N=150, matches the block-size
     sweep) and SSC-Sparse-Block-TV max_iter=100 (was 50, ditto).
  5. WALK_STDS restricted to (0, 0.05, 0.10, 0.15, 0.2, 0.25, 0.30).
lambda_1, lambda_2, and block_size are each swept normally (independent
grid axes, no row-equivalent rescaling of lambda_2 by block_size) -- that
rescaling only applies to the separate ARI-vs-block_size experiment, see
sweep_blocksize_shared_lambda_rowscaled.py.

Protocol
--------
Generate Y in R^{D x N} as contiguous runs on K overlapping d-dimensional
subspaces (see sbm.make_sequential_instance). Cluster noise is sigma=0.3.
Subspaces share a common component (mix=0.25). The sweep is
node-to-node coefficient change: walk_std in {0.05, 0.25, 0.30, 0.35, 0.4, 0.5}.
For each walk_std, sample 10 train sequences (seeds 0-9) and 10 test
sequences (seeds 20-29). Tune one hyperparameter vector per
(method, walk_std) by mean train ARI; evaluate on the held-out test
sequences.

Grids use 7 points on every axis; no lambda below 1e-3:
SBTV lambda_1 logspace(-3,0,7), lambda_2 logspace(-3,2,7),
block_size {2,3,4,5,8,10,12}; OSC lambda_1 logspace(-3,1,7),
lambda_2 logspace(-3,1,7); TKSS d in {1,2,3,4,5,6,8},
lam logspace(-2,1,7), s in {1,2,3,4,6,8,12}.

Data: 3 contiguous unequal blocks [22, 33, 45] (N=100), D=10, subspace
dim 4, mix=0.25, 20% burst outliers (length 5), unit-norm columns, known k.
The proposed solver is ssc_admm_sparse_block_tv (||C D^T||_1 TV).

Usage
-----
    python benchmark_sbm.py
    python benchmark_sbm.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
FINAL_CODE = HERE.parent
sys.path.insert(0, str(FINAL_CODE))
sys.path.insert(0, str(HERE))

from sbm import (  # noqa: E402
    SEQ_D,
    SEQ_D_SUB,
    SEQ_OUTLIER_BURST,
    make_sequential_instance,
)
from sscbench.methods import (  # noqa: E402
    run_gram_ncut,
    run_osc,
    run_ssc_sparse_block_tv,
    run_tkss,
)

# No burst-outlier columns for this rerun (sbm.SEQ_OUTLIER_FRAC = 0.20 is the
# original default used by the outlier-bearing benchmark_sbm.py runs).
OUTLIER_FRAC = 0.0

# mix=0 -> identical subspaces across clusters (was SEQ_SUBSPACE_MIX=0.25).
# sigma=1.5 -> much higher cluster noise (was 0.3).
SEQ_SUBSPACE_MIX = 0.25
SEQ_SIGMA = 0.5

# Unequal contiguous blocks so TKSS's equal-length init is not the truth.
#BLOCK_SIZES = [22, 33, 45]  # K=3, N=100 (OLD leaving this in for my reference)

BLOCK_SIZES = [15,20,30,40,45] # K=5, matches block sweep
K = len(BLOCK_SIZES)
N = sum(BLOCK_SIZES)

# Node-to-node coefficient change. Cluster separation is pinned at
# SEQ_SIGMA = 0.3.
WALK_STDS = (0, ) #(0, 0.05, 0.10, 0.15, 0.2, 0.25, 0.30)
TRAIN_SEEDS = range(0, 10)
TEST_SEEDS = range(20, 40)

METHODS = ("SSC-Sparse-Block-TV", "OSC", "TKSS", "DP NCut (trivial)")

N_GRID = 7  # same number of points on every hyperparameter axis

# SSC-Sparse-Block-TV's lambda_1 grid was floored at 1e-3, and walk_std in
# {0.40, 0.50} tuned to exactly that floor -- a grid-edge artifact (the true
# optimum may lie below 1e-3). Extended down to 1e-5 at the SAME 0.5-decade
# spacing as the original 7-point grid, so this is a strict superset (every
# previously-tuned value, e.g. 0.316, is still reachable) with 4 new points
# prepended below the old floor -- not a resample of the whole range, which
# would silently drop prior optima. lambda_2/block_size/other-method grids
# are unchanged.
SSC_LAMBDA_1_GRID = np.logspace(-5, 0, 11).tolist()

# 7 points per axis. Continuous lambdas are floored at 1e-3.
# block_size=2 is a structural floor.
GRIDS = {
    "SSC-Sparse-Block-TV": {
        "lambda_1": SSC_LAMBDA_1_GRID,
        "lambda_2": np.logspace(-3, 2, N_GRID).tolist(),
        "block_size": [2, 3, 4, 5, 8, 10, 12],
    },
    "OSC": {
        "lambda_1": np.logspace(-3, 1, N_GRID).tolist(),
        "lambda_2": np.logspace(-3, 1, N_GRID).tolist(),
    },
    "TKSS": {
        "d": [1, 2, 3, 4, 5, 6, 8],
        "lam": np.logspace(-2, 1, N_GRID).tolist(),
        "s": [1, 2, 3, 4, 6, 8, 12],
    },
    "DP NCut (trivial)": {},
}

RUNNERS = {
    "SSC-Sparse-Block-TV": run_ssc_sparse_block_tv,
    "OSC": run_osc,
    "TKSS": run_tkss,
    "DP NCut (trivial)": run_gram_ncut,
}

OUT_DIR = HERE / "results" / "n150_k5_hard_sparse_extrange_l1_no_outlier_mix0_sigma15"


def grid_combos(spec):
    if not spec:
        return [dict()]
    keys = list(spec)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(spec[k] for k in keys))]


def predict(method, Y, params):
    if method == "DP NCut (trivial)":
        return run_gram_ncut(Y, K)
    if method == "SSC-Sparse-Block-TV":
        return RUNNERS[method](Y, K, max_iter=100, **params)
    return RUNNERS[method](Y, K, **params)


def aris_on(method, graphs, params):
    return [adjusted_rand_score(labels, predict(method, Y, params))
            for Y, labels in graphs]


def param_str(params):
    if not params:
        return "(none)"
    return " ".join(f"{k}={v}" for k, v in params.items())


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def rows_except_walks(rows, walk_stds):
    skip = {round(float(w), 4) for w in walk_stds}
    kept = []
    for row in rows:
        try:
            if round(float(row["walk_std"]), 4) in skip:
                continue
        except (KeyError, TypeError, ValueError):
            pass
        kept.append(row)
    return kept


def load_instances(walk_std, seeds):
    return [make_sequential_instance(seed, walk_std, block_sizes=BLOCK_SIZES,
                                     outlier_frac=OUTLIER_FRAC,
                                     mix=SEQ_SUBSPACE_MIX, sigma=SEQ_SIGMA)
            for seed in seeds]


def run_density(walk_std, train_seeds, test_seeds):
    print(f"\n=== N={N}  K={K}  D={SEQ_D}  d={SEQ_D_SUB}  "
          f"sigma={SEQ_SIGMA:g}  mix={SEQ_SUBSPACE_MIX:g}  "
          f"walk_std={walk_std:g}  outlier_frac={OUTLIER_FRAC:g}  "
          f"burst={SEQ_OUTLIER_BURST} ===", flush=True)
    t0 = time.time()
    train = load_instances(walk_std, train_seeds)
    test = load_instances(walk_std, test_seeds)
    print(f"  built {len(train)} train / {len(test)} sequences "
          f"({time.time() - t0:.1f}s)", flush=True)

    tune_rows = []
    best = {}
    for method in METHODS:
        combos = grid_combos(GRIDS[method])
        print(f"  {method}: {len(combos)} combo(s) x {len(train)} train",
              flush=True)
        ranked = []
        for params in combos:
            t1 = time.time()
            scores = aris_on(method, train, params)
            mean_ari = float(np.mean(scores))
            elapsed = time.time() - t1
            ranked.append((mean_ari, params, scores))
            tune_rows.append({
                "n": N, "k": K, "D": SEQ_D, "d_sub": SEQ_D_SUB,
                "mix": SEQ_SUBSPACE_MIX, "walk_std": walk_std,
                "sigma": SEQ_SIGMA, "outlier_frac": OUTLIER_FRAC,
                "burst_len": SEQ_OUTLIER_BURST, "method": method,
                "split": "train", **params,
                "mean_ari": mean_ari,
                "std_ari": float(np.std(scores)),
                "seconds": elapsed,
            })
            print(f"    train ARI={mean_ari:.3f}  {param_str(params)}  "
                  f"({elapsed:.1f}s)", flush=True)
        ranked.sort(key=lambda t: (-t[0], list(t[1].values())))
        best_ari, best_params, _ = ranked[0]
        best[method] = (best_params, best_ari)
        print(f"  best {method}: train ARI={best_ari:.3f}  "
              f"{param_str(best_params)}", flush=True)

    test_rows = []
    summary_rows = []
    for method in METHODS:
        params, train_ari = best[method]
        t1 = time.time()
        scores = aris_on(method, test, params)
        elapsed = time.time() - t1
        mean_ari = float(np.mean(scores))
        std_ari = float(np.std(scores))
        print(f"  test {method}: ARI={mean_ari:.3f} +/- {std_ari:.3f}  "
              f"{param_str(params)}  ({elapsed:.1f}s)", flush=True)
        for seed, ari in zip(test_seeds, scores):
            test_rows.append({
                "n": N, "k": K, "D": SEQ_D, "d_sub": SEQ_D_SUB,
                "mix": SEQ_SUBSPACE_MIX, "walk_std": walk_std,
                "sigma": SEQ_SIGMA, "outlier_frac": OUTLIER_FRAC,
                "burst_len": SEQ_OUTLIER_BURST, "method": method,
                "seed": seed, "ari": ari, **params,
            })
        summary_rows.append({
            "n": N, "k": K, "D": SEQ_D, "d_sub": SEQ_D_SUB,
            "mix": SEQ_SUBSPACE_MIX, "walk_std": walk_std,
            "sigma": SEQ_SIGMA, "outlier_frac": OUTLIER_FRAC,
            "burst_len": SEQ_OUTLIER_BURST, "method": method,
            **params,
            "train_ari": train_ari,
            "test_ari_mean": mean_ari,
            "test_ari_std": std_ari,
            "n_test": len(scores),
            "seconds": elapsed,
        })
    return tune_rows, test_rows, summary_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 train + 2 test sequences, walk_std=0.25 only")
    ap.add_argument("--walks", type=float, nargs="+", default=None,
                    help="Subset of walk_std values; default is the full sweep")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.smoke:
        walk_stds = (0.25,)
        train_seeds = range(0, 2)
        test_seeds = range(20, 22)
        out_dir = args.out_dir / "smoke"
    else:
        walk_stds = tuple(args.walks) if args.walks is not None else WALK_STDS
        train_seeds = TRAIN_SEEDS
        test_seeds = TEST_SEEDS
        out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"N={N}  K={K}  blocks={BLOCK_SIZES}", flush=True)
    print(f"D={SEQ_D}  d_sub={SEQ_D_SUB}  mix={SEQ_SUBSPACE_MIX}  "
          f"sigma={SEQ_SIGMA}  outlier_frac={OUTLIER_FRAC}  "
          f"burst={SEQ_OUTLIER_BURST}  walk_std={list(walk_stds)}  "
          f"n_train={len(list(train_seeds))}  n_test={len(list(test_seeds))}  "
          f"out={out_dir}", flush=True)
    for method in METHODS:
        n_combo = len(grid_combos(GRIDS[method]))
        print(f"  {method}: {n_combo} combo(s)", flush=True)

    tune_fields = ["n", "k", "D", "d_sub", "mix", "walk_std", "sigma",
                   "outlier_frac", "burst_len", "method", "split",
                   "lambda_1", "lambda_2", "block_size", "d", "lam", "s",
                   "mean_ari", "std_ari", "seconds"]
    test_fields = ["n", "k", "D", "d_sub", "mix", "walk_std", "sigma",
                   "outlier_frac", "burst_len", "method", "seed", "ari",
                   "lambda_1", "lambda_2", "block_size", "d", "lam", "s"]
    summary_fields = ["n", "k", "D", "d_sub", "mix", "walk_std", "sigma",
                      "outlier_frac", "burst_len", "method",
                      "lambda_1", "lambda_2", "block_size", "d", "lam", "s",
                      "train_ari", "test_ari_mean", "test_ari_std",
                      "n_test", "seconds"]

    tune_rows = rows_except_walks(load_csv(out_dir / "sbm_tune_grid.csv"), walk_stds)
    test_rows = rows_except_walks(load_csv(out_dir / "sbm_test.csv"), walk_stds)
    summary_rows = rows_except_walks(load_csv(out_dir / "sbm_summary.csv"), walk_stds)
    for walk_std in walk_stds:
        t, te, s = run_density(walk_std, train_seeds, test_seeds)
        tune_rows.extend(t)
        test_rows.extend(te)
        summary_rows.extend(s)
        write_csv(out_dir / "sbm_tune_grid.csv", tune_rows, tune_fields)
        write_csv(out_dir / "sbm_test.csv", test_rows, test_fields)
        write_csv(out_dir / "sbm_summary.csv", summary_rows, summary_fields)

    method_rank = {m: i for i, m in enumerate(METHODS)}
    summary_rows.sort(key=lambda r: (float(r["walk_std"]),
                                     method_rank.get(r["method"], 99)))
    write_csv(out_dir / "sbm_summary.csv", summary_rows, summary_fields)

    print("\n=== test ARI summary ===", flush=True)
    print(f"{'walk':>6}  {'method':<24}  {'train':>6}  {'test':>6}  "
          f"{'std':>6}  hyps", flush=True)
    for row in summary_rows:
        hyps = {k: row[k] for k in
                ("lambda_1", "lambda_2", "block_size", "d", "lam", "s")
                if k in row and row[k] not in (None, "")}
        print(f"{float(row['walk_std']):6.2f}  {row['method']:<24}  "
              f"{float(row['train_ari']):6.3f}  {float(row['test_ari_mean']):6.3f}  "
              f"{float(row['test_ari_std']):6.3f}  {param_str(hyps)}",
              flush=True)
    print(f"\nwrote {out_dir / 'sbm_tune_grid.csv'}", flush=True)
    print(f"wrote {out_dir / 'sbm_test.csv'}", flush=True)
    print(f"wrote {out_dir / 'sbm_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
