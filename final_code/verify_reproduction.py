#!/usr/bin/env python3
"""Check a reproduction against the reported reference numbers.

Two independent checks, both cheap:

  --check-solvers   the six solver modules in sscbench/solvers/ are byte-identical
                    to the development-tree originals they were copied from.
  --results FILE    a run_inference.py output agrees, per cell, with the
                    ``ref_test_ari`` / ``ref_test_acc`` values recorded in
                    hyperparameters.csv.

The numeric check compares CELL MEANS (the quantity the figures plot), and
also reports the worst per-matrix disagreement, which is the more sensitive
signal: identical hyperparameters on identically-rebuilt data should agree to
floating-point noise, so a per-matrix delta above ~1e-6 means the data build
diverged, not that the solver is nondeterministic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLVER_DIR = HERE / "sscbench" / "solvers"
ORIGINALS = HERE.parent / "surveillance_dataset"
SOLVERS = ["ssc_block_tv_col21.py", "ssc_tv.py", "osc.py", "bdosc.py",
           "tkss.py", "dp_contiguous_partition.py"]


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def check_solvers():
    """Confirm the solver copies are unmodified."""
    print("solver integrity (vs the development-tree originals):")
    ok = True
    for name in SOLVERS:
        mine, orig = SOLVER_DIR / name, ORIGINALS / name
        if not mine.exists():
            print(f"  MISSING  {name}")
            ok = False
            continue
        h = md5(mine)
        if not orig.exists():
            print(f"  {name:30s} md5={h}  (original not reachable to compare)")
            continue
        same = h == md5(orig)
        ok &= same
        print(f"  {'OK  ' if same else 'DIFF'}     {name:30s} md5={h}")
    return ok


def load_reference(path):
    """(dataset, method, k, sigma, fold) -> row from hyperparameters.csv."""
    return {(r["dataset"], r["method"], int(r["k"]), float(r["sigma"]),
             int(r["fold"])): r for r in csv.DictReader(open(path))}


def check_results(results_path, hyper_path, tol):
    ref = load_reference(hyper_path)
    per_matrix = defaultdict(dict)
    cells = defaultdict(list)
    with open(results_path) as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["method"], int(r["k"]), float(r["sigma"]),
                   int(r["fold"]))
            cells[key].append(r)
            per_matrix[key][r["example"]] = r

    # Optional per-matrix comparison, when the reference export is present.
    ref_matrix_path = HERE / "results" / "reference_metrics.csv"
    ref_matrix = defaultdict(dict)
    if ref_matrix_path.exists() and ref_matrix_path != results_path:
        with open(ref_matrix_path) as f:
            for r in csv.DictReader(f):
                key = (r["dataset"], r["method"], int(r["k"]),
                       float(r["sigma"]), int(r["fold"]))
                ref_matrix[key][r["example"]] = r

    print(f"\ncell means vs hyperparameters.csv  (tolerance {tol}):")
    print(f"  {'dataset':13s} {'method':20s} {'k':>2} {'sigma':>5} {'fold':>4} "
          f"{'n':>3} {'ari':>9} {'ref':>9} {'d_ari':>9} {'d_acc':>9}")
    n_ok = n_bad = 0
    worst_matrix = (0.0, None)
    for key in sorted(cells):
        rows = cells[key]
        r = ref.get(key)
        if r is None:
            print(f"  no reference row for {key}")
            n_bad += 1
            continue
        ari = statistics.mean(float(x["ari"]) for x in rows)
        acc = statistics.mean(float(x["acc"]) for x in rows)
        d_ari = abs(ari - float(r["ref_test_ari"]))
        d_acc = abs(acc - float(r["ref_test_acc"]))
        good = d_ari <= tol and d_acc <= tol
        n_ok, n_bad = n_ok + good, n_bad + (not good)
        flag = "" if good else "   <-- MISMATCH"
        print(f"  {key[0]:13s} {key[1]:20s} {key[2]:>2} {key[3]:>5g} "
              f"{key[4]:>4} {len(rows):>3} {ari:9.6f} "
              f"{float(r['ref_test_ari']):9.6f} {d_ari:9.6f} {d_acc:9.6f}{flag}")
        for ex, row in per_matrix[key].items():
            other = ref_matrix.get(key, {}).get(ex)
            if other is None:
                continue
            d = abs(float(row["ari"]) - float(other["ari"]))
            if d > worst_matrix[0]:
                worst_matrix = (d, (*key, ex))

    if n_ok == 0 and n_bad == 0:
        # An empty or unparsed results file must not read as success -- that
        # is exactly the case a verification script exists to catch.
        print("\n  NO CELLS COMPARED -- the results file has no rows the "
              "hyperparameters cover. Nothing was verified.")
        return False
    print(f"\n  {n_ok} cell(s) matched, {n_bad} mismatched")
    if worst_matrix[1] is not None:
        d, where = worst_matrix
        verdict = ("floating-point noise" if d <= 1e-6
                   else "REAL DIVERGENCE -- the rebuilt data differs")
        print(f"  worst per-matrix ARI delta: {d:.2e} at {where}  ({verdict})")
    return n_bad == 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path,
                    help="run_inference.py output CSV to check")
    ap.add_argument("--hyperparameters", type=Path,
                    default=HERE / "hyperparameters.csv")
    ap.add_argument("--check-solvers", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max allowed cell-mean delta (default: %(default)s)")
    args = ap.parse_args()

    if not args.results and not args.check_solvers:
        ap.error("pass --results FILE and/or --check-solvers")

    ok = True
    if args.check_solvers:
        ok &= check_solvers()
    if args.results:
        if not args.results.exists():
            raise SystemExit(f"{args.results} not found")
        ok &= check_results(args.results, args.hyperparameters, args.tol)

    print("\nRESULT:", "all checks passed" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
