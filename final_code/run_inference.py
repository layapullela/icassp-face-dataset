#!/usr/bin/env python3
"""Reproduce the benchmark's test metrics from the reported hyperparameters.

This is INFERENCE ONLY: it reads the tuned hyperparameters out of
``hyperparameters.csv`` and evaluates each method on the held-out test
matrices. Nothing is tuned, so there is no grid search, no tuning pool, and no
risk of test data reaching a tuning objective -- that separation already
happened when the reported values were selected (README §4.1).

Examples
--------
    # one cell, fast -- good first check (~1 min)
    python run_inference.py --dataset ballet --k 3 --sigma 0 --fold 0

    # everything the CSV reports (hours; use --dataset to split the work)
    python run_inference.py --all

    # one method across all its cells
    python run_inference.py --all --methods OSC "DP NCut (trivial)"

Output is one row per (method, test matrix) written to
``results/inference_<tag>.csv``, with the same ``acc``/``nmi``/``ari`` columns
the development tree recorded, so it can be diffed against the reference
values carried in ``hyperparameters.csv``. Compare with
``verify_reproduction.py``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sscbench import data_ballet, data_surveillance          # noqa: E402
from sscbench.config import METHODS, SIGMAS                  # noqa: E402
from sscbench.methods import params_from_row, predict        # noqa: E402
from sscbench.metrics import metrics                         # noqa: E402

HERE = Path(__file__).resolve().parent
HYPERPARAMS = HERE / "hyperparameters.csv"

FIELDS = ["dataset", "method", "k", "sigma", "fold", "example", "unit",
          "n_frames", "k_pred", "acc", "nmi", "ari", "seconds"]


def load_hyperparameters(path=HYPERPARAMS):
    if not path.exists():
        raise SystemExit(f"{path} not found -- run build_hyperparameters_csv.py")
    return list(csv.DictReader(open(path)))


def select(rows, dataset=None, methods=None, ks=None, sigmas=None, folds=None):
    def keep(r):
        return ((dataset is None or r["dataset"] == dataset)
                and (methods is None or r["method"] in methods)
                and (ks is None or int(r["k"]) in ks)
                and (sigmas is None or float(r["sigma"]) in sigmas)
                and (folds is None or int(r["fold"]) in folds))
    return [r for r in rows if keep(r)]


def run_rows(rows, writer, handle=None, progress=True):
    """Evaluate the selected hyperparameter rows, streaming results out.

    Rows are grouped so the expensive data build happens once per
    (dataset, k, sigma, fold) rather than once per method.
    """
    # Surveillance needs all 24 sequences decoded once; ~1-2 min, then reused.
    surv_loaded = None
    if any(r["dataset"] == "surveillance" for r in rows):
        print("loading surveillance sequences (once, shared across all cells)...")
        surv_loaded = data_surveillance.load_all_sequences(verbose=progress)

    ballet_segments = ballet_dirs = None
    if any(r["dataset"] == "ballet" for r in rows):
        ballet_segments = data_ballet.load_dancer_segments()
        ballet_dirs = data_ballet.name_to_dir_map()

    cells = {}
    for r in rows:
        cells.setdefault(
            (r["dataset"], int(r["k"]), float(r["sigma"]), int(r["fold"])), []
        ).append(r)

    n_done = 0
    for (dataset, k, sigma, fold), cell_rows in sorted(cells.items()):
        if dataset == "surveillance":
            mats = [(name, Y, labs, ",".join(nms)) for name, Y, labs, nms
                    in data_surveillance.fold_test_matrices(
                        surv_loaded, k, sigma, fold)]
            units = [(i, m[0], m[1], m[2], m[3]) for i, m in enumerate(mats)]
        else:
            units = [(idx, f"dancer{dancer}", Y, labels, " ".join(descs))
                     for dancer, idx, Y, labels, descs
                     in data_ballet.fold_test_matrices(
                         ballet_segments, k, sigma, fold, ballet_dirs)]

        print(f"\n=== {dataset} k={k} sigma={sigma:g} fold={fold}: "
              f"{len(units)} test matrices x {len(cell_rows)} method(s) ===")
        for r in cell_rows:
            method = r["method"]
            params = params_from_row(method, r)
            for ex, unit, Y, labels, _desc in units:
                t0 = time.perf_counter()
                pred = predict(method, Y, k, params)
                elapsed = time.perf_counter() - t0
                m = metrics(labels, pred)
                writer.writerow({
                    "dataset": dataset, "method": method, "k": k,
                    "sigma": f"{sigma:g}", "fold": fold, "example": ex,
                    "unit": unit, "n_frames": Y.shape[1],
                    "k_pred": m["k_pred"], "acc": f"{m['acc']:.6f}",
                    "nmi": f"{m['nmi']:.6f}", "ari": f"{m['ari']:.6f}",
                    "seconds": f"{elapsed:.2f}",
                })
                # Flush per row: these runs take hours, and an unflushed
                # buffer means a killed or hung job leaves an empty file with
                # nothing salvageable.
                if handle is not None:
                    handle.flush()
                n_done += 1
            print(f"  {method:22s} done ({len(units)} matrices)")
    return n_done


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("ballet", "surveillance"))
    ap.add_argument("--methods", nargs="+", choices=METHODS, metavar="METHOD",
                    help=f"any of: {'; '.join(METHODS)}")
    ap.add_argument("--k", type=int, nargs="+", dest="ks")
    ap.add_argument("--sigma", type=float, nargs="+", dest="sigmas",
                    choices=SIGMAS, metavar="SIGMA",
                    help=f"any of: {', '.join(f'{s:g}' for s in SIGMAS)}")
    ap.add_argument("--fold", type=int, nargs="+", dest="folds")
    ap.add_argument("--all", action="store_true",
                    help="run every row in hyperparameters.csv")
    ap.add_argument("--out", type=Path,
                    help="output CSV (default: results/inference_<tag>.csv)")
    ap.add_argument("--list", action="store_true",
                    help="list the selected cells and exit without running")
    args = ap.parse_args()

    if not args.all and not any((args.dataset, args.methods, args.ks,
                                 args.sigmas, args.folds)):
        ap.error("select something to run, or pass --all "
                 "(see --help for examples)")

    rows = select(load_hyperparameters(), args.dataset, args.methods,
                  args.ks, args.sigmas, args.folds)
    if not rows:
        raise SystemExit("no hyperparameter rows matched that selection")

    if args.list:
        for r in rows:
            print(f"{r['dataset']:13s} {r['method']:20s} k={r['k']} "
                  f"sigma={r['sigma']:4s} fold={r['fold']}  "
                  f"ref_ari={r['ref_test_ari']}")
        print(f"\n{len(rows)} row(s) selected")
        return

    if args.out is None:
        bits = [args.dataset or "all"]
        if args.ks:
            bits.append("k" + "".join(str(k) for k in args.ks))
        if args.sigmas:
            bits.append("s" + "".join(f"{s:g}" for s in args.sigmas))
        if args.folds:
            bits.append("f" + "".join(str(f) for f in args.folds))
        args.out = HERE / "results" / f"inference_{'_'.join(bits)}.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{len(rows)} hyperparameter row(s) selected -> {args.out}")
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        n = run_rows(rows, writer, handle=f)
    print(f"\nwrote {n} result rows to {args.out}")
    print("compare against the reported reference values with:\n"
          f"  python verify_reproduction.py --results {args.out}")


if __name__ == "__main__":
    main()
