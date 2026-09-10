#!/usr/bin/env python3
"""Regenerate ``hyperparameters.csv`` from the development result tree.

This is the ONLY script here that reads the original sprawling tree
(``../block_ssc_benchmark/cv/``). Run it when new tuning jobs land, then
commit the refreshed CSV; everything else in this package reads the CSV, not
the tree.

What it does: for each (dataset, method, k, sigma, fold) it finds the CSV that
``plot_fold0_figure.py`` would have read, pulls the tuned hyperparameters out
of that file's ``params`` column, and records them alongside the reference
test metrics so a reproduction can be checked against them.

Which folds are reported (matching what the runs can actually support):

  * ballet -- ALL folds (7 at k=3, 4 at k=5). Those jobs ran to completion.
  * surveillance -- FOLD 0 ONLY. The full-CV surveillance runs were stopped
    part-way to save compute, so cells have differing fold coverage; fold 0 is
    the one fold present in every cell, which makes all five methods
    comparable on identical data.

Two provenance wrinkles are recorded in the ``notes`` column rather than
silently smoothed over -- see the module constants below and README §8.
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sscbench.config import (          # noqa: E402
    BLOCK_SIZE_BY_K,
    K_VALUES,
    LAMBDA_E_PIN,
    METHODS,
    METHOD_SLUG,
    SIGMAS,
    sigma_tag,
)

HERE = Path(__file__).resolve().parent
DEFAULT_TREE = HERE.parent / "block_ssc_benchmark" / "cv"

# TKSS's widened-d rerun (README §8.9) was cancelled part-way, and the files it
# did write live in this quarantine directory rather than in results/. The
# figure prefers them wherever they carry fold-0 rows, because they are the
# grid the documented procedure specifies; the cells where that file is empty
# fall back to the older narrow-d grid and are flagged.
QUARANTINE = "cancelled_partial_quarantine"

# The quarantine's two empty cells (surveillance k=8, sigma in {0, 0.25}) were
# specifically backfilled afterwards -- one job per cell, --only-fold 0, widened
# d grid -- rather than rerunning the full (expensive, ~8h/cell) original job.
# Output lives in a directory of its own, a sibling of cv/ rather than inside
# it (unlike QUARANTINE, which is a subdirectory of the tree), so it is
# addressed as an absolute path rather than relative to --tree. This is
# preferred over the quarantine fallback for k=8/sigma in {0, 0.25} because it
# is the widened grid where the quarantine file for those two cells is empty.
K8_BACKFILL = HERE.parent / "block_ssc_benchmark" / "cv_tkss_k8_widened"

# Column order: identity, then hyperparameters, then reference metrics.
FIELDS = [
    "dataset", "method", "k", "sigma", "fold",
    # SSC-Block-TV-Col21
    "block_size", "lambda_e", "lambda_z", "gamma_q",
    # OSC / BD-OSC
    "lambda_1", "lambda_2", "gamma_1", "p", "max_iter",
    # TKSS
    "d", "lam", "s",
    # reference values + provenance
    "n_test_matrices", "ref_test_ari", "ref_test_acc", "source_file", "notes",
]

# Per-matrix reference rows use run_inference.py's schema exactly, so the
# figure and verification scripts cannot tell a reference file from a freshly
# computed one.
REFERENCE_FIELDS = ["dataset", "method", "k", "sigma", "fold", "example",
                    "unit", "n_frames", "k_pred", "acc", "nmi", "ari",
                    "seconds"]


def parse_params(text):
    """'gamma_q=1, lambda_z=0.01, lambda_e=0.01' -> {'gamma_q': '1', ...}."""
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, val = part.partition("=")
        out[key.strip()] = val.strip()
    return out


def _unique_match(glob_pattern):
    """Unique non-backup CSV matching an absolute glob pattern, or None.

    Refuses an ambiguous match rather than taking whichever the filesystem
    returns first: the tree carries several superseded generations under
    similar names, and ``glob`` order is not defined.
    """
    fs = [f for f in glob.glob(glob_pattern) if "backup" not in f]
    if not fs:
        return None
    if len(fs) > 1:
        raise SystemExit(
            f"ambiguous match for {glob_pattern!r}:\n  "
            + "\n  ".join(sorted(Path(f).name for f in fs))
            + "\nResolve by moving the superseded file aside."
        )
    return Path(fs[0])


def pick_file(pattern, tree):
    """Unique non-backup CSV matching a glob relative to ``tree``, or None."""
    return _unique_match(str(tree / pattern))


def pick_backfill(pattern):
    """Unique non-backup CSV matching a glob under K8_BACKFILL, or None."""
    return _unique_match(str(K8_BACKFILL / pattern))


def read_rows(path, fold=None):
    rows = [r for r in csv.DictReader(open(path)) if r["split"] == "test"]
    if fold is not None:
        rows = [r for r in rows if r["fold"] == str(fold)]
    return rows


def surveillance_cell(tree, method, k, sigma):
    """Fold-0 test rows for one surveillance cell, mirroring the figure's logic."""
    slug, stag = METHOD_SLUG[method], sigma_tag(sigma)
    notes = []
    if method == "TKSS":
        # 1. The one-off backfill (k=8, sigma in {0, 0.25}: the two cells the
        #    quarantine's own rerun never reached before cancellation).
        backfilled = pick_backfill(f"results/surveillance/*_tkw_cv_tkss_k{k}_sigma{stag}.csv")
        if backfilled is not None:
            rows = read_rows(backfilled, fold=0)
            if rows:
                return rows, backfilled, [
                    "widened d grid {1,4,8,11,15,25,40,60}; backfilled "
                    "2026-09-10 via a dedicated --only-fold 0 job after the "
                    "original widened-d rerun for this cell was cancelled"
                ]
        # 2. The main widened-d rerun, wherever it survived cancellation.
        wide = pick_file(f"{QUARANTINE}/*_gs_cv_tkss_k{k}_sigma{stag}.csv", tree)
        if wide is not None:
            rows = read_rows(wide, fold=0)
            if rows:
                return rows, wide, ["widened d grid {1,4,8,11,15,25,40,60}"]
        notes.append(
            "NARROW d grid {1,4,8,11,15}: the widened-d rerun for this cell "
            "was cancelled before it wrote fold 0"
        )
    path = pick_file(f"results/surveillance/*_cv_{slug}_k{k}_sigma{stag}.csv", tree)
    if path is None:
        return None, None, notes
    return read_rows(path, fold=0), path, notes


def ballet_cell(tree, method, k, sigma):
    """All test rows for one ballet cell, grouped by fold."""
    slug, stag = METHOD_SLUG[method], sigma_tag(sigma)
    path = pick_file(f"results/ballet/*_cv_{slug}_k{k}_sigma{stag}.csv", tree)
    if path is None:
        return None, None
    by_fold = defaultdict(list)
    for r in read_rows(path):
        by_fold[int(r["fold"])].append(r)
    return by_fold, path


def reference_records(dataset, method, k, sigma, fold, rows):
    """The cell's per-matrix rows, translated into run_inference.py's schema."""
    out = []
    for r in rows:
        unit = (r.get("people") or "" if dataset == "surveillance"
                else f"dancer{r.get('dancer')}")
        out.append({
            "dataset": dataset, "method": method, "k": k,
            "sigma": f"{sigma:g}", "fold": fold, "example": r["example"],
            "unit": unit, "n_frames": r["n_frames"], "k_pred": r["k_pred"],
            "acc": r["acc"], "nmi": r["nmi"], "ari": r["ari"],
            "seconds": r.get("seconds", ""),
        })
    return out


def emit(dataset, method, k, sigma, fold, rows, path, notes):
    """One CSV row from one cell's per-matrix result rows."""
    distinct = {tuple(sorted(parse_params(r["params"]).items())) for r in rows}
    if len(distinct) > 1:
        raise SystemExit(
            f"{dataset} {method} k={k} sigma={sigma} fold={fold}: rows "
            f"disagree on hyperparameters ({sorted(distinct)}) -- this cell's "
            "CSV mixes runs, which means one of them is stale."
        )
    p = parse_params(rows[0]["params"])
    notes = list(notes)

    rec = {f: "" for f in FIELDS}
    rec.update(
        dataset=dataset, method=method, k=k, sigma=f"{sigma:g}", fold=fold,
        n_test_matrices=len(rows),
        ref_test_ari=f"{statistics.mean(float(r['ari']) for r in rows):.6f}",
        ref_test_acc=f"{statistics.mean(float(r['acc']) for r in rows):.6f}",
        source_file=Path(path).name,
    )
    for key, val in p.items():
        if key not in rec:
            raise SystemExit(f"unexpected hyperparameter {key!r} in {path}")
        rec[key] = val

    # block_size is pinned via a CLI flag, so it never appears in the params
    # column -- fill it from the pin table that the submit scripts used.
    if method == "SSC-Block-TV-Col21":
        rec["block_size"] = BLOCK_SIZE_BY_K[dataset][k]
        if float(rec["lambda_e"]) != LAMBDA_E_PIN:
            notes.append(
                f"lambda_e={rec['lambda_e']} (not the {LAMBDA_E_PIN} pin): "
                "this cell predates the unified pin and was not rerun"
            )
        if dataset == "surveillance" and k == 8:
            notes.append(
                "block_size=3 pin; the block-size-searched probe later showed "
                "2 is better at k=8 at every sigma"
            )
    rec["notes"] = "; ".join(notes)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, default=DEFAULT_TREE,
                    help="development cv/ directory to read (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=HERE / "hyperparameters.csv")
    args = ap.parse_args()

    if not args.tree.exists():
        raise SystemExit(f"result tree not found: {args.tree}")

    out_rows, reference_rows, missing = [], [], []
    for method in METHODS:
        for k in K_VALUES["surveillance"]:
            for sigma in SIGMAS:
                rows, path, notes = surveillance_cell(args.tree, method, k, sigma)
                if not rows:
                    missing.append(f"surveillance {method} k={k} sigma={sigma:g}")
                    continue
                out_rows.append(emit("surveillance", method, k, sigma, 0,
                                     rows, path, notes))
                reference_rows += reference_records(
                    "surveillance", method, k, sigma, 0, rows)
    for method in METHODS:
        for k in K_VALUES["ballet"]:
            for sigma in SIGMAS:
                by_fold, path = ballet_cell(args.tree, method, k, sigma)
                if not by_fold:
                    missing.append(f"ballet {method} k={k} sigma={sigma:g}")
                    continue
                for fold in sorted(by_fold):
                    out_rows.append(emit("ballet", method, k, sigma, fold,
                                         by_fold[fold], path, []))
                    reference_rows += reference_records(
                        "ballet", method, k, sigma, fold, by_fold[fold])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    # The per-matrix reference rows, in run_inference.py's own output schema so
    # make_figures.py and verify_reproduction.py treat reference and freshly
    # computed results identically. The figure's whiskers are bootstrapped, so
    # they need these individual values, not just each cell's mean.
    ref_path = args.out.parent / "results" / "reference_metrics.csv"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ref_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REFERENCE_FIELDS)
        w.writeheader()
        w.writerows(reference_rows)

    n_surv = sum(1 for r in out_rows if r["dataset"] == "surveillance")
    n_ball = len(out_rows) - n_surv
    print(f"wrote {args.out}")
    print(f"wrote {ref_path}  ({len(reference_rows)} per-matrix reference rows)")
    print(f"  {len(out_rows)} rows: {n_surv} surveillance (fold 0) "
          f"+ {n_ball} ballet (all folds)")
    flagged = [r for r in out_rows if r["notes"]]
    if flagged:
        print(f"  {len(flagged)} rows carry provenance notes:")
        for r in flagged:
            print(f"    {r['dataset']:13s} {r['method']:20s} k={r['k']} "
                  f"sigma={r['sigma']:4s} fold={r['fold']}: {r['notes']}")
    if missing:
        print(f"  WARNING: {len(missing)} cell(s) had no result file:")
        for m in missing:
            print(f"    {m}")


if __name__ == "__main__":
    main()
