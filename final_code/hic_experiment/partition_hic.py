#!/usr/bin/env python3
"""Partition Hi-C contact maps into k contiguous domains using
SSC-Block-TV-Col21 (final_code/sscbench/solvers/ssc_block_tv_col21.py)
followed by DP-NCut contiguous post-processing
(final_code/sscbench/solvers/dp_contiguous_partition.py).

Fast train/test design
-----------------------
Loading a whole .hic chromosome via repeated per-window hicstraw queries is
slow. Instead this script:

1. Loads ONE chromosome's contact matrix ONCE (a single hicstraw query,
   10kbp resolution, KR-balanced) and keeps it in memory as a dense matrix.
2. Slices out non-overlapping n_bins x n_bins (default 200x200 = 2Mbp)
   windows from that in-memory matrix -- no further I/O.
3. Keeps only well-covered windows (few all-empty rows), takes the first
   `--n-train` as the tuning set and the next `--n-test` as the held-out
   test set.
4. Grid-searches (block_size, lambda_e) on the TRAIN windows only: for each
   candidate, runs the ADMM solver + DP-NCut on every train window and
   scores it by the mean insulation score at the k-1 predicted boundaries
   (lower insulation = a real domain boundary), averaged across train
   windows. lambda_z, gamma_q are held at solver defaults.
5. Re-evaluates the single best (block_size, lambda_e) on the TEST windows
   and reports per-window boundaries + insulation objective there.

Insulation score is a Crane-et-al.-style sliding-window log2 score computed
directly from the contact matrix (fanc is not installed/needed).

Run with an interpreter that has hicstraw + numpy (e.g. the `test_env`
conda env). sklearn is NOT required (only used in solver __main__ blocks).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SOLVERS_DIR = Path(
    "/nfs/turbo/umms-minjilab/lpullela/icassp_ssc_tv2/final_code/sscbench/solvers"
)
sys.path.insert(0, str(SOLVERS_DIR))

from ssc_block_tv_col21 import ssc_admm_block_tv_col21  # noqa: E402
from dp_contiguous_partition import dp_contiguous_ncut_partition  # noqa: E402

import hicstraw  # noqa: E402


# ── Data loading (once per chromosome) ──────────────────────────────────────

def load_chrom_matrix(hic_path, chrom, resolution, norm):
    """Single hicstraw query for a whole chromosome -> dense symmetric matrix."""
    t0 = time.time()
    recs = hicstraw.straw("observed", norm, hic_path, chrom, chrom, "BP", resolution)
    n_bins = int(np.ceil(
        {c.name: c.length for c in hicstraw.HiCFile(hic_path).getChromosomes()}[chrom]
        / resolution
    ))
    binX = np.fromiter((r.binX for r in recs), dtype=np.int64, count=len(recs)) // resolution
    binY = np.fromiter((r.binY for r in recs), dtype=np.int64, count=len(recs)) // resolution
    counts = np.fromiter((r.counts for r in recs), dtype=np.float64, count=len(recs))
    finite = np.isfinite(counts)
    binX, binY, counts = binX[finite], binY[finite], counts[finite]

    M = np.full((n_bins, n_bins), np.nan)
    M[binX, binY] = counts
    M[binY, binX] = counts
    print(f"  loaded chr{chrom}: {n_bins} bins, {len(recs)} records in {time.time() - t0:.1f}s")
    return M


def find_windows(M, n_bins, max_empty_frac, n_wanted, start_bin=0, stride_bins=None):
    """Slice non-overlapping n_bins windows out of M, keep well-covered ones."""
    if stride_bins is None:
        stride_bins = n_bins
    return find_windows_in_range(
        M, n_bins, max_empty_frac, start_bin, M.shape[0], stride_bins, n_wanted,
    )


def find_windows_in_range(M, n_bins, max_empty_frac, start, end, stride_bins, n_wanted=None):
    """Well-covered n_bins windows with start+n_bins <= end."""
    windows = []
    s = start
    while s + n_bins <= end and (n_wanted is None or len(windows) < n_wanted):
        sub = M[s:s + n_bins, s:s + n_bins]
        row_empty = np.isnan(sub).all(axis=1)
        if row_empty.mean() <= max_empty_frac:
            windows.append((s, sub))
        s += stride_bins
    return windows


# ── Insulation score (Crane et al. 2015 style, computed directly) ─────────

def insulation_score_log2(M, w):
    """Sliding-window insulation score.

    For bin index i (w <= i < n-w), the score is the mean contact value in
    the w x w square straddling the diagonal between bin i-1 and bin i:
        IS_raw[i] = mean(M[i-w:i, i:i+w])
    Reported as log2(IS_raw / mean(IS_raw over valid bins)); low values mark
    insulating (domain-boundary) positions, matching fanc/cooltools convention.
    """
    n = M.shape[0]
    raw = np.full(n, np.nan)
    for i in range(w, n - w):
        sub = M[i - w:i, i:i + w]
        vals = sub[np.isfinite(sub)]
        if vals.size:
            raw[i] = vals.mean()
    valid = np.isfinite(raw) & (raw > 0)
    if not valid.any():
        raise RuntimeError("Insulation score: no valid bins in window (too sparse).")
    ref = raw[valid].mean()
    log2_is = np.full(n, np.nan)
    log2_is[valid] = np.log2(raw[valid] / ref)
    return log2_is


def boundary_objective(log2_is, interior_boundaries):
    """-mean(insulation score at boundary bins); higher is better. A boundary
    landing on a no-data bin is penalized as the worst value seen in the
    profile (so missing data can't be gamed into a free "good" score)."""
    finite = log2_is[np.isfinite(log2_is)]
    worst = float(finite.max()) if finite.size else 0.0
    vals = []
    for b in interior_boundaries:
        vals.append(log2_is[b] if 0 <= b < len(log2_is) and np.isfinite(log2_is[b]) else worst)
    return -float(np.mean(vals))


# ── One (block_size, lambda_e) evaluation on one window ────────────────────

def mappable_bins(sub_M):
    """Local indices whose row/column is not all-zero (unmappable)."""
    return np.flatnonzero(np.abs(np.nan_to_num(sub_M, nan=0.0)).sum(axis=1) > 0)


def solve_boundaries(sub_M, k, block_size, lambda_e, lambda_z, gamma_q, max_iter, min_size):
    keep = mappable_bins(sub_M)
    if keep.size < k * min_size:
        return None, None
    Y = np.nan_to_num(sub_M, nan=0.0)[np.ix_(keep, keep)]
    np.fill_diagonal(Y, 0.0)
    _X, C, _E, _bounds = ssc_admm_block_tv_col21(
        Y, lambda_e=lambda_e, lambda_z=lambda_z, gamma_q=gamma_q,
        block_size=block_size, max_iter=max_iter,
    )
    _labels, boundaries, ncut = dp_contiguous_ncut_partition(C, k, min_size=min_size)
    interior = [int(keep[b]) for b in boundaries[1:-1]]
    return interior, float(ncut)


def solve_and_score(sub_M, k, block_size, lambda_e, lambda_z, gamma_q, max_iter,
                     min_size, is_window_bins):
    interior, ncut = solve_boundaries(
        sub_M, k, block_size, lambda_e, lambda_z, gamma_q, max_iter, min_size,
    )
    if interior is None:
        return {
            "boundaries": [],
            "ncut": float("nan"),
            "objective_neg_mean_is": float("nan"),
            "log2_is": [],
        }
    log2_is = insulation_score_log2(sub_M, is_window_bins)
    obj = boundary_objective(log2_is, interior)
    return {
        "boundaries": interior,
        "ncut": ncut,
        "objective_neg_mean_is": obj,
        "log2_is": [None if not np.isfinite(v) else float(v) for v in log2_is],
    }


def detections_from_windows(windows, k, block_size, lambda_e, lambda_z, gamma_q,
                             max_iter, min_size, log2_is_chrom, resolution, verbose=False):
    detections = []
    for start_bin, sub_M in windows:
        n_drop = sub_M.shape[0] - mappable_bins(sub_M).size
        interior, ncut = solve_boundaries(
            sub_M, k, block_size, lambda_e, lambda_z, gamma_q, max_iter, min_size,
        )
        if interior is None:
            if verbose:
                print(f"  window_start_bin={start_bin} skipped ({n_drop} zero columns)")
            continue
        if verbose:
            print(f"  window_start_bin={start_bin} dropped_zero_cols={n_drop} "
                  f"boundaries={interior} ncut={ncut:.4f}")
        for b in interior:
            g = start_bin + b
            isv = log2_is_chrom[g] if 0 <= g < len(log2_is_chrom) else np.nan
            detections.append({
                "window_start_bin": start_bin,
                "local_bin": b,
                "chrom_bin": g,
                "bp": int(g * resolution),
                "log2_is": None if not np.isfinite(isv) else float(isv),
            })
    return detections


def finite_is(detections):
    return np.array([d["log2_is"] for d in detections if d["log2_is"] is not None])


def matched_random_is(detections, windows, log2_is_chrom, rng):
    """Same count of random loci per window, from mappable bins with finite IS."""
    win = {s: sub for s, sub in windows}
    by_win = {}
    for d in detections:
        by_win.setdefault(d["window_start_bin"], []).append(d)
    out = []
    for start, dets in by_win.items():
        n = sum(1 for d in dets if d["log2_is"] is not None)
        if n == 0:
            continue
        keep = mappable_bins(win[start])
        cands = []
        for loc in keep:
            g = start + int(loc)
            if 0 <= g < len(log2_is_chrom) and np.isfinite(log2_is_chrom[g]):
                cands.append(float(log2_is_chrom[g]))
        if not cands:
            continue
        take = rng.choice(cands, size=n, replace=len(cands) < n)
        out.extend(np.atleast_1d(take).tolist())
    return np.array(out)


# ── Whole-chromosome scan (one config, every 200x200 window) ──────────────

def scan_chromosome(args, M_chrom, out_dir):
    block_size = args.block_sizes[0]
    lambda_e = args.lambda_es[0]
    windows = find_windows(
        M_chrom, args.n_bins, args.max_empty_frac, n_wanted=M_chrom.shape[0],
        start_bin=args.start_bin,
    )
    print(f"  {len(windows)} well-covered {args.n_bins}x{args.n_bins} windows "
          f"(block_size={block_size}, lambda_e={lambda_e}, k={args.k})")

    log2_is_chrom = insulation_score_log2(M_chrom, args.is_window_bins)
    detections = []
    t0 = time.time()
    for start_bin, sub_M in windows:
        r = solve_and_score(
            sub_M, args.k, block_size, lambda_e, args.lambda_z, args.gamma_q,
            args.max_iter, args.min_segment_bins, args.is_window_bins,
        )
        for b in r["boundaries"]:
            g = start_bin + b
            isv = log2_is_chrom[g] if 0 <= g < len(log2_is_chrom) else np.nan
            detections.append({
                "window_start_bin": start_bin,
                "local_bin": b,
                "chrom_bin": g,
                "bp": int(g * args.resolution),
                "log2_is": None if not np.isfinite(isv) else float(isv),
            })
        print(f"  window_start_bin={start_bin} boundaries={r['boundaries']} "
              f"ncut={r['ncut']:.4f}")
    elapsed = time.time() - t0
    det_is = np.array([d["log2_is"] for d in detections if d["log2_is"] is not None])
    bg_is = log2_is_chrom[np.isfinite(log2_is_chrom)]
    print(f"Scan finished in {elapsed:.1f}s. {len(detections)} boundaries, "
          f"{len(det_is)} with finite IS. mean detected log2 IS={det_is.mean():.3f}")

    summary = {
        "hic": args.hic,
        "chrom": args.chrom,
        "resolution": args.resolution,
        "n_bins": args.n_bins,
        "k": args.k,
        "block_size": block_size,
        "lambda_e": lambda_e,
        "n_windows": len(windows),
        "window_starts_bin": [s for s, _ in windows],
        "elapsed_s": elapsed,
        "n_detections": len(detections),
        "mean_detected_log2_is": float(det_is.mean()) if det_is.size else None,
        "median_detected_log2_is": float(np.median(det_is)) if det_is.size else None,
        "mean_background_log2_is": float(bg_is.mean()),
        "detections": detections,
    }
    out_json = out_dir / "chr_scan.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mb = np.arange(len(log2_is_chrom)) * args.resolution / 1e6
    det_mb = np.array([d["bp"] / 1e6 for d in detections if d["log2_is"] is not None])
    fig, (ax_track, ax_hist) = plt.subplots(
        2, 1, figsize=(10, 6.2), gridspec_kw={"height_ratios": [1.5, 1]},
    )
    ax_track.plot(mb, log2_is_chrom, color="0.65", lw=0.4, label="all bins")
    ax_track.scatter(det_mb, det_is, s=18, c="crimson", zorder=3, label="detected boundaries")
    ax_track.set_ylabel("log2 IS")
    ax_track.set_xlabel(f"chr{args.chrom} position (Mb)")
    ax_track.legend(loc="upper right", fontsize=8)
    ax_track.set_title(
        f"chr{args.chrom}  {len(windows)}×{args.n_bins}x{args.n_bins}  "
        f"block_size={block_size}  lambda_e={lambda_e}  k={args.k}"
    )

    ax_hist.hist(bg_is, bins=40, density=True, color="0.75",
                 label=f"all bins (n={len(bg_is)})")
    ax_hist.hist(det_is, bins=20, density=True, color="crimson", alpha=0.65,
                 label=f"detected (n={len(det_is)}, mean={det_is.mean():.3f})")
    ax_hist.set_xlabel("log2 insulation score")
    ax_hist.set_ylabel("density")
    ax_hist.legend(fontsize=8)
    fig.tight_layout()
    out_png = out_dir / "chr_insulation_distribution.png"
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")


def tune_split(args, M_chrom, out_dir):
    """Tune (block_size, lambda_e, lambda_z) on the first 10% of the mappable
    chromosome (skipping the first 3 Mbp), then evaluate on the remaining 90%.

    Objective: minimize mean chromosome-wide log2 IS at predicted boundaries
    (stronger insulation). gamma_q is held at the CLI default.
    """
    unmap_bin = int(3_000_000 / args.resolution)
    mappable = M_chrom.shape[0] - unmap_bin
    split_bin = unmap_bin + int(0.1 * mappable)
    train_windows = find_windows_in_range(
        M_chrom, args.n_bins, args.max_empty_frac, unmap_bin, split_bin,
        stride_bins=max(args.n_bins // 2, 1),
    )
    test_windows = find_windows_in_range(
        M_chrom, args.n_bins, args.max_empty_frac, split_bin, M_chrom.shape[0],
        stride_bins=args.n_bins,
    )
    if not train_windows:
        raise SystemExit(
            f"No well-covered {args.n_bins}-bin windows in train region "
            f"(bins {unmap_bin}–{split_bin}) of chr{args.chrom}."
        )
    if not test_windows:
        raise SystemExit(f"No well-covered windows in the remaining 90% of chr{args.chrom}.")

    block_sizes = args.block_sizes
    lambda_es = args.lambda_es
    lambda_zs = args.lambda_zs
    n_configs = len(block_sizes) * len(lambda_es) * len(lambda_zs)
    print(f"  skip first {unmap_bin * args.resolution / 1e6:.1f} Mb (unmappable)")
    print(f"  train {unmap_bin * args.resolution / 1e6:.2f}–{split_bin * args.resolution / 1e6:.2f} Mb "
          f"({len(train_windows)} windows) starts {[s for s, _ in train_windows]}")
    print(f"  test  {split_bin * args.resolution / 1e6:.2f} Mb–end ({len(test_windows)} windows)")
    print(f"  grid: {len(block_sizes)} block_size x {len(lambda_es)} lambda_e x "
          f"{len(lambda_zs)} lambda_z = {n_configs} configs x {len(train_windows)} "
          f"train windows (gamma_q={args.gamma_q})")

    log2_is_chrom = insulation_score_log2(M_chrom, args.is_window_bins)
    if n_configs == 1:
        best = {
            "block_size": block_sizes[0],
            "lambda_e": lambda_es[0],
            "lambda_z": lambda_zs[0],
            "mean_train_log2_is": None,
            "mean_train_objective": None,
        }
        grid = [best]
        train_elapsed = 0.0
        print("  using given hyperparameters (no grid)")
    else:
        t0 = time.time()
        grid = []
        for block_size in block_sizes:
            for lambda_e in lambda_es:
                for lambda_z in lambda_zs:
                    dets = detections_from_windows(
                        train_windows, args.k, block_size, lambda_e, lambda_z, args.gamma_q,
                        args.max_iter, args.min_segment_bins, log2_is_chrom, args.resolution,
                    )
                    isv = finite_is(dets)
                    mean_is = float(isv.mean()) if isv.size else float("nan")
                    obj = -mean_is
                    grid.append({
                        "block_size": block_size,
                        "lambda_e": lambda_e,
                        "lambda_z": lambda_z,
                        "mean_train_log2_is": mean_is,
                        "mean_train_objective": obj,
                    })
                    print(f"  bs={block_size:3d} le={lambda_e:5.2f} lz={lambda_z:5.2f} "
                          f"-> mean_train_log2_IS={mean_is:.4f}  obj={obj:.4f}")
        train_elapsed = time.time() - t0
        best = max(grid, key=lambda r: r["mean_train_objective"])
        print(f"Tune finished in {train_elapsed:.1f}s. best: block_size={best['block_size']}, "
              f"lambda_e={best['lambda_e']}, lambda_z={best['lambda_z']}, "
              f"mean_train_log2_IS={best['mean_train_log2_is']:.4f}")

    print(f"\nEvaluating best config on {len(test_windows)} held-out 90% windows ...")
    t0 = time.time()
    train_dets = detections_from_windows(
        train_windows, args.k, best["block_size"], best["lambda_e"], best["lambda_z"],
        args.gamma_q, args.max_iter, args.min_segment_bins, log2_is_chrom,
        args.resolution, verbose=True,
    )
    test_dets = detections_from_windows(
        test_windows, args.k, best["block_size"], best["lambda_e"], best["lambda_z"],
        args.gamma_q, args.max_iter, args.min_segment_bins, log2_is_chrom,
        args.resolution, verbose=True,
    )
    test_elapsed = time.time() - t0
    train_is = finite_is(train_dets)
    test_is = finite_is(test_dets)
    rng = np.random.default_rng(0)
    random_is = matched_random_is(test_dets, test_windows, log2_is_chrom, rng)
    print(f"Test finished in {test_elapsed:.1f}s. "
          f"mean train log2 IS={train_is.mean():.3f} (n={len(train_is)}), "
          f"mean test log2 IS={test_is.mean():.3f} (n={len(test_is)}), "
          f"mean random log2 IS={random_is.mean():.3f} (n={len(random_is)})")

    summary = {
        "hic": args.hic,
        "chrom": args.chrom,
        "resolution": args.resolution,
        "n_bins": args.n_bins,
        "k": args.k,
        "gamma_q": args.gamma_q,
        "unmap_bin": unmap_bin,
        "split_bin": split_bin,
        "train_window_starts_bin": [s for s, _ in train_windows],
        "test_window_starts_bin": [s for s, _ in test_windows],
        "grid_search_elapsed_s": train_elapsed,
        "test_elapsed_s": test_elapsed,
        "best": best,
        "grid": grid,
        "mean_train_log2_is": float(train_is.mean()),
        "mean_test_log2_is": float(test_is.mean()),
        "median_test_log2_is": float(np.median(test_is)),
        "mean_random_log2_is": float(random_is.mean()) if random_is.size else None,
        "median_random_log2_is": float(np.median(random_is)) if random_is.size else None,
        "train_detections": train_dets,
        "test_detections": test_dets,
        "random_test_log2_is": [float(v) for v in random_is],
    }
    out_json = out_dir / "chr_tune_split.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    parts = ax.violinplot(
        [random_is, test_is], positions=[1, 2], showmeans=True, showmedians=True,
        widths=0.7,
    )
    parts["bodies"][0].set_facecolor("0.65")
    parts["bodies"][0].set_alpha(0.85)
    parts["bodies"][1].set_facecolor("crimson")
    parts["bodies"][1].set_alpha(0.75)
    for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if key in parts:
            parts[key].set_color("0.2")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([
        f"random loci\n(n={len(random_is)}, mean={random_is.mean():.3f})",
        f"detected\n(n={len(test_is)}, mean={test_is.mean():.3f})",
    ])
    ax.set_ylabel("log2 insulation score")
    ax.set_title(
        f"chr{args.chrom} test 90%  "
        f"block_size={best['block_size']}, λe={best['lambda_e']}, λz={best['lambda_z']}"
    )
    fig.tight_layout()
    out_png = out_dir / "chr_tune_split.png"
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hic", default=(
        "/nfs/turbo/umms-minjilab/lpullela/559/559_spectralTAD/data/sample_mouse.hic"
    ))
    p.add_argument("--chrom", default="19", help="default: chr19, smallest mouse autosome -> fastest load")
    p.add_argument("--resolution", type=int, default=10_000)
    p.add_argument("--n-bins", type=int, default=200, help="window size in bins (200 -> 2Mbp)")
    p.add_argument("--norm", default="KR")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n-train", type=int, default=5)
    p.add_argument("--n-test", type=int, default=5)
    p.add_argument("--start-bin", type=int, default=0)
    p.add_argument("--max-empty-frac", type=float, default=0.2)
    p.add_argument("--is-window-bins", type=int, default=5,
                    help="insulation-score sliding window w, in bins (5 -> 50kb)")
    p.add_argument("--block-sizes", type=int, nargs="+", default=[5, 10, 20, 40])
    p.add_argument("--lambda-es", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0, 5.0])
    p.add_argument("--lambda-z", type=float, default=0.1)
    p.add_argument("--lambda-zs", type=float, nargs="+", default=[0.05, 0.1, 0.5],
                    help="lambda_z grid used by --tune-split")
    p.add_argument("--gamma-q", type=float, default=0.1)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--min-segment-bins", type=int, default=10,
                    help="min contiguous segment length passed to the DP")
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    p.add_argument("--scan", action="store_true",
                   help="run one (block_size, lambda_e) on every well-covered window; "
                        "plot detected-boundary insulation across the chromosome")
    p.add_argument("--tune-split", action="store_true",
                   help="tune block_size/lambda_e/lambda_z on first 10% of the chromosome, "
                        "then plot IS distribution on the remaining 90%")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading chr{args.chrom} from {args.hic} (resolution={args.resolution}, norm={args.norm}) ...")
    M_chrom = load_chrom_matrix(args.hic, args.chrom, args.resolution, args.norm)

    if args.scan:
        scan_chromosome(args, M_chrom, out_dir)
        return
    if args.tune_split:
        tune_split(args, M_chrom, out_dir)
        return

    n_needed = args.n_train + args.n_test
    print(f"Slicing {args.n_bins}x{args.n_bins} windows, need {n_needed} "
          f"({args.n_train} train + {args.n_test} test) ...")
    windows = find_windows(M_chrom, args.n_bins, args.max_empty_frac, n_needed, args.start_bin)
    if len(windows) < n_needed:
        raise SystemExit(
            f"Only found {len(windows)} well-covered windows on chr{args.chrom}, need {n_needed}. "
            f"Try --max-empty-frac higher, a different --chrom, or fewer --n-train/--n-test."
        )
    train_windows = windows[:args.n_train]
    test_windows = windows[args.n_train:args.n_train + args.n_test]
    print("  train window starts (bin):", [s for s, _ in train_windows])
    print("  test  window starts (bin):", [s for s, _ in test_windows])

    print(f"\nGrid search (train only): {len(args.block_sizes)} block_sizes x "
          f"{len(args.lambda_es)} lambda_e = {len(args.block_sizes) * len(args.lambda_es)} "
          f"configs x {len(train_windows)} windows "
          f"(k={args.k}, lambda_z={args.lambda_z}, gamma_q={args.gamma_q}) ...")
    t0 = time.time()
    grid_results = []
    for block_size in args.block_sizes:
        for lambda_e in args.lambda_es:
            per_window = []
            for start_bin, sub_M in train_windows:
                r = solve_and_score(
                    sub_M, args.k, block_size, lambda_e, args.lambda_z, args.gamma_q,
                    args.max_iter, args.min_segment_bins, args.is_window_bins,
                )
                r["window_start_bin"] = start_bin
                per_window.append(r)
            mean_obj = float(np.mean([r["objective_neg_mean_is"] for r in per_window]))
            grid_results.append({
                "block_size": block_size,
                "lambda_e": lambda_e,
                "mean_train_objective": mean_obj,
                "per_window": per_window,
            })
            print(f"  block_size={block_size:3d} lambda_e={lambda_e:5.2f} "
                  f"-> mean_train_obj={mean_obj:.4f}")
    train_elapsed = time.time() - t0
    print(f"Grid search finished in {train_elapsed:.1f}s")

    best = max(grid_results, key=lambda r: r["mean_train_objective"])
    print(f"\nBest on train: block_size={best['block_size']}, lambda_e={best['lambda_e']}, "
          f"mean_train_objective={best['mean_train_objective']:.4f}")

    print(f"\nEvaluating best config on {len(test_windows)} held-out test windows ...")
    t0 = time.time()
    test_results = []
    for start_bin, sub_M in test_windows:
        r = solve_and_score(
            sub_M, args.k, best["block_size"], best["lambda_e"], args.lambda_z, args.gamma_q,
            args.max_iter, args.min_segment_bins, args.is_window_bins,
        )
        r["window_start_bin"] = start_bin
        test_results.append(r)
        print(f"  window_start_bin={start_bin} -> boundaries={r['boundaries']} "
              f"ncut={r['ncut']:.4f} obj={r['objective_neg_mean_is']:.4f}")
    test_elapsed = time.time() - t0
    mean_test_obj = float(np.mean([r["objective_neg_mean_is"] for r in test_results]))
    print(f"Test finished in {test_elapsed:.1f}s. mean_test_objective={mean_test_obj:.4f}")

    summary = {
        "hic": args.hic,
        "chrom": args.chrom,
        "resolution": args.resolution,
        "norm": args.norm,
        "n_bins": args.n_bins,
        "k": args.k,
        "is_window_bins": args.is_window_bins,
        "lambda_z": args.lambda_z,
        "gamma_q": args.gamma_q,
        "max_iter": args.max_iter,
        "min_segment_bins": args.min_segment_bins,
        "train_window_starts_bin": [s for s, _ in train_windows],
        "test_window_starts_bin": [s for s, _ in test_windows],
        "grid_search_elapsed_s": train_elapsed,
        "best_block_size": best["block_size"],
        "best_lambda_e": best["lambda_e"],
        "best_mean_train_objective": best["mean_train_objective"],
        "test_elapsed_s": test_elapsed,
        "mean_test_objective": mean_test_obj,
        "test_results": test_results,
        "grid_results": grid_results,
    }
    out_json = out_dir / "partition_hic_gridsearch.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_plot = len(test_results)
        fig, axes = plt.subplots(n_plot, 1, figsize=(9, 2.2 * n_plot), squeeze=False)
        for ax, (start_bin, sub_M), r in zip(axes[:, 0], test_windows, test_results):
            log2_is = insulation_score_log2(sub_M, args.is_window_bins)
            ax.plot(log2_is, lw=1, color="black")
            for b in r["boundaries"]:
                ax.axvline(b, color="crimson", ls="--", lw=1)
            end_bp = (start_bin + args.n_bins) * args.resolution
            start_bp = start_bin * args.resolution
            ax.set_title(
                f"chr{args.chrom}:{start_bp}-{end_bp}  boundaries={r['boundaries']}  "
                f"obj={r['objective_neg_mean_is']:.3f}",
                fontsize=9,
            )
            ax.set_ylabel("log2 IS")
        axes[-1, 0].set_xlabel(f"bin ({args.resolution}bp each)")
        fig.suptitle(
            f"Test windows: block_size={best['block_size']}, lambda_e={best['lambda_e']}, "
            f"k={args.k}, mean_test_obj={mean_test_obj:.3f}"
        )
        fig.tight_layout()
        out_png = out_dir / "partition_hic_test_windows.png"
        fig.savefig(out_png, dpi=150)
        print(f"Wrote {out_png}")
    except ImportError:
        print("matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
