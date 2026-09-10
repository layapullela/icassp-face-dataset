#!/usr/bin/env python3
"""The two headline figures: test ARI and test accuracy.

  figures/fold0_surv_crossfold_ballet.{png,pdf}
  figures/fold0_surv_crossfold_ballet_accuracy.{png,pdf}

Each dataset is aggregated at the level its data actually supports, which is
why the two rows are labelled separately rather than sharing one caption:

  * SURVEILLANCE (balanced, 50-100 frames) -- FOLD 0 ONLY. The full-CV
    surveillance runs were stopped part-way to save compute, so cells have
    differing fold coverage. Fold 0 is present in every cell, so restricting
    to it makes all five methods comparable on identical data. Whiskers
    resample that fold's 24 test matrices (one per sequence).
  * BALLET (balanced, 50-100 frames) -- ALL FOLDS (7 at k=3, 4 at k=5). Those
    jobs ran to completion, so cross-fold means are used; throwing folds away
    would be strictly worse evidence. Whiskers resample the per-fold means.

WHISKERS ARE A BOOTSTRAP INTERVAL, NOT +/-1 STD. ARI and accuracy are both
bounded above by 1.0, and a symmetric std whisker computed near that ceiling
can extend past it -- which earlier drafts of this figure did. A bootstrap
resample only ever contains values already present in the sample, so the
resampled mean and every percentile of its distribution stay within
[min(values), max(values)], which can never exceed 1.0. The 16th-84th
percentile split is chosen because it brackets ~68% of a Gaussian, i.e. it
carries roughly the same visual weight as the old std whisker without ever
leaving the data's own range. It is a confidence interval on the MEAN, so it
is visibly tighter than a spread-of-observations whisker -- that is the
intended consequence, not a bug.

Design notes (dataviz):
  * Grouped bars, faceted by (dataset, k), sigma on the within-panel x-axis so
    the noise trend reads inside a facet rather than across facets.
  * Bars anchored at y=0 over the metric's full [0, 1] range -- no truncated
    axis exaggerating small differences.
  * Palette: validated categorical slots (blue/yellow/magenta/green/violet),
    worst-pair CVD deltaE 13.0 / normal-vision 16.3 on the all-pairs list,
    which is the right list because all five bars sit side by side. Identity
    is never colour-alone: there is always a legend, and the trivial control
    additionally carries a hatch.
  * The winning bar in each sigma group is directly labelled, which also
    discharges the contrast obligation for the yellow and magenta slots.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sscbench.config import K_VALUES, METHODS, SIGMAS   # noqa: E402

HERE = Path(__file__).resolve().parent

SHORT = {
    "SSC-Block-TV-Col21": "SSC-Block-TV-Col21 (ours)",
    "OSC": "OSC",
    "BD-OSC": "BD-OSC",
    "TKSS": "TKSS",
    "DP NCut (trivial)": "DP NCut (trivial control)",
}
COLORS = {
    "SSC-Block-TV-Col21": "#2a78d6",   # blue
    "OSC":                "#eda100",   # yellow
    "BD-OSC":             "#e87ba4",   # magenta
    "TKSS":               "#008300",   # green
    "DP NCut (trivial)":  "#4a3aa7",   # violet
}
HATCH = {"DP NCut (trivial)": "///"}   # secondary encoding for the control

INK, MUTED, SURFACE = "#1a1a19", "#6b6b66", "#fcfcfb"

METRIC_LABEL = {"ari": "Test ARI", "acc": "Test accuracy"}


def bootstrap_ci(values, n_boot=2000, lo_pct=16, hi_pct=84, seed=0):
    """Percentile bootstrap interval for the mean. Bounded by the data's range.

    With <2 values there is nothing to resample, so lo = hi = the value: a
    single observation has no variance to estimate, and drawing a whisker on
    it would be inventing a number.
    """
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [lo_pct, hi_pct])
    return mean, float(lo), float(hi)


def load_metrics(path):
    """Per-matrix rows -> {(dataset, method, k, sigma, fold): [values...]}."""
    by_cell = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["method"], int(r["k"]), float(r["sigma"]))
            by_cell[key][int(r["fold"])].append(r)
    return by_cell


def cell_value(by_cell, dataset, method, k, sigma, metric):
    """(mean, lo, hi, n) for one panel bar, aggregated per dataset convention.

    Surveillance resamples fold 0's individual test matrices; ballet resamples
    the per-fold means. Returns None if the cell has no data.
    """
    folds = by_cell.get((dataset, method, k, sigma))
    if not folds:
        return None
    if dataset == "surveillance":
        rows = folds.get(0)
        if not rows:
            return None
        vals = [float(r[metric]) for r in rows]
        return (*bootstrap_ci(vals), len(rows))
    fold_means = [float(np.mean([float(r[metric]) for r in rows]))
                  for _, rows in sorted(folds.items())]
    if not fold_means:
        return None
    return (*bootstrap_ci(fold_means), len(fold_means))


def load_flags(path):
    """(dataset, method, k, sigma) -> short caveat to stamp on that bar.

    Driven from hyperparameters.csv's ``notes`` column rather than hardcoded,
    so the annotation disappears on its own once a cell is rerun with the
    documented grid -- a hardcoded caption would quietly go stale instead.
    """
    flags = {}
    if not path.exists():
        return flags
    for r in csv.DictReader(open(path)):
        key = (r["dataset"], r["method"], int(r["k"]), float(r["sigma"]))
        if "NARROW d" in (r["notes"] or ""):
            flags[key] = "narrow d"
    return flags


def rounded_bar(ax, x, w, h, color, hatch=None, r_pts=4.0):
    """Bar with rounded top corners, square base, anchored at y=0."""
    if not np.isfinite(h) or h <= 0:
        return
    px, inv = ax.transData.transform, ax.transData.inverted().transform
    p0 = px((0, 0))
    rx = abs(inv((p0[0] + r_pts * ax.figure.dpi / 72, p0[1]))[0])
    ry = abs(inv((p0[0], p0[1] + r_pts * ax.figure.dpi / 72))[1])
    rx, ry = min(rx, w / 2), min(ry, h)
    x0, x1 = x - w / 2, x + w / 2
    verts = [(x0, 0), (x0, h - ry), (x0, h), (x0 + rx, h),
             (x1 - rx, h), (x1, h), (x1, h - ry), (x1, 0), (x0, 0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3,
             MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
             MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color,
                           edgecolor=SURFACE, linewidth=1.6, hatch=hatch,
                           zorder=3))


def draw_panel(ax, by_cell, dataset, k, metric, title, subtitle, flags=None):
    n = len(METHODS)
    group_w, gap = 0.82, 0.035
    bw = group_w / n - gap
    for gi, sigma in enumerate(SIGMAS):
        vals = [cell_value(by_cell, dataset, m, k, sigma, metric)
                for m in METHODS]
        best = max((v[0] for v in vals if v), default=None)
        for mi, (m, v) in enumerate(zip(METHODS, vals)):
            if v is None:
                continue
            mean, lo, hi, _ = v
            xc = gi + (mi - (n - 1) / 2) * (group_w / n)
            rounded_bar(ax, xc, bw, mean, COLORS[m], HATCH.get(m))
            ax.errorbar(xc, mean,
                        yerr=[[max(mean - lo, 0.0)], [max(hi - mean, 0.0)]],
                        ecolor=MUTED, elinewidth=1.1, capsize=2.5,
                        capthick=1.1, fmt="none", zorder=4)
            flag = (flags or {}).get((dataset, m, k, sigma))
            if flag:
                ax.text(xc, min(hi + 0.022, 0.955), flag, ha="center",
                        va="bottom", fontsize=6.2, color=MUTED,
                        style="italic", zorder=5)
            if best is not None and abs(mean - best) < 1e-12:
                ax.text(xc, min(hi + 0.050, 0.975), f"{mean:.2f}",
                        ha="center", va="bottom", fontsize=7.5, color=INK,
                        fontweight="bold", zorder=5)
    ax.set_xticks(range(len(SIGMAS)))
    ax.set_xticklabels([f"σ={s:g}" for s in SIGMAS], fontsize=9, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="y", labelsize=8, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, color="#cfcfc9", alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#cfcfc9")
    ax.set_title(title, fontsize=11, color=INK, pad=18, fontweight="bold")
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8.5, color=MUTED)


def make_figure(by_cell, metric, out_path, flags=None):
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.6), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)

    for col, k in enumerate(K_VALUES["surveillance"]):
        draw_panel(axes[0, col], by_cell, "surveillance", k, metric,
                   f"Surveillance   k = {k}",
                   "balanced (50–100)  ·  fold 0 only", flags)

    for col, k in enumerate(K_VALUES["ballet"]):
        v = cell_value(by_cell, "ballet", "OSC", k, 0.0, metric)
        n_folds = v[3] if v else "?"
        draw_panel(axes[1, col], by_cell, "ballet", k, metric,
                   f"Ballet   k = {k}",
                   f"balanced (50–100)  ·  {n_folds} folds", flags)

    axes[0, 0].set_ylabel(METRIC_LABEL[metric], fontsize=10, color=INK)
    axes[1, 0].set_ylabel(METRIC_LABEL[metric], fontsize=10, color=INK)

    lg = axes[1, 2]
    lg.axis("off")
    handles = [Patch(facecolor=COLORS[m], edgecolor=SURFACE, linewidth=1.4,
                     hatch=HATCH.get(m), label=SHORT[m]) for m in METHODS]
    lg.legend(handles=handles, loc="upper left", frameon=False, fontsize=9.5,
              labelspacing=0.9, handlelength=1.6, handleheight=1.1,
              bbox_to_anchor=(0.0, 0.99))
    lg.text(0.0, 0.36,
            "Whiskers: 16th–84th percentile BOOTSTRAP interval on the\n"
            "mean (2000 resamples), not ±1 std — bounded by the data's\n"
            "own range, so it cannot exceed 1.0 the way a symmetric std\n"
            "whisker could near the ceiling.\n"
            "  · surveillance — resampled over that fold's 24 test\n"
            "    matrices (one per sequence)\n"
            "  · ballet — resampled over the per-fold means\n\n"
            "Surveillance is fold 0 only: its full-CV runs were stopped\n"
            "to save compute. Ballet ran to completion, so all folds\n"
            "are used.\n\n"
            + ("Bars marked \"narrow d\" use TKSS's older, narrower d grid;\n"
               "their widened-grid rerun was cancelled.\n\n" if flags else "")
            + "Value shown on the best method in each σ group.",
            transform=lg.transAxes, fontsize=8.2, color=MUTED, va="top")

    fig.suptitle(
        f"Sequential subspace clustering — known k, test set   "
        f"(surveillance: fold 0 · ballet: cross-fold, balanced)",
        fontsize=13.5, color=INK, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"wrote {out_path.with_suffix('.pdf')}")


def validate(by_cell, metric):
    """Every interval must satisfy 0 <= lo <= mean <= hi <= 1."""
    bad = []
    for dataset, ks in (("surveillance", K_VALUES["surveillance"]),
                        ("ballet", K_VALUES["ballet"])):
        for m in METHODS:
            for k in ks:
                for s in SIGMAS:
                    v = cell_value(by_cell, dataset, m, k, s, metric)
                    if v is None:
                        continue
                    mean, lo, hi, _ = v
                    if not (0 <= lo <= mean + 1e-12 and mean <= hi + 1e-12
                            and hi <= 1.0 + 1e-12):
                        bad.append((dataset, m, k, s, mean, lo, hi))
    return bad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path,
                    default=HERE / "results" / "reference_metrics.csv",
                    help="per-matrix metrics CSV (default: %(default)s). Pass "
                         "a run_inference.py output to plot your own run.")
    ap.add_argument("--hyperparameters", type=Path,
                    default=HERE / "hyperparameters.csv",
                    help="source of the per-cell provenance caveats stamped "
                         "on affected bars (default: %(default)s)")
    ap.add_argument("--metric", choices=("ari", "acc", "both"), default="both")
    ap.add_argument("--outdir", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    if not args.results.exists():
        raise SystemExit(f"{args.results} not found -- run "
                         "build_hyperparameters_csv.py or run_inference.py")
    by_cell = load_metrics(args.results)

    flags = load_flags(args.hyperparameters)

    metrics = ("ari", "acc") if args.metric == "both" else (args.metric,)
    for metric in metrics:
        bad = validate(by_cell, metric)
        if bad:
            print(f"WARNING: {len(bad)} {metric} interval(s) out of bounds:")
            for b in bad:
                print("   ", b)
        else:
            print(f"{metric}: all intervals satisfy 0 <= lo <= mean <= hi <= 1")
        suffix = "" if metric == "ari" else "_accuracy"
        make_figure(by_cell, metric,
                    args.outdir / f"fold0_surv_crossfold_ballet{suffix}.png",
                    flags)


if __name__ == "__main__":
    main()
