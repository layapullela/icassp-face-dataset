#!/usr/bin/env python3
"""Summarize the surveillance hold-out CV benchmark: table + figures.

Reads every results CSV under cv/results/surveillance/ (one per
method x k x sigma) plus the matching *_timing.json inference-time
summaries, and produces:

  cv/analysis/summary_table.csv / .md   -- test/train ARI,NMI,ACC per
      (method, k, sigma), mean +/- std across CV folds
  cv/analysis/inference_time.csv        -- mean inference seconds per
      (method, k), pooled across sigma
  cv/analysis/figures/metrics_sigma{s}.png  -- ARI/NMI/ACC bar charts,
      one figure per sigma, grouped by k, error bars = std across folds
  cv/analysis/figures/inference_time.png    -- mean inference time per
      method, faceted by k (log-scale y)

k=3,5,8 (k=12 dropped -- see cv/deprecated/ and the README's validity
notes: a 25-person pool only supports 2 disjoint groups of 12, and once
one is held out only C(13,12)=13 distinct size-12 subsets exist in the
remaining 13 people, not enough combinatorial room for a real tuning
pool). Every tuned method (SSC-Block-TV-Col21, OSC, TKSS) here was
tuned with --tune-pool-groups 20: each fold's hyperparameter search
sees >=20 distinct, possibly-overlapping k-sized subsets of that fold's
held-in people (not just its 1-7 fixed partition groups), while the
reported train/test matrices are still the original fixed partition.
See cv/deprecated/analysis/overlap_ablation.md for the earlier ablation
that established this was worth doing (and that it mainly helps
SSC-Block-TV-Col21, not OSC/TKSS, consistent with it having more
hyperparameters).

The CV unit of resampling is the *fold*, not the individual test
matrix: for each (method, k, sigma, fold) we first average a metric
over that fold's matrices, then report mean +/- std of those per-fold
means across folds. Fold counts from 25 people: k=3 -> 8, k=5 -> 5,
k=8 -> 3.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # cv/
RESULTS = ROOT / "results" / "surveillance"
OUT = Path(__file__).resolve().parent  # cv/analysis/
FIGURES = OUT / "figures"

METHODS = ["SSC-Block-TV-Col21", "OSC", "BDOSC", "TKSS", "Gram-NCut"]
METHOD_LABELS = {
    "SSC-Block-TV-Col21": "SSC-Block-TV-Col21",
    "OSC": "OSC",
    "BDOSC": "BD-OSC",
    "TKSS": "TKSS",
    "Gram-NCut": "DP NCut (trivial)",
}
COLORS = {
    "SSC-Block-TV-Col21": "#2a6f97",
    "OSC": "#8b5e3c",
    "BDOSC": "#b56576",
    "TKSS": "#4a7c59",
    "Gram-NCut": "#8a8a8a",
}
METRICS = [("ari", "ARI"), ("nmi", "NMI"), ("acc", "Accuracy")]
K_VALUES = [3, 5, 8]
SIGMAS = [0.0, 0.25, 0.5]
METHOD_TAGS = {
    "SSC-Block-TV-Col21": "ssc_block_tv_col21",
    "OSC": "osc",
    "BDOSC": "bdosc",
    "TKSS": "tkss",
    "Gram-NCut": "gram_ncut",
}


def sigma_tag(sigma: float) -> str:
    return f"{sigma:g}".replace(".", "p")


def find_csv(method: str, k: int, sigma: float) -> Path:
    tag = f"cv_{METHOD_TAGS[method]}_k{k}_sigma{sigma_tag(sigma)}"
    matches = list(RESULTS.glob(f"*_{tag}.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly 1 CSV for {tag}, found {matches}")
    return matches[0]


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fold_means(rows: list[dict], split: str, metric: str) -> np.ndarray:
    """Per-fold mean of `metric` over that fold's `split` matrices."""
    by_fold: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["split"] == split:
            by_fold[r["fold"]].append(float(r[metric]))
    return np.asarray([np.mean(v) for _, v in sorted(by_fold.items())], dtype=float)


def summarize() -> dict:
    """(method, k, sigma) -> split -> metric -> (mean, std, n_folds)."""
    out = {}
    for method in METHODS:
        for k in K_VALUES:
            for sigma in SIGMAS:
                rows = load_rows(find_csv(method, k, sigma))
                entry = {}
                for split in ("train", "test"):
                    entry[split] = {}
                    for metric, _ in METRICS:
                        vals = fold_means(rows, split, metric)
                        entry[split][metric] = (
                            float(vals.mean()), float(vals.std()), int(vals.size),
                        )
                out[(method, k, sigma)] = entry
    return out


def summarize_timing() -> dict:
    """(method, k) -> (mean_s, std_s, n_calls) pooled across sigma."""
    out = {}
    for method in METHODS:
        for k in K_VALUES:
            total_s, n_calls, sq = 0.0, 0, 0.0
            for sigma in SIGMAS:
                csv_path = find_csv(method, k, sigma)
                timing_path = csv_path.with_name(f"{csv_path.stem}_timing.json")
                data = json.loads(timing_path.read_text())[method]
                n = data["n_calls"]
                total_s += data["total_s"]
                n_calls += n
                # sum of squares from mean/std (population std) to pool variance
                sq += n * (data["std_s"] ** 2 + data["mean_s"] ** 2)
            mean_s = total_s / n_calls
            var_s = sq / n_calls - mean_s ** 2
            out[(method, k)] = (mean_s, float(np.sqrt(max(var_s, 0.0))), n_calls)
    return out


def write_summary_table(summary: dict) -> tuple[Path, Path]:
    csv_path = OUT / "summary_table.csv"
    md_path = OUT / "summary_table.md"
    header = [
        "method", "k", "n_folds", "sigma", "split",
        "ari_mean", "ari_std", "nmi_mean", "nmi_std", "acc_mean", "acc_std",
    ]
    rows_out = []
    for method in METHODS:
        for k in K_VALUES:
            for sigma in SIGMAS:
                entry = summary[(method, k, sigma)]
                for split in ("train", "test"):
                    a_m, a_s, n = entry[split]["ari"]
                    m_m, m_s, _ = entry[split]["nmi"]
                    c_m, c_s, _ = entry[split]["acc"]
                    rows_out.append([
                        METHOD_LABELS[method], k, n, sigma, split,
                        f"{a_m:.4f}", f"{a_s:.4f}", f"{m_m:.4f}", f"{m_s:.4f}",
                        f"{c_m:.4f}", f"{c_s:.4f}",
                    ])
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows_out)

    with md_path.open("w") as f:
        f.write("# Surveillance hold-out CV summary\n\n")
        f.write(
            "Mean +/- std across CV folds (fold = one held-out group of k "
            "people). k=3: 8 folds, k=5: 5 folds, k=8: 3 folds. Tuned "
            "methods (SSC-Block-TV-Col21, OSC, TKSS) use "
            "--tune-pool-groups 20 (see module docstring).\n\n"
        )
        for split in ("test", "train"):
            f.write(f"## {split.capitalize()}\n\n")
            f.write("| Method | k | folds | sigma | ARI | NMI | ACC |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for method in METHODS:
                for k in K_VALUES:
                    for sigma in SIGMAS:
                        entry = summary[(method, k, sigma)][split]
                        a_m, a_s, n = entry["ari"]
                        m_m, m_s, _ = entry["nmi"]
                        c_m, c_s, _ = entry["acc"]
                        f.write(
                            f"| {METHOD_LABELS[method]} | {k} | {n} | {sigma:g} | "
                            f"{a_m:.3f} +/- {a_s:.3f} | {m_m:.3f} +/- {m_s:.3f} | "
                            f"{c_m:.3f} +/- {c_s:.3f} |\n"
                        )
            f.write("\n")
    return csv_path, md_path


def write_timing_table(timing: dict) -> Path:
    path = OUT / "inference_time.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "k", "mean_seconds", "std_seconds", "n_calls"])
        for method in METHODS:
            for k in K_VALUES:
                mean_s, std_s, n = timing[(method, k)]
                w.writerow([
                    METHOD_LABELS[method], k, f"{mean_s:.4f}", f"{std_s:.4f}", n,
                ])
    return path


def draw_bars(ax, means, stds, colors, x, width):
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    bars = ax.bar(
        x, means, width, yerr=stds, color=colors,
        edgecolor="white", linewidth=0.8, capsize=3,
        error_kw=dict(ecolor="#333333", lw=1),
    )
    finite = np.isfinite(means) & np.isfinite(stds)
    highs = means + stds
    y_top = float(np.nanmax(highs[finite])) + 0.05 if np.any(finite) else 1.0
    y_bot = max(0.0, float(np.nanmin((means - stds)[finite])) - 0.05) if np.any(finite) else 0.0
    ax.set_ylim(y_bot, y_top)
    if y_bot <= 1.0 <= y_top:
        ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls="--", zorder=0)
    for bar, mean, high in zip(bars, means, highs):
        if np.isfinite(mean):
            ax.text(
                bar.get_x() + bar.get_width() / 2, high + 0.008 * (y_top - y_bot),
                f"{mean:.2f}", ha="center", va="bottom", fontsize=7, color="#222222",
            )
    return bars


def plot_metrics(summary: dict) -> list[Path]:
    paths = []
    FIGURES.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(METHODS))
    width = 0.62
    colors = [COLORS[m] for m in METHODS]

    for sigma in SIGMAS:
        fig, axes = plt.subplots(
            len(METRICS), len(K_VALUES), figsize=(11.5, 8.5), constrained_layout=True,
        )
        for col, k in enumerate(K_VALUES):
            n_folds = summary[(METHODS[0], k, sigma)]["test"]["ari"][2]
            for row, (metric, metric_label) in enumerate(METRICS):
                ax = axes[row, col]
                means = [summary[(m, k, sigma)]["test"][metric][0] for m in METHODS]
                stds = [summary[(m, k, sigma)]["test"][metric][1] for m in METHODS]
                draw_bars(ax, means, stds, colors, x, width)
                ax.set_xticks(x)
                ax.set_xticklabels(
                    [METHOD_LABELS[m] for m in METHODS], rotation=25, ha="right", fontsize=8,
                )
                if col == 0:
                    ax.set_ylabel(metric_label, fontsize=11)
                if row == 0:
                    ax.set_title(f"k = {k}  ({n_folds} folds)", fontsize=11)
                ax.grid(axis="y", linestyle=":", alpha=0.45)
                ax.set_axisbelow(True)
        fig.suptitle(
            f"Surveillance hold-out CV — test-set, known k, σ={sigma:g}  "
            "(error bars: std across CV folds)",
            fontsize=13,
        )
        path = FIGURES / f"metrics_sigma{sigma_tag(sigma)}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_timing(timing: dict) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(11, 3.8), constrained_layout=True)
    x = np.arange(len(METHODS))
    width = 0.62
    colors = [COLORS[m] for m in METHODS]

    for ax, k in zip(axes, K_VALUES):
        means = [timing[(m, k)][0] for m in METHODS]
        stds = [timing[(m, k)][1] for m in METHODS]
        bars = ax.bar(
            x, means, width, yerr=stds, color=colors,
            edgecolor="white", linewidth=0.8, capsize=3,
            error_kw=dict(ecolor="#333333", lw=1),
        )
        ax.set_yscale("log")
        top = max(m + s for m, s in zip(means, stds))
        ax.set_ylim(top=top * 6)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [METHOD_LABELS[m] for m in METHODS], rotation=25, ha="right", fontsize=8,
        )
        ax.set_title(f"k = {k}", fontsize=11)
        ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2, (mean + std) * 1.25,
                f"{mean:.2g}s", ha="center", va="bottom", fontsize=7, color="#222222",
            )
    axes[0].set_ylabel("mean inference seconds / matrix (log scale)", fontsize=10)
    fig.suptitle(
        "Surveillance hold-out CV — inference time per method "
        "(pooled across σ, excludes tuning time)",
        fontsize=13,
    )
    path = FIGURES / "inference_time.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    summary = summarize()
    timing = summarize_timing()

    csv_path, md_path = write_summary_table(summary)
    timing_csv = write_timing_table(timing)
    metric_figs = plot_metrics(summary)
    timing_fig = plot_timing(timing)

    print("Test-set ARI (mean +/- std across folds):")
    for k in K_VALUES:
        print(f"  k={k}")
        for sigma in SIGMAS:
            print(f"    sigma={sigma:g}")
            for method in METHODS:
                a_m, a_s, n = summary[(method, k, sigma)]["test"]["ari"]
                t_m, t_s, _ = timing[(method, k)]
                print(
                    f"      {METHOD_LABELS[method]:20s} ARI={a_m:.3f}+/-{a_s:.3f}  "
                    f"(n_folds={n})  inference={t_m:.3f}s/matrix"
                )

    print("\nWrote:")
    for p in [csv_path, md_path, timing_csv, *metric_figs, timing_fig]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
