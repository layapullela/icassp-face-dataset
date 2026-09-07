#!/usr/bin/env python3
"""Plot ARI / NMI / ACC for SSC-Block-TV-Col21 vs OSC, TKSS, BDOSC.

Error bars are 95% percentile bootstrap CIs of the test-set mean
(resample matrices with replacement). Because ARI/NMI/ACC are bounded
in [0, 1], bootstrap means cannot exceed 1, so the upper interval is
calibrated by the sampling distribution rather than clipped.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

METHODS = ["SSC-Block-TV-Col21", "OSC", "TKSS", "BDOSC"]
METHOD_LABELS = {
    "SSC-Block-TV-Col21": "SSC-Block-TV-Col21",
    "OSC": "OSC",
    "TKSS": "TKSS",
    "BDOSC": "BD-OSC",
}
METRICS = [("ari", "ARI"), ("nmi", "NMI"), ("acc", "Accuracy")]
K_VALUES = [5, 8, 12]
COLORS = {
    "SSC-Block-TV-Col21": "#2a6f97",
    "OSC": "#8b5e3c",
    "TKSS": "#4a7c59",
    "BDOSC": "#b56576",
}
N_BOOT = 10_000
CI_LEVEL = 0.95
BOOT_SEED = 0


def result_paths(dataset: str, k: int) -> tuple[Path, Path]:
    if dataset == "surveillance":
        stem = f"split_k{k}_clean_min30_trials10_knownk_scaled_k{k}"
        return (
            RESULTS / dataset / f"{stem}_variants.csv",
            RESULTS / dataset / f"{stem}_baselines.csv",
        )
    return (
        RESULTS / dataset / f"ballet_cluster_knownk_scaled_results_k{k}_variants.csv",
        RESULTS / dataset / f"ballet_cluster_knownk_scaled_results_k{k}_baselines.csv",
    )


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def test_scores(rows: list[dict], method: str, metric: str) -> np.ndarray:
    vals = [
        float(r[metric])
        for r in rows
        if r["method"] == method and r["split"] == "test"
    ]
    return np.asarray(vals, dtype=float)


def bootstrap_mean_ci(
    vals: np.ndarray,
    n_boot: int = N_BOOT,
    ci_level: float = CI_LEVEL,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, int]:
    """Observed mean and percentile bootstrap CI of the mean.

    Returns (mean, ci_low, ci_high, n).
    """
    vals = np.asarray(vals, dtype=float)
    n = int(vals.size)
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    mean = float(vals.mean())
    if n == 1:
        return (mean, mean, mean, 1)
    if rng is None:
        rng = np.random.default_rng(BOOT_SEED)
    draws = rng.choice(vals, size=(n_boot, n), replace=True).mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    ci_low = float(np.quantile(draws, alpha))
    ci_high = float(np.quantile(draws, 1.0 - alpha))
    return (mean, ci_low, ci_high, n)


def bootstrap_yerr(means, lows, highs) -> np.ndarray:
    means = np.asarray(means, dtype=float)
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lower = np.maximum(means - lows, 0.0)
    upper = np.maximum(highs - means, 0.0)
    return np.vstack([lower, upper])


def axis_limits(means, lows, highs) -> tuple[float, float]:
    """Y-min is 10% below the lowest-scoring bar's CI lower bound."""
    means = np.asarray(means, dtype=float)
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    finite = np.isfinite(means) & np.isfinite(lows) & np.isfinite(highs)
    if not np.any(finite):
        return 0.5, 1.03
    idx = int(np.nanargmin(np.where(finite, means, np.inf)))
    y_bot = 0.9 * float(lows[idx])
    y_top = float(np.nanmax(highs[finite])) + 0.04
    return y_bot, y_top


def draw_bars(ax, means, lows, highs, colors, x, width):
    yerr = bootstrap_yerr(means, lows, highs)
    y_bot, y_top = axis_limits(means, lows, highs)
    bars = ax.bar(
        x,
        means,
        width,
        yerr=yerr,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        capsize=3,
        error_kw=dict(ecolor="#333333", lw=1),
    )
    ax.set_ylim(y_bot, y_top)
    if y_bot <= 1.0 <= y_top:
        ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls="--", zorder=0)
    for bar, mean, high in zip(bars, means, highs):
        if np.isfinite(mean):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                high + 0.008 * (y_top - y_bot),
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#222222",
            )
    return bars


def summarize(
    dataset: str,
) -> dict[int, dict[str, dict[str, tuple[float, float, float, int]]]]:
    """k -> method -> metric -> (mean, ci_low, ci_high, n)."""
    out: dict[int, dict[str, dict[str, tuple[float, float, float, int]]]] = {}
    rng = np.random.default_rng(BOOT_SEED)
    for k in K_VALUES:
        rows = load_rows(list(result_paths(dataset, k)))
        out[k] = {}
        for method in METHODS:
            out[k][method] = {}
            for metric, _ in METRICS:
                vals = test_scores(rows, method, metric)
                out[k][method][metric] = bootstrap_mean_ci(vals, rng=rng)
    return out


def plot_dataset(dataset: str, summary: dict) -> Path:
    fig, axes = plt.subplots(
        len(METRICS),
        len(K_VALUES),
        figsize=(11.5, 8.5),
        constrained_layout=True,
    )
    x = np.arange(len(METHODS))
    width = 0.62

    for col, k in enumerate(K_VALUES):
        for row, (metric, metric_label) in enumerate(METRICS):
            ax = axes[row, col]
            means = [summary[k][m][metric][0] for m in METHODS]
            lows = [summary[k][m][metric][1] for m in METHODS]
            highs = [summary[k][m][metric][2] for m in METHODS]
            colors = [COLORS[m] for m in METHODS]
            draw_bars(ax, means, lows, highs, colors, x, width)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [METHOD_LABELS[m] for m in METHODS],
                rotation=25,
                ha="right",
                fontsize=8,
            )
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=11)
            if row == 0:
                n = summary[k][METHODS[0]][metric][3]
                ax.set_title(f"k = {k}  (n={n} test mats)", fontsize=11)
            ax.grid(axis="y", linestyle=":", alpha=0.45)
            ax.set_axisbelow(True)

    fig.suptitle(
        f"{dataset.capitalize()} — test-set comparison "
        f"(known k, clean σ=0; 95% bootstrap CI)",
        fontsize=13,
    )
    out_dir = LOGS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_col21_vs_baselines_metrics.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_per_k(dataset: str, summary: dict) -> list[Path]:
    """One compact figure per k (3 metrics side by side)."""
    paths = []
    out_dir = LOGS / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(METHODS))
    width = 0.62

    for k in K_VALUES:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
        for ax, (metric, metric_label) in zip(axes, METRICS):
            means = [summary[k][m][metric][0] for m in METHODS]
            lows = [summary[k][m][metric][1] for m in METHODS]
            highs = [summary[k][m][metric][2] for m in METHODS]
            colors = [COLORS[m] for m in METHODS]
            draw_bars(ax, means, lows, highs, colors, x, width)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [METHOD_LABELS[m] for m in METHODS], rotation=25, ha="right", fontsize=8,
            )
            ax.set_title(metric_label)
            ax.grid(axis="y", linestyle=":", alpha=0.45)
            ax.set_axisbelow(True)
        n = summary[k][METHODS[0]]["ari"][3]
        fig.suptitle(
            f"{dataset.capitalize()}  k={k}  (test, n={n}; 95% bootstrap CI)",
            fontsize=12,
        )
        path = out_dir / f"{dataset}_k{k}_col21_vs_baselines.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_summary_csv(dataset: str, summary: dict) -> Path:
    path = LOGS / dataset / f"{dataset}_col21_vs_baselines_summary.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "k", "method", "metric", "mean",
            "ci_low", "ci_high", "n_test", "n_boot", "ci_level",
        ])
        for k in K_VALUES:
            for method in METHODS:
                for metric, _ in METRICS:
                    mean, lo, hi, n = summary[k][method][metric]
                    w.writerow([
                        dataset, k, METHOD_LABELS[method], metric,
                        f"{mean:.6f}", f"{lo:.6f}", f"{hi:.6f}",
                        n, N_BOOT, CI_LEVEL,
                    ])
    return path


def main() -> None:
    written = []
    for dataset in ("surveillance", "ballet"):
        summary = summarize(dataset)
        written.append(plot_dataset(dataset, summary))
        written.extend(plot_per_k(dataset, summary))
        written.append(write_summary_csv(dataset, summary))
        print(f"\n{dataset}  (95% percentile bootstrap CI, {N_BOOT} resamples):")
        for k in K_VALUES:
            print(f"  k={k}")
            for method in METHODS:
                a, al, ah, n = summary[k][method]["ari"]
                m, ml, mh, _ = summary[k][method]["nmi"]
                c, cl, ch, _ = summary[k][method]["acc"]
                print(
                    f"    {METHOD_LABELS[method]:22s}  "
                    f"ARI={a:.3f} [{al:.3f},{ah:.3f}]  "
                    f"NMI={m:.3f} [{ml:.3f},{mh:.3f}]  "
                    f"ACC={c:.3f} [{cl:.3f},{ch:.3f}]  (n={n})"
                )
    print("\nWrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
