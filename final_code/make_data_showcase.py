#!/usr/bin/env python3
"""Two data-showcase figures: what a clustering matrix actually contains.

  figures/ballet_data_matrix.{png,pdf}
  figures/surveillance_faces.{png,pdf}

Ballet is one dancer's k pose-segments concatenated into Y (the object the
methods partition). Surveillance is k people's face crops from one sequence,
the same construction, shown as faces because that is what the columns are.
Both use the benchmark loaders, seed, and first fold-0 example at σ=0, so the
pixels on the page are a real matrix from the experiment — not a schematic.
"""

from __future__ import annotations

import argparse
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sscbench.config import (  # noqa: E402
    BALLET_DOWN_HW,
    BALLET_MATRICES_PER_DANCER,
    N_FRAMES_HI,
    N_FRAMES_LO,
    SEED,
    SURV_DOWN_HW,
)
from sscbench.data_ballet import (  # noqa: E402
    eligible_dancers,
    load_dancer_segments,
    load_image,
    name_to_dir_map,
    sample_combos,
)
from sscbench.data_surveillance import (  # noqa: E402
    chunk_folds,
    downsample_image,
    read_pgm,
    sequence_dirs,
    subset_people,
)

HERE = Path(__file__).resolve().parent
INK, MUTED, SURFACE = "#1a1a19", "#6b6b66", "#fcfcfb"
BLOCK_COLORS = ("#2a78d6", "#eda100", "#008300", "#e87ba4", "#4a3aa7",
                "#c44e52", "#55a868", "#8172b3")
BOUNDARY = "#f4e27a"


def even_indices(n, n_show):
    n_show = min(int(n_show), int(n))
    if n_show <= 0:
        return []
    if n_show == 1:
        return [0]
    return np.linspace(0, n - 1, n_show).round().astype(int).tolist()


def block_lengths(labels):
    labels = np.asarray(labels, dtype=int)
    lengths, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            lengths.append(i - start)
            start = i
    return lengths


def load_gray(path, size=None):
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    if size is not None:
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float64) / 255.0


def load_pgm_gray(path):
    return read_pgm(path).astype(np.float64) / 255.0


def first_ballet_example(k=3):
    """Fold-0 dancer, first of that dancer's k-combo matrices, with paths."""
    dancer_segments = load_dancer_segments()
    name_to_dir = name_to_dir_map()
    dancer_id = eligible_dancers(dancer_segments, k)[0]
    segments = dancer_segments[dancer_id]
    rng = np.random.default_rng(SEED + dancer_id)
    n_all = comb(len(segments), k)
    combos = sample_combos(range(len(segments)), k,
                           min(BALLET_MATRICES_PER_DANCER, n_all), rng)
    combo = combos[0]
    cols, labs, descs, path_blocks = [], [], [], []
    for pos, seg_idx in enumerate(combo):
        seq_name, lo, hi = segments[seg_idx]
        frames = sorted(name_to_dir[seq_name].glob("*.jpg"))[lo:hi]
        n_lo, n_hi = N_FRAMES_LO, max(N_FRAMES_LO, N_FRAMES_HI)
        n = min(int(rng.integers(n_lo, n_hi + 1)), len(frames))
        start = int(rng.integers(0, len(frames) - n + 1))
        paths = frames[start:start + n]
        cols.append(np.stack([load_image(p) for p in paths], axis=1))
        labs.append(np.full(len(paths), pos, dtype=int))
        descs.append(f"{seq_name}  frames {lo}–{hi - 1}")
        path_blocks.append(paths)
    return (dancer_id,
            np.concatenate(cols, axis=1),
            np.concatenate(labs),
            descs,
            path_blocks)


def first_surv_example(k=5):
    """Sequence 0, fold-0 test group of k people, σ=0, with original paths.

    Consumes the shared ``default_rng(SEED)`` stream across every person in
    the first sequence (the prefix of ``load_all_sequences``), then subsets
    to fold 0's identities — so this is the first test matrix of fold 0.
    """
    rng = np.random.default_rng(SEED)
    seq = sequence_dirs()[0]
    lo, hi = N_FRAMES_LO, max(N_FRAMES_LO, N_FRAMES_HI)
    person_dirs = sorted((p for p in seq.iterdir() if p.is_dir()),
                         key=lambda p: p.name)
    images, labels, all_paths = [], [], []
    for person_idx, person_dir in enumerate(person_dirs):
        frames = sorted(person_dir.glob("*.pgm"), key=lambda p: p.name)
        n = min(int(rng.integers(lo, hi + 1)), len(frames))
        start = int(rng.integers(0, len(frames) - n + 1))
        for frame_path in frames[start:start + n]:
            images.append(downsample_image(read_pgm(frame_path)).reshape(-1))
            labels.append(person_idx)
            all_paths.append(frame_path)
    Y = np.stack(images, axis=1) / 255.0
    labels = np.asarray(labels, dtype=int)
    names = sorted(p.name for p in person_dirs)
    keep = chunk_folds(names, k)[0]
    Ys, labs, nms = subset_people(Y, labels, person_dirs, keep)
    path_blocks = []
    for name in nms:
        old = next(i for i, p in enumerate(person_dirs) if p.name == name)
        path_blocks.append([p for p, lab in zip(all_paths, labels) if lab == old])
    return seq.name, Ys, labs, nms, path_blocks


def style_heatmap(ax, Y, lengths, hw):
    n, n_cols = Y.shape
    ax.imshow(Y, aspect="auto", cmap="gray", interpolation="nearest",
              vmin=0, vmax=1, origin="upper")
    boundaries = np.cumsum([0] + list(lengths))
    for b in boundaries[1:-1]:
        ax.axvline(b - 0.5, color=BOUNDARY, lw=1.35, zorder=3)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylabel(f"pixel  ({hw}×{hw} = {n})", fontsize=9.5, color=INK)
    ax.set_xlabel("column index  (temporal order within each block)",
                  fontsize=9.2, color=INK, labelpad=6)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#cfcfc9")
    ax.spines["bottom"].set_color("#cfcfc9")
    return boundaries


def draw_block_bar(ax, lengths, labels, n_cols):
    boundaries = np.cumsum([0] + list(lengths))
    for i, (lab, n) in enumerate(zip(labels, lengths)):
        start, end = boundaries[i], boundaries[i + 1]
        ax.axvspan(start, end, color=BLOCK_COLORS[i], alpha=0.92)
        ax.text(0.5 * (start + end), 0.5, f"{lab}  ·  {n}",
                ha="center", va="center", fontsize=8.8, color="white",
                fontweight="bold")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for b in boundaries:
        ax.axvline(b, color=SURFACE, lw=1.6)


def draw_filmstrip(fig, spec, paths, color, ylabel, loader, n_show):
    idxs = even_indices(len(paths), n_show)
    inner = spec.subgridspec(1, len(idxs), wspace=0.045)
    for j, i in enumerate(idxs):
        ax = fig.add_subplot(inner[0, j])
        ax.imshow(loader(paths[i]), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.0)
        if j == 0:
            ax.set_ylabel(ylabel, fontsize=8.4, color=INK, rotation=0,
                          ha="right", va="center", labelpad=10)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")
    print(f"wrote {path.with_suffix('.pdf')}")


def render_showcase(Y, lengths, bar_labels, path_blocks, row_labels, loader,
                    hw, column_noun, title, subtitle, out_path, n_show):
    """Cluster bar + Y heatmap on top, one filmstrip per block below."""
    k = len(lengths)
    n_cols = Y.shape[1]
    n = Y.shape[0]
    fig = plt.figure(figsize=(13.4, 5.4 + 0.95 * k), facecolor=SURFACE)
    outer = GridSpec(
        2, 1, figure=fig, height_ratios=[1.85, 0.78 * k],
        hspace=0.28, left=0.11, right=0.985, top=0.82, bottom=0.04,
    )
    top = outer[0].subgridspec(2, 1, height_ratios=[0.18, 1.55], hspace=0.12)
    bottom = outer[1].subgridspec(k, 1, hspace=0.16)

    ax_bar = fig.add_subplot(top[0])
    draw_block_bar(ax_bar, lengths, bar_labels, n_cols)

    ax_hm = fig.add_subplot(top[1])
    style_heatmap(ax_hm, Y, lengths, hw)

    for i, (paths, ylab) in enumerate(zip(path_blocks, row_labels)):
        draw_filmstrip(fig, bottom[i], paths, BLOCK_COLORS[i], ylab,
                       loader, n_show)

    fig.suptitle(title, fontsize=14, color=INK, fontweight="bold",
                 x=0.11, ha="left", y=0.975)
    fig.text(0.11, 0.938, subtitle, fontsize=9.2, color=MUTED, ha="left")
    fig.text(
        0.11, 0.908,
        f"data matrix Y  ({n} × {n_cols})   ·   "
        f"each column is one flattened {hw}×{hw} {column_noun}",
        fontsize=10.2, color=INK, ha="left",
    )
    save(fig, out_path)


def make_ballet_figure(out_path, k=3, n_show=8):
    dancer_id, Y, labels, descs, path_blocks = first_ballet_example(k)
    lengths = block_lengths(labels)
    render_showcase(
        Y, lengths,
        [f"segment {i + 1}" for i in range(k)],
        path_blocks,
        [f"seg {i + 1}\n{d.split()[0]}" for i, d in enumerate(descs)],
        load_gray, BALLET_DOWN_HW, "frame",
        f"Ballet  —  one dancer, {k} pose-segments concatenated in time",
        f"Dancer {dancer_id}  ·  fold 0  ·  σ = 0  ·  seed = {SEED}  ·  "
        f"the task is to recover the {k - 1} block boundaries in Y",
        out_path, n_show,
    )
    print(f"  dancer {dancer_id}  Y={Y.shape}  blocks={lengths}")
    for i, d in enumerate(descs):
        print(f"  block {i}: {d}  ({lengths[i]} cols)")


def make_surv_figure(out_path, k=5, n_show=8):
    seq_name, Y, labels, names, path_blocks = first_surv_example(k)
    lengths = block_lengths(labels)
    render_showcase(
        Y, lengths,
        [f"{n}" for n in names],
        path_blocks,
        [f"person\n{n}" for n in names],
        load_pgm_gray, SURV_DOWN_HW, "face crop",
        f"Surveillance  —  {k} people from one PETS 2006 sequence",
        f"{seq_name}  ·  fold 0  ·  σ = 0  ·  seed = {SEED}  ·  "
        f"original 96×96 crops below; Y uses {SURV_DOWN_HW}×{SURV_DOWN_HW}",
        out_path, n_show,
    )
    print(f"  {seq_name}  Y={Y.shape}  people={names}  blocks={lengths}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=HERE / "figures")
    ap.add_argument("--ballet-k", type=int, default=3)
    ap.add_argument("--surv-k", type=int, default=5)
    ap.add_argument("--n-show", type=int, default=8,
                    help="thumbnails per block (evenly spaced in time)")
    args = ap.parse_args()

    make_ballet_figure(args.outdir / "ballet_data_matrix.png",
                       k=args.ballet_k, n_show=args.n_show)
    make_surv_figure(args.outdir / "surveillance_faces.png",
                     k=args.surv_k, n_show=args.n_show)


if __name__ == "__main__":
    main()
