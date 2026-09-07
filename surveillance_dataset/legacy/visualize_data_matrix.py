"""Visualize the surveillance face clustering data matrix Y and its column structure."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

from cluster_experiment import (
    K,
    N_FRAMES_HI,
    N_FRAMES_LO,
    SEED,
    chunk_people,
    load_sequence,
    read_pgm,
    sequence_dirs,
    subset_people,
)

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "surveillance_data_matrix_example.png"
PERSON_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def load_frame(path):
    return read_pgm(path).astype(np.float64) / 255.0


def main():
    rng = np.random.default_rng(SEED)
    seq = sequence_dirs()[0]
    Y, labels, person_dirs, paths, n_kept = load_sequence(seq, rng=rng)

    people = sorted(p.name for p in person_dirs)
    train_groups, _ = chunk_people(people)
    keep_names = train_groups[0]
    Ys, labs, nms = subset_people(Y, labels, person_dirs, keep_names)

    # Paths in the same column order as Ys
    subset_paths = []
    for path, lab in zip(paths, labels):
        if person_dirs[lab].name in nms:
            subset_paths.append(path)

    # Per-person frame counts in subset
    person_n_kept = [int(np.sum(labs == i)) for i in range(len(nms))]
    boundaries = np.cumsum([0] + person_n_kept)

    d, n_cols = Ys.shape

    # Downsample rows for heatmap
    row_idx = np.arange(0, d, max(1, d // 400))
    Y_vis = Ys[row_idx] / 255.0

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.08, 1.2, 1.0], hspace=0.35)

    # --- Top: person label bar ---
    ax_bar = fig.add_subplot(gs[0, 0])
    for person_idx in range(len(nms)):
        start, end = boundaries[person_idx], boundaries[person_idx + 1]
        ax_bar.axvspan(start, end, color=PERSON_COLORS[person_idx], alpha=0.85)
        mid = 0.5 * (start + end)
        ax_bar.text(
            mid, 0.5, f"person {nms[person_idx]}\n({person_n_kept[person_idx]} frames)",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold",
        )
    ax_bar.set_xlim(0, n_cols)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    for b in boundaries:
        ax_bar.axvline(b, color="white", lw=1.5)
    ax_bar.set_title(
        f"{seq.name}: columns grouped by person, temporal order preserved within each block  "
        f"(k={K}, n ~ Unif[{N_FRAMES_LO}, {N_FRAMES_HI}], seed={SEED}, train group 0)",
        fontsize=11, loc="left", pad=8,
    )

    # --- Middle: heatmap ---
    ax_hm = fig.add_subplot(gs[1, 0])
    im = ax_hm.imshow(
        Y_vis, aspect="auto", cmap="gray", interpolation="nearest",
        vmin=0, vmax=1,
    )
    for b in boundaries[1:-1]:
        ax_hm.axvline(b - 0.5, color="#FFD700", lw=1.2, alpha=0.9)
    ax_hm.set_ylabel(f"pixel index (subsampled, d={d:,})", fontsize=10)
    ax_hm.set_xlabel("column index (frame)", fontsize=10)
    ax_hm.set_title(
        f"Data matrix Y  shape = ({d:,} x {n_cols})  —  each column is one flattened face frame",
        fontsize=11,
    )
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.02, pad=0.01)
    cbar.set_label("intensity", fontsize=9)

    # --- Bottom: example thumbnails ---
    ax_thumbs = fig.add_subplot(gs[2, 0])
    ax_thumbs.set_xlim(0, n_cols)
    ax_thumbs.set_ylim(0, 1)
    ax_thumbs.axis("off")
    ax_thumbs.set_title(
        "Example frames along columns (first 3 per person shown)",
        fontsize=11, loc="left", pad=8,
    )

    thumb_w = 0.9
    x = 0.0
    shown_per_person = 3
    counts = [0] * len(nms)
    for path, lab in zip(subset_paths, labs):
        if counts[lab] >= shown_per_person:
            continue
        thumb = load_frame(path)
        thumb = np.array(
            Image.fromarray((thumb * 255).astype(np.uint8)).resize(
                (48, 48), Image.Resampling.BILINEAR,
            )
        ) / 255.0
        extent = [x, x + thumb_w, 0.05, 0.95]
        ax_thumbs.imshow(thumb, cmap="gray", extent=extent, aspect="auto")
        ax_thumbs.plot(
            [x + thumb_w / 2], [0.02], marker="v",
            color=PERSON_COLORS[lab], markersize=6, clip_on=False,
        )
        x += thumb_w + 0.15
        counts[lab] += 1

    patches = [
        mpatches.Patch(
            color=PERSON_COLORS[i],
            label=f"person {nms[i]} ({person_n_kept[i]} cols)",
        )
        for i in range(len(nms))
    ]
    ax_hm.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.9)

    people_str = " | ".join(
        f"person {n} frames (ascending time)" for n in nms
    )
    fig.text(
        0.5, 0.01,
        f"Y = [ {people_str} ]",
        ha="center", fontsize=10, style="italic", color="#333333",
    )

    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PATH}")
    print(f"sequence: {seq.name}")
    print(f"Y shape: {Ys.shape}  people: {nms}  frames/person: {person_n_kept}")


if __name__ == "__main__":
    main()
