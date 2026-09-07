"""Visualize the ballet clustering data matrix Y and its column structure."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cluster_experiment import (
    FRAMES_DIR,
    N_FRAMES_HI,
    N_FRAMES_LO,
    SEED,
    load_ballet_sequences,
)

OUT_PATH = HERE / "ballet_data_matrix_example.png"
K = 5
SEQ_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def load_frame_thumbnail(path, size=64):
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.float64) / 255.0


def main():
    rng = np.random.default_rng(SEED)
    Y, labels, seq_dirs, frame_paths, n_kept = load_ballet_sequences(
        frames_dir=FRAMES_DIR, k=K, rng=rng,
    )

    d, n_cols = Y.shape
    boundaries = np.cumsum([0] + n_kept)
    seq_names = [p.name for p in seq_dirs]

    # Downsample rows for heatmap (every 50th pixel)
    row_idx = np.arange(0, d, max(1, d // 400))
    Y_vis = Y[row_idx] / 255.0

    fig = plt.figure(figsize=(14, 9), facecolor="white")
    gs = GridSpec(
        3, 1, figure=fig, height_ratios=[0.08, 1.2, 1.0], hspace=0.35,
    )

    # --- Top: sequence label bar ---
    ax_bar = fig.add_subplot(gs[0, 0])
    for seq_idx in range(K):
        start, end = boundaries[seq_idx], boundaries[seq_idx + 1]
        ax_bar.axvspan(start, end, color=SEQ_COLORS[seq_idx], alpha=0.85)
        mid = 0.5 * (start + end)
        ax_bar.text(
            mid, 0.5, f"{seq_names[seq_idx]}\n({n_kept[seq_idx]} frames)",
            ha="center", va="center", fontsize=9, color="white", fontweight="bold",
        )
    ax_bar.set_xlim(0, n_cols)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    for b in boundaries:
        ax_bar.axvline(b, color="white", lw=1.5)
    ax_bar.set_title(
        f"Column order: frames grouped by sequence, temporal order preserved within each block  "
        f"(k={K}, n ~ Unif[{N_FRAMES_LO}, {N_FRAMES_HI}], seed={SEED})",
        fontsize=11, loc="left", pad=8,
    )

    # --- Middle: heatmap of Y ---
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
        f"Data matrix Y  shape = ({d:,} x {n_cols})  —  each column is one flattened frame",
        fontsize=11,
    )
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.02, pad=0.01)
    cbar.set_label("intensity", fontsize=9)

    # --- Bottom: example thumbnails in column order ---
    ax_thumbs = fig.add_subplot(gs[2, 0])
    ax_thumbs.set_xlim(0, n_cols)
    ax_thumbs.set_ylim(0, 1)
    ax_thumbs.axis("off")
    ax_thumbs.set_title(
        "Example frames along columns (first 3 per sequence shown at full resolution)",
        fontsize=11, loc="left", pad=8,
    )

    thumb_w = 0.9
    x = 0.0
    shown_per_seq = 3
    counts = [0] * K
    for col, (path, lab) in enumerate(zip(frame_paths, labels)):
        if counts[lab] >= shown_per_seq:
            continue
        thumb = load_frame_thumbnail(path, size=48)
        extent = [x, x + thumb_w, 0.05, 0.95]
        ax_thumbs.imshow(thumb, cmap="gray", extent=extent, aspect="auto")
        ax_thumbs.plot(
            [x + thumb_w / 2], [0.02], marker="v", color=SEQ_COLORS[lab],
            markersize=6, clip_on=False,
        )
        x += thumb_w + 0.15
        counts[lab] += 1

    # Legend
    patches = [
        mpatches.Patch(color=SEQ_COLORS[i], label=f"{seq_names[i]} ({n_kept[i]} cols)")
        for i in range(K)
    ]
    ax_hm.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.9)

    # Annotation for matrix layout
    fig.text(
        0.5, 0.01,
        "Y = [ seq_000001 frames (ascending time) | seq_000002 frames | ... | seq_000005 frames ]",
        ha="center", fontsize=10, style="italic", color="#333333",
    )

    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PATH}")
    print(f"Y shape: {Y.shape}  n_kept: {n_kept}  total columns: {n_cols}")


if __name__ == "__main__":
    main()
