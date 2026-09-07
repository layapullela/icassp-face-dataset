"""Show the selected ballet frames for one or two sequences."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cluster_experiment import FRAMES_DIR, SEED, load_ballet_sequences

OUT_PATH = HERE / "ballet_selected_frames.png"
SEQ_INDICES = (0, 1)  # seq_000001 and seq_000002
MAX_COLS = 10


def load_frame(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float64) / 255.0


def main():
    rng = np.random.default_rng(SEED)
    _, labels, seq_dirs, frame_paths, n_kept = load_ballet_sequences(
        frames_dir=FRAMES_DIR, k=5, rng=rng,
    )

    seq_blocks = []
    start = 0
    for seq_idx, n in enumerate(n_kept):
        end = start + n
        if seq_idx in SEQ_INDICES:
            seq_blocks.append({
                "name": seq_dirs[seq_idx].name,
                "paths": frame_paths[start:end],
            })
        start = end

    section_heights = []
    for block in seq_blocks:
        nrows = int(np.ceil(len(block["paths"]) / MAX_COLS))
        section_heights.append(nrows + 0.35)  # extra for title

    fig = plt.figure(figsize=(2.0 * MAX_COLS, 2.2 * sum(section_heights)), facecolor="white")
    gs = GridSpec(len(seq_blocks), 1, figure=fig, height_ratios=section_heights, hspace=0.45)

    for block_idx, block in enumerate(seq_blocks):
        paths = block["paths"]
        n = len(paths)
        nrows = int(np.ceil(n / MAX_COLS))
        inner = gs[block_idx].subgridspec(nrows, MAX_COLS, hspace=0.15, wspace=0.05)

        fig.text(
            0.5,
            inner[0, 0].get_position(fig).y1 + 0.02,
            f"{block['name']}  ({n} frames, left→right = ascending time)",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

        for i, path in enumerate(paths):
            ax = fig.add_subplot(inner[i // MAX_COLS, i % MAX_COLS])
            ax.imshow(load_frame(path), cmap="gray", vmin=0, vmax=1)
            ax.set_title(path.stem, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        for i in range(n, nrows * MAX_COLS):
            ax = fig.add_subplot(inner[i // MAX_COLS, i % MAX_COLS])
            ax.axis("off")

    fig.suptitle(f"Selected frames in data matrix Y (seed={SEED})", fontsize=14, y=0.995)
    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PATH}")
    for block in seq_blocks:
        print(f"  {block['name']}: {len(block['paths'])} frames")


if __name__ == "__main__":
    main()
