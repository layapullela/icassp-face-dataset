"""Show selected face frames for one or two people (clusters) from the surveillance dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from cluster_experiment import SEED, chunk_people, load_sequence, read_pgm, sequence_dirs, subset_people

HERE = Path(__file__).resolve().parent
OUT_PATH = HERE / "surveillance_selected_frames.png"
PERSON_INDICES = (0, 1)  # first two people in train group 0
MAX_COLS = 10


def load_frame(path):
    return read_pgm(path).astype(np.float64) / 255.0


def main():
    rng = np.random.default_rng(SEED)
    seq = sequence_dirs()[0]
    Y, labels, person_dirs, paths, n_kept = load_sequence(seq, rng=rng)

    people = sorted(p.name for p in person_dirs)
    train_groups, _ = chunk_people(people)
    keep_names = train_groups[0]
    _, labs, nms = subset_people(Y, labels, person_dirs, keep_names)

    # Collect paths per person in column order
    person_blocks = []
    for person_idx in PERSON_INDICES:
        person_name = nms[person_idx]
        person_paths = [
            path for path, lab in zip(paths, labels)
            if person_dirs[lab].name == person_name
        ]
        person_blocks.append({
            "name": f"person {person_name}",
            "paths": person_paths,
        })

    section_heights = []
    for block in person_blocks:
        nrows = int(np.ceil(len(block["paths"]) / MAX_COLS))
        section_heights.append(nrows + 0.35)

    fig = plt.figure(
        figsize=(2.0 * MAX_COLS, 2.2 * sum(section_heights)),
        facecolor="white",
    )
    gs = GridSpec(len(person_blocks), 1, figure=fig, height_ratios=section_heights, hspace=0.45)

    for block_idx, block in enumerate(person_blocks):
        paths_block = block["paths"]
        n = len(paths_block)
        nrows = int(np.ceil(n / MAX_COLS))
        inner = gs[block_idx].subgridspec(nrows, MAX_COLS, hspace=0.15, wspace=0.05)

        fig.text(
            0.5,
            inner[0, 0].get_position(fig).y1 + 0.02,
            f"{block['name']}  ({n} frames, left→right = ascending time)",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

        for i, path in enumerate(paths_block):
            ax = fig.add_subplot(inner[i // MAX_COLS, i % MAX_COLS])
            ax.imshow(load_frame(path), cmap="gray", vmin=0, vmax=1)
            ax.set_title(path.stem, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])

        for i in range(n, nrows * MAX_COLS):
            ax = fig.add_subplot(inner[i // MAX_COLS, i % MAX_COLS])
            ax.axis("off")

    fig.suptitle(
        f"{seq.name}: selected frames in data matrix Y (seed={SEED}, train group 0)",
        fontsize=14, y=0.995,
    )
    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT_PATH}")
    for block in person_blocks:
        print(f"  {block['name']}: {len(block['paths'])} frames")


if __name__ == "__main__":
    main()
