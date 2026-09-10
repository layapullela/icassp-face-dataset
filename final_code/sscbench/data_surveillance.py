"""Surveillance (PETS 2006 P1E/P1L person crops) -> clustering matrices.

One matrix is one (sequence, group-of-k-people) pair: the columns of the k
chosen people concatenated in sorted person order, giving k contiguous blocks
of 50-100 columns each (README §2.1).

REPRODUCIBILITY NOTE -- the RNG streams here are load-bearing.

  * ``load_all_sequences`` consumes ONE ``default_rng(SEED)`` stream across
    all 24 sequences, in ``sequence_dirs()`` order. Loading a subset of
    sequences, or loading them in a different order, changes every frame
    sample downstream. So the full set is always loaded.
  * Noise is drawn per sigma from a fresh ``default_rng(SEED)``, one
    ``standard_normal`` call per sequence in the same order, and the noisy Y
    is then SHARED across all CV folds. That is why a fold can be evaluated
    standalone and still match the sequential run.

Both details are inherited exactly from the development tree; they are the
reason a single fold reproduces without replaying the others.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .config import (
    N_FRAMES_HI,
    N_FRAMES_LO,
    SEED,
    SURVEILLANCE_ROOTS,
    SURV_DOWN_HW,
)
from .metrics import apply_noise


def read_pgm(path):
    """Minimal binary (P5) PGM reader; the crops are 96x96 grayscale."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"{path}: expected P5 PGM, got {magic!r}")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        width, height = map(int, line.split())
        maxval = int(f.readline())
        dtype = np.uint8 if maxval < 256 else np.uint16
        pixels = np.frombuffer(f.read(), dtype=dtype)
    if pixels.size != width * height:
        raise ValueError(
            f"{path}: expected {width * height} pixels, got {pixels.size}")
    return pixels.reshape(height, width)


def downsample_image(img, size=SURV_DOWN_HW):
    """Bilinear resize to size x size, float64. Matches the tree's PIL path."""
    pil = Image.fromarray(np.asarray(img, dtype=np.float32), mode="F")
    pil = pil.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float64)


def sequence_dirs(roots=SURVEILLANCE_ROOTS):
    """The 24 sequence folders, P1E then P1L, each sorted by name."""
    seqs = []
    for root in roots:
        seqs.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    return seqs


def load_sequence(root, rng):
    """One sequence -> (Y_raw, labels, person_dirs).

    Per person: n ~ Unif{N_FRAMES_LO..N_FRAMES_HI} (capped at folder length)
    and a start index, so the kept frames are CONSECUTIVE. Shorter folders
    keep everything they have.
    """
    lo, hi = N_FRAMES_LO, max(N_FRAMES_LO, N_FRAMES_HI)
    person_dirs = sorted((p for p in root.iterdir() if p.is_dir()),
                         key=lambda p: p.name)
    images, labels, n_kept = [], [], []
    for person_idx, person_dir in enumerate(person_dirs):
        frames = sorted(person_dir.glob("*.pgm"), key=lambda p: p.name)
        n = int(rng.integers(lo, hi + 1))
        n = min(n, len(frames))
        start = int(rng.integers(0, len(frames) - n + 1))
        for frame_path in frames[start:start + n]:
            images.append(downsample_image(read_pgm(frame_path)).reshape(-1))
            labels.append(person_idx)
        n_kept.append(n)
    Y = np.stack(images, axis=1)
    return Y, np.asarray(labels, dtype=int), person_dirs, n_kept


def load_all_sequences(verbose=True):
    """All 24 sequences, pixels scaled to [0,1]. ONE shared RNG stream.

    Returns a list of (name, Y01, labels, person_dirs). This is the expensive
    step (~24 x 25 people x 50-100 frames of image decoding); callers should
    do it once and reuse it across k, sigma and fold.
    """
    rng = np.random.default_rng(SEED)
    loaded = []
    for seq in sequence_dirs():
        Y_raw, labels, person_dirs, n_kept = load_sequence(seq, rng)
        loaded.append((seq.name, Y_raw / 255.0, labels, person_dirs))
        if verbose:
            print(f"  loaded {seq.name}: Y={Y_raw.shape} "
                  f"people={len(person_dirs)} min_frames={min(n_kept)}")
    return loaded


def people_names(loaded):
    return sorted(p.name for p in loaded[0][3])


def chunk_folds(names, k, seed=SEED):
    """Shuffle the 25 people once, cut into disjoint groups of k.

    Each group rotates through as the held-out test group, so fold i's test
    identities are ``chunk_folds(...)[i]``. The remainder is dropped (1 person
    at k=3, 0 at k=5, 1 at k=8).
    """
    names = np.asarray(sorted(names))
    rng = np.random.default_rng(seed)
    names = names[rng.permutation(len(names))]
    n_groups = len(names) // k
    names = names[: n_groups * k]
    return [sorted(g.tolist()) for g in names.reshape(n_groups, k)]


def subset_people(Y, labels, person_dirs, keep_names):
    """Keep only ``keep_names``' columns, relabelled 0..k-1 in that order."""
    name_to_old = {p.name: i for i, p in enumerate(person_dirs)}
    ids = [name_to_old[n] for n in keep_names]
    mask = np.isin(labels, ids)
    remap = {int(old): new for new, old in enumerate(ids)}
    return (Y[:, mask],
            np.array([remap[int(x)] for x in labels[mask]], dtype=int),
            list(keep_names))


def noisy_sequences(loaded, sigma):
    """One noisy Y per sequence for this sigma, shared across all folds."""
    rng = np.random.default_rng(SEED)
    return [(name, apply_noise(Y01, sigma, rng), labels, person_dirs)
            for name, Y01, labels, person_dirs in loaded]


def fold_test_matrices(loaded, k, sigma, fold):
    """The 24 test matrices (one per sequence) for one (k, sigma, fold).

    Yields (sequence_name, Y, labels, people). Ground truth is k contiguous
    blocks in column order.
    """
    groups = chunk_folds(people_names(loaded), k)
    if not 0 <= fold < len(groups):
        raise ValueError(
            f"fold {fold} out of range: k={k} gives {len(groups)} folds")
    test_group = groups[fold]
    for name, Y, labels, person_dirs in noisy_sequences(loaded, sigma):
        Ys, labs, nms = subset_people(Y, labels, person_dirs, test_group)
        yield name, Ys, labs, nms


def n_folds(k, n_people=25):
    return n_people // k
