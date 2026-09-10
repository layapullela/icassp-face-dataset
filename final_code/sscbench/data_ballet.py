"""Ballet clips -> clustering matrices, one dancer's pose-segments at a time.

The task is NOT "which clip is this frame from" and NOT "which dancer is
this" -- both were tried and dropped, because raw-pixel separability on clip
identity is high enough that a trivial similarity method nearly solves it,
leaving nothing for subspace clustering to contribute. What is benchmarked is:
cluster ONE dancer's own pose-segments (appearances) against each other, where
a pose-segment is one contiguous run of frames carrying that dancer's id
inside one sequence (README §2.2).

REPRODUCIBILITY NOTE -- ballet's seeding is per DANCER, not per fold.

  * frame sampling: ``default_rng(SEED + dancer_id)``
  * noise:          ``default_rng([SEED, dancer_id])``

Neither stream involves the fold index or sigma. That makes a dancer's
matrices identical in every fold that uses them -- as this fold's test dancer,
as a train dancer, or in a single-fold job that skips every earlier fold. It
also means the sigma levels deliberately share common random numbers. An
earlier version drew noise from one stream consumed in fold order, so fold i
run standalone saw different noise than fold i inside a full sequential run
(README §8.8); the per-dancer seeding is the fix, and it is what makes
leave-one-dancer-out folds independently reproducible here.
"""

from __future__ import annotations

from collections import defaultdict
from math import comb

import numpy as np
import scipy.io as sio
from PIL import Image

from .config import (
    BALLET_DOWN_HW,
    BALLET_EXCLUDED_DANCERS,
    BALLET_FRAMES_DIR,
    BALLET_LABELS_PATH,
    BALLET_MATRICES_PER_DANCER,
    N_FRAMES_HI,
    N_FRAMES_LO,
    SEED,
)
from .metrics import apply_noise


def load_image(path, size=BALLET_DOWN_HW):
    """Grayscale -> size x size -> flattened, scaled to [0, 1].

    THE SINGLE SCALING POINT. Do not divide by 255 again downstream. This
    matters because ``apply_noise`` adds N(0, sigma^2) in these units before
    column-normalizing: on raw [0,255] pixels a sigma of 0.25/0.5 perturbs the
    normalized columns by <1e-3, i.e. the noise axis silently becomes a no-op.
    That was a real bug in the tree's CV path (README §8.1) -- ballet sigma=0.5
    was equivalent to surveillance sigma~0.002, a ~255x mismatch.
    """
    img = Image.open(path)
    if img.mode != "L":
        img = img.convert("L")
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    return (np.array(img, dtype=np.float64) / 255.0).reshape(-1)


def sequence_dirs(frames_dir=BALLET_FRAMES_DIR):
    return sorted(p for p in frames_dir.iterdir() if p.is_dir())


def load_dancer_segments(frames_dir=BALLET_FRAMES_DIR,
                         labels_path=BALLET_LABELS_PATH):
    """dancer_id -> [(seq_name, frame_start, frame_end)], frame_end exclusive.

    ``labels.mat`` holds one per-frame dancer-id array per sequence, in the
    same sorted order as ``sequence_dirs()`` (frame counts verified to match
    for all 44 sequences). A sequence can hold 1-2 contiguous dancer segments.
    """
    seq_dirs = sequence_dirs(frames_dir)
    labels_by_seq = sio.loadmat(str(labels_path))["labels"][0]
    if len(labels_by_seq) != len(seq_dirs):
        raise ValueError(f"labels.mat has {len(labels_by_seq)} sequences, "
                         f"found {len(seq_dirs)} in {frames_dir}")
    segments = defaultdict(list)
    for seq_dir, arr in zip(seq_dirs, labels_by_seq):
        arr = arr.flatten()
        cur, start = int(arr[0]), 0
        for j in range(1, len(arr) + 1):
            if j == len(arr) or int(arr[j]) != cur:
                segments[cur].append((seq_dir.name, start, j))
                if j < len(arr):
                    cur, start = int(arr[j]), j
    return dict(segments)


def sample_combos(pool, k, n_combos, rng):
    """Sample n_combos distinct k-subsets of pool (as sorted tuples)."""
    pool = list(pool)
    if len(pool) < k:
        raise ValueError(f"pool size {len(pool)} < k={k}")
    n_combos = min(int(n_combos), comb(len(pool), k))
    seen, groups, attempts = set(), [], 0
    max_attempts = max(n_combos * 1000, 1000)
    while len(groups) < n_combos:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"could not sample {n_combos} distinct "
                               f"{k}-subsets from {len(pool)} items")
        g = tuple(sorted(pool[i] for i in rng.choice(len(pool), size=k,
                                                     replace=False)))
        if g in seen:
            continue
        seen.add(g)
        groups.append(list(g))
    return groups


def load_segment_frames(seq_dir, lo, hi, rng):
    """n ~ Unif{LO..HI} CONSECUTIVE frames from a random start inside [lo, hi)."""
    frames = sorted(seq_dir.glob("*.jpg"))[lo:hi]
    n_lo, n_hi = N_FRAMES_LO, max(N_FRAMES_LO, N_FRAMES_HI)
    n = min(int(rng.integers(n_lo, n_hi + 1)), len(frames))
    start = int(rng.integers(0, len(frames) - n + 1))
    return np.stack([load_image(p) for p in frames[start:start + n]], axis=1)


def build_pose_matrices(segments, k, n_matrices, name_to_dir, rng):
    """Up to n_matrices matrices pooling k of this dancer's pose-segments.

    Columns are concatenated in sorted segment-index order and labelled by
    position 0..k-1 within the combo, so ground truth is k contiguous blocks.
    If fewer than n_matrices distinct k-combos exist (dancer 4 at k=3 has
    exactly one), combos are cycled with a FRESH frame draw each time.
    """
    n_all = comb(len(segments), k)
    combos = sample_combos(range(len(segments)), k, min(n_matrices, n_all), rng)
    out = []
    for i in range(n_matrices):
        combo = combos[i % len(combos)]
        cols, labs, descs = [], [], []
        for pos, seg_idx in enumerate(combo):
            seq_name, lo, hi = segments[seg_idx]
            Yi = load_segment_frames(name_to_dir[seq_name], lo, hi, rng)
            cols.append(Yi)
            labs.append(np.full(Yi.shape[1], pos, dtype=int))
            descs.append(f"{seq_name}:{lo}-{hi}")
        out.append((np.concatenate(cols, axis=1), np.concatenate(labs), descs))
    return out


def eligible_dancers(dancer_segments, k):
    """Dancers with >= k segments, excluding dancer 8.

    Dancer 8 is excluded from CV entirely: 14 segments against 3-11 for
    everyone else, and its segments are 19-49 frames -- an outlier in both
    combinatorics and length (README §2.2).
    """
    return sorted(d for d, segs in dancer_segments.items()
                  if len(segs) >= k and d not in BALLET_EXCLUDED_DANCERS)


_CLEAN_CACHE = {}


def clean_dancer_matrices(dancer_segments, dancer_id, k, name_to_dir,
                          n_matrices=BALLET_MATRICES_PER_DANCER, cache=True):
    """This dancer's PRE-NOISE matrices, memoized across sigma.

    Because the frame sampler is seeded from ``dancer_id`` alone, the clean
    matrices are identical for every sigma and every fold that uses this
    dancer -- so decoding them once and reusing them is exact, not an
    approximation. Without this, running all three sigmas re-decodes the same
    few thousand JPEGs three times, which dominates the runtime.

    Roughly 20 MB per (dancer, k); pass ``cache=False`` if memory is tight.
    """
    key = (dancer_id, k, n_matrices)
    if cache and key in _CLEAN_CACHE:
        return _CLEAN_CACHE[key]
    frame_rng = np.random.default_rng(SEED + dancer_id)
    built = build_pose_matrices(dancer_segments[dancer_id], k, n_matrices,
                                name_to_dir, frame_rng)
    if cache:
        _CLEAN_CACHE[key] = built
    return built


def dancer_matrices(dancer_segments, dancer_id, k, sigma, name_to_dir,
                    n_matrices=BALLET_MATRICES_PER_DANCER, cache=True):
    """This dancer's ``n_matrices`` noisy matrices. Seeded from dancer_id ONLY.

    Yields (example_index, Y, labels, segment_descriptions). Independent of
    fold index and of every other dancer -- see the module docstring.
    """
    built = clean_dancer_matrices(dancer_segments, dancer_id, k, name_to_dir,
                                  n_matrices, cache=cache)
    # Fresh noise stream per call, consumed in matrix order -- the same
    # sequence the development tree drew, so cached clean matrices do not
    # change the noise realization.
    noise_rng = np.random.default_rng([SEED, dancer_id])
    for i, (Y01, labels, descs) in enumerate(built):
        yield i, apply_noise(Y01, sigma, noise_rng), labels, descs


def fold_test_matrices(dancer_segments, k, sigma, fold, name_to_dir,
                       n_matrices=BALLET_MATRICES_PER_DANCER):
    """Test matrices for leave-one-dancer-out fold ``fold``.

    Fold i holds out ``eligible_dancers(...)[i]``; its test set is that
    dancer's ``n_matrices`` matrices.
    """
    eligible = eligible_dancers(dancer_segments, k)
    if not 0 <= fold < len(eligible):
        raise ValueError(f"fold {fold} out of range: k={k} gives "
                         f"{len(eligible)} folds (dancers {eligible})")
    held_out = eligible[fold]
    for idx, Y, labels, descs in dancer_matrices(
            dancer_segments, held_out, k, sigma, name_to_dir, n_matrices):
        yield held_out, idx, Y, labels, descs


def name_to_dir_map(frames_dir=BALLET_FRAMES_DIR):
    return {p.name: p for p in sequence_dirs(frames_dir)}
