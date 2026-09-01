"""Contiguous k-segment partitioning of an ordered affinity matrix via DP.

Drop-in alternative to free (unconstrained) spectral clustering for
problems where the ground-truth clusters are known to be contiguous blocks
along the existing index order (e.g. the SBM benchmark cases in
benchmark_sbm.py, where cluster membership is a contiguous range).

Rather than embedding C's derived affinity and running k-means /
discretize on the eigenvectors (which has no notion of "contiguous"),
this searches exactly over the restricted hypothesis class of contiguous
partitions and picks the one that minimizes the k-way normalized cut --
solvable exactly by dynamic programming because NCut decomposes additively
over segments (see derivation in the docstring below).
"""

import numpy as np


def dp_contiguous_ncut_partition(C, k, min_size=1, symmetrize=True):
    """
    Partition N points (indexed 0..N-1, assumed already in the order in
    which contiguous clusters are expected to appear) into k contiguous
    segments minimizing the k-way normalized cut of the self-expressiveness
    affinity derived from C.

    Normalized-cut objective
    -------------------------
    For a partition into segments S_1..S_k,

        NCut = sum_s [ 1 - within(S_s) / vol(S_s) ]
             = k - sum_s within(S_s) / vol(S_s)

    so minimizing NCut is equivalent to maximizing the additive quantity
    sum_s within(S_s) / vol(S_s), which lets it be solved exactly by DP
    over segment boundaries -- no eigendecomposition or k-means needed.

        within(S) = sum_{i,j in S} W_ij
        vol(S)    = sum_{i in S} deg_i        (deg_i = full-graph row sum)

    Parameters
    ----------
    C : ndarray (N, N)
        Self-expressiveness coefficient matrix (or any square affinity-like
        matrix already in index order).
    k : int
        Number of contiguous segments (clusters).
    min_size : int, default 1
        Minimum allowed segment length; set >1 to forbid degenerate
        near-empty clusters.
    symmetrize : bool, default True
        If True, use W = |C| + |C|^T as the affinity (matches the W used
        elsewhere for OSC / SSC-TV spectral clustering in this benchmark).
        If False, C is used directly and assumed already symmetric and
        nonnegative.

    Returns
    -------
    labels : ndarray (N,), int
        Cluster label 0..k-1 for each point, in index order.
    boundaries : list of int, length k+1
        Segment s covers indices boundaries[s] : boundaries[s+1].
    ncut_value : float
        Achieved (DP-optimal) normalized-cut value for the returned
        partition -- useful as a diagnostic / for comparing against
        unconstrained spectral clustering's NCut on the same W.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square (N, N) matrix.")
    N = C.shape[0]
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1.")
    if k > N:
        raise ValueError(f"k ({k}) cannot exceed N ({N}).")
    if min_size < 1:
        raise ValueError("min_size must be >= 1.")
    if k * min_size > N:
        raise ValueError(
            f"k * min_size ({k * min_size}) exceeds N ({N}); "
            "relax min_size or reduce k."
        )

    W = np.abs(C) + np.abs(C).T if symmetrize else np.array(C, copy=True)
    np.fill_diagonal(W, 0.0)  # no self-loops in within/degree accounting

    # 2D prefix sums for O(1) within(a, b) queries; S[i, j] = sum W[0:i, 0:j]
    S = np.zeros((N + 1, N + 1))
    S[1:, 1:] = np.cumsum(np.cumsum(W, axis=0), axis=1)
    deg = W.sum(axis=1)  # full-graph degree of each node
    deg_cum = np.concatenate([[0.0], np.cumsum(deg)])

    def within(a, b):
        return S[b, b] - S[a, b] - S[b, a] + S[a, a]

    def vol(a, b):
        return deg_cum[b] - deg_cum[a]

    # gain[a, b] = within(a,b) / vol(a,b) for segment [a, b); this is the
    # DP transition value (maximized <=> NCut minimized). vol == 0 (an
    # isolated segment) is given gain 0, matching the NCut convention that
    # an isolated cluster contributes the worst-case NCut_s = 1.
    NEG_INF = -np.inf
    gain = np.full((N + 1, N + 1), NEG_INF)
    for a in range(N):
        for b in range(a + min_size, N + 1):
            v = vol(a, b)
            gain[a, b] = within(a, b) / v if v > 0 else 0.0

    # dp[s, b] = best total gain using exactly s segments covering [0, b)
    dp = np.full((k + 1, N + 1), NEG_INF)
    dp[0, 0] = 0.0
    back = np.full((k + 1, N + 1), -1, dtype=int)

    for s in range(1, k + 1):
        lo = s * min_size
        hi = N - (k - s) * min_size
        for b in range(lo, hi + 1):
            a_lo = (s - 1) * min_size
            a_hi = b - min_size
            best_val, best_a = NEG_INF, -1
            for a in range(a_lo, a_hi + 1):
                prev = dp[s - 1, a]
                if prev == NEG_INF:
                    continue
                val = prev + gain[a, b]
                if val > best_val:
                    best_val, best_a = val, a
            dp[s, b] = best_val
            back[s, b] = best_a

    if dp[k, N] == NEG_INF:
        raise RuntimeError(
            "No feasible contiguous partition found; check k and min_size "
            "against N."
        )

    # Backtrack to recover boundaries.
    boundaries = [N]
    b, s = N, k
    while s > 0:
        a = back[s, b]
        boundaries.append(a)
        b, s = a, s - 1
    boundaries.reverse()

    labels = np.empty(N, dtype=int)
    for seg_idx in range(k):
        labels[boundaries[seg_idx]:boundaries[seg_idx + 1]] = seg_idx

    ncut_value = k - dp[k, N]
    return labels, boundaries, float(ncut_value)


def cluster_from_C_ordered(C, k, min_size=1, symmetrize=True):
    """Drop-in replacement for ``cluster_from_C(coeff, k)`` that restricts
    to contiguous partitions. Only appropriate when the input order is
    known to align with true cluster membership (e.g. the SBM benchmark
    cases, which generate contiguous blocks by construction)."""
    labels, _, _ = dp_contiguous_ncut_partition(
        C, k, min_size=min_size, symmetrize=symmetrize
    )
    return labels


if __name__ == "__main__":
    # Minimal sanity check on a synthetic 3-block affinity matrix.
    rng = np.random.default_rng(0)
    sizes = [50, 60, 90]
    labels_true = np.repeat(np.arange(3), sizes)
    N = len(labels_true)
    same = labels_true[:, None] == labels_true[None, :]
    C_syn = np.where(same, rng.uniform(0.4, 1.0, (N, N)),
                      rng.uniform(0.0, 0.05, (N, N)))
    np.fill_diagonal(C_syn, 0.0)

    labels, boundaries, ncut = dp_contiguous_ncut_partition(C_syn, k=3)
    from sklearn.metrics import adjusted_rand_score
    print("boundaries:", boundaries)
    print("NCut:", ncut)
    print("ARI vs ground truth:", adjusted_rand_score(labels_true, labels))