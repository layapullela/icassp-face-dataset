"""Clustering metrics: ARI, NMI, accuracy, and a k_pred degeneracy check.

Definitions match the development tree exactly (README §6), with one
deliberate substitution: clustering accuracy uses
``scipy.optimize.linear_sum_assignment`` instead of the tree's from-scratch
e-maxx Hungarian implementation. That hand-rolled version had a
``if not found: break`` path that could return a partial assignment; it
validated clean against scipy over 300 random label pairs, and scipy was
already a dependency, so swapping it removes the risk at no cost. ARI and NMI
are kept as the same hand-rolled contingency-table formulas the tree used
(both were verified against scikit-learn over the same 300 pairs, 0
mismatches) so the reported numbers stay bit-comparable.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def contingency(y_true, y_pred):
    """Counts table, rows = true labels, cols = predicted labels."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    rows = np.unique(y_true)
    cols = np.unique(y_pred)
    table = np.zeros((rows.size, cols.size), dtype=np.int64)
    r_idx = {v: i for i, v in enumerate(rows)}
    c_idx = {v: i for i, v in enumerate(cols)}
    for t, p in zip(y_true, y_pred):
        table[r_idx[int(t)], c_idx[int(p)]] += 1
    return table


def adjusted_rand_score(y_true, y_pred):
    """Adjusted Rand index from the contingency table."""
    table = contingency(y_true, y_pred)
    n = table.sum()
    if n < 2:
        return 1.0

    def _comb2(x):
        x = np.asarray(x, dtype=np.float64)
        return (x * (x - 1.0) / 2.0).sum()

    sum_ij = _comb2(table)
    sum_i = _comb2(table.sum(axis=1))
    sum_j = _comb2(table.sum(axis=0))
    total = n * (n - 1.0) / 2.0
    expected = sum_i * sum_j / total
    maximum = 0.5 * (sum_i + sum_j)
    if maximum == expected:
        return 1.0
    return float((sum_ij - expected) / (maximum - expected))


def normalized_mutual_info(y_true, y_pred):
    """NMI with arithmetic-mean normalization: 2*MI / (H(y) + H(yhat))."""
    table = contingency(y_true, y_pred).astype(np.float64)
    n = table.sum()
    if n == 0:
        return 0.0
    p_ij = table / n
    p_i = p_ij.sum(axis=1, keepdims=True)
    p_j = p_ij.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = p_ij * np.log(p_ij / (p_i @ p_j))
    mi = float(np.nansum(np.where(p_ij > 0, term, 0.0)))
    h_true = float(-np.sum(p_i * np.log(np.where(p_i > 0, p_i, 1.0))))
    h_pred = float(-np.sum(p_j * np.log(np.where(p_j > 0, p_j, 1.0))))
    denom = h_true + h_pred
    if denom <= 0.0:
        return 1.0 if mi == 0.0 else 0.0
    return float(2.0 * mi / denom)


def clustering_accuracy(y_true, y_pred):
    """Best-permutation matched fraction (1 - misclassified/total)."""
    table = contingency(y_true, y_pred)
    rows, cols = linear_sum_assignment(-table)
    return float(table[rows, cols].sum() / table.sum())


def metrics(y_true, y_pred):
    """All reported metrics for one matrix, plus the k_pred degeneracy check."""
    return {
        "acc": clustering_accuracy(y_true, y_pred),
        "nmi": normalized_mutual_info(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
        "k_pred": int(np.unique(np.asarray(y_pred)).size),
    }


def column_normalize(Y):
    norms = np.linalg.norm(Y, axis=0, keepdims=True)
    return Y / np.maximum(norms, 1e-12)


def apply_noise(Y01, sigma, rng):
    """Add i.i.d. N(0, sigma^2) in [0,1] pixel units, then column-normalize.

    Shared by both datasets -- the two development copies of this function
    were byte-identical. Column normalization is scale-invariant, which is
    why sigma=0 results are insensitive to the pixel-scaling bug fixed in
    README §8.1 while sigma>0 results were not.
    """
    noise = rng.standard_normal(Y01.shape)
    scale = np.asarray(sigma, dtype=float)
    if scale.ndim == 0:
        Y = Y01 + scale * noise
    else:
        Y = Y01 + noise * scale.reshape(1, -1)
    return column_normalize(Y)
