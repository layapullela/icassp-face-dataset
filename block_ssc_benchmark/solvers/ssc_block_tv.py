"""SSC-Block-TV: L1 on CD^T where D is a *block* difference operator, and
block-wise (contiguous-segment) group sparsity on E instead of per-column.

Objective
---------
    min   lambda_e * ||E||_{2,1}^{block}  +  (lambda_z/2) ||Y - YX - E||_F^2
          + gamma_q * ||C Db^T||_1
    s.t.  X = C_off,  C Db^T = Q,  diag(C) = 0

Differences from ``ssc_col_tv.ssc_admm_col_tv``:

1. ``D`` -> ``Db``: instead of the (N-1) x N adjacent finite-difference
   operator, we build a block finite-difference operator with block size
   ``k``. Columns 0..N-1 are partitioned into contiguous blocks of size k
   (last block may be short); ``Db = Dstride @ B`` where ``B`` is a
   (num_blocks x N) block-averaging matrix and ``Dstride`` is the ordinary
   adjacent-difference operator over the *blocks* (num_blocks-1 x
   num_blocks). So ``Db`` penalizes differences between neighboring block
   *averages* of C's rows rather than between neighboring individual
   columns -- smoothing is enforced only across block boundaries, not
   within a block.

2. ||E||_{2,1}^{block}: the group-lasso proximal step for E now groups
   whole contiguous segments of ``k`` columns (samples) together (using
   their joint Frobenius norm) rather than shrinking a single column to
   zero. A block of frames is entirely zeroed out of E together if it is
   collectively well-explained; otherwise the whole block is kept as
   (scaled) residual. This is the natural generalization of column-wise
   L2,1 sparsity to segment-wise sparsity, consistent with the contiguous-
   cluster assumption used by ``dp_contiguous_partition``.

The Q-update remains an elementwise soft-threshold (L1), matching the
stated objective ``gamma_q * ||C Db^T||_1`` (anisotropic, not L2,1).
"""

import warnings
import numpy as np

from ssc_tv import cluster_from_C

warnings.filterwarnings('ignore', message='.*matmul.*', category=RuntimeWarning)


def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


# ── Block operator construction ──────────────────────────────────────────────

def block_boundaries(N, k):
    """Contiguous block boundaries of size k over N columns (last block short)."""
    bounds = list(range(0, N, k))
    bounds.append(N)
    return bounds


def block_average_matrix(N, bounds):
    """B: (num_blocks, N) with B[b, j] = 1/|block b| for j in block b."""
    num_blocks = len(bounds) - 1
    B = np.zeros((num_blocks, N))
    for b in range(num_blocks):
        a, c = bounds[b], bounds[b + 1]
        B[b, a:c] = 1.0 / (c - a)
    return B


def stride_diff_matrix(num_blocks):
    """Adjacent finite-difference operator over block indices."""
    Ds = np.zeros((num_blocks - 1, num_blocks))
    idx = np.arange(num_blocks - 1)
    Ds[idx, idx] = -1.0
    Ds[idx, idx + 1] = 1.0
    return Ds


def block_finite_diff_matrix(N, k):
    """Db = Dstride @ B, shape (num_blocks-1, N)."""
    bounds = block_boundaries(N, k)
    B = block_average_matrix(N, bounds)
    Ds = stride_diff_matrix(len(bounds) - 1)
    return Ds @ B, bounds


def block_soft_threshold_segments(M, tau, bounds):
    """Proximal operator of tau * sum_b ||M[:, block_b]||_F (segment L2,1).

    Each contiguous column-segment (block) of M is shrunk toward zero as a
    whole, using the group lasso rule on its joint Frobenius norm:
        prox(M_b) = max(0, 1 - tau / ||M_b||_F) * M_b
    """
    out = np.empty_like(M)
    for b in range(len(bounds) - 1):
        a, c = bounds[b], bounds[b + 1]
        block = M[:, a:c]
        norm = np.linalg.norm(block)
        scale = max(0.0, 1.0 - tau / max(norm, 1e-12))
        out[:, a:c] = scale * block
    return out


# ── ADMM solver ──────────────────────────────────────────────────────────────

def ssc_admm_block_tv(
    Y,
    lambda_e=1.0,
    lambda_z=0.1,
    gamma_q=0.1,
    block_size=5,
    mu=1.0,
    sigma=1.0,
    max_iter=50,
    tol=1e-4,
):
    """
    Parameters
    ----------
    Y          : ndarray (n, N)
    lambda_e   : weight on the block-wise ||E||_{2,1} term
    lambda_z   : weight on reconstruction loss
    gamma_q    : TV weight for block-wise differences  gamma_q ||C Db^T||_1
    block_size : k, contiguous block length used for both Db and E's groups
    mu, sigma  : ADMM penalties
    max_iter, tol : as usual

    Returns
    -------
    X, C, E, bounds
    """
    n, N = Y.shape
    mu_max, gamma_0 = 10.0, 1.1

    Db, bounds = block_finite_diff_matrix(N, block_size)
    K = Db.T @ Db
    eigs, V = np.linalg.eigh(K)
    scale = mu + sigma * eigs
    s_g, U = np.linalg.eigh(Y.T @ Y)

    n_aux = Db.shape[0]
    X = np.zeros((N, N))
    C = np.zeros((N, N))
    E = np.zeros((n, N))
    Q = np.zeros((N, n_aux))

    Lambda = np.zeros((N, N))
    Pi_Q = np.zeros((N, n_aux))

    for it in range(max_iter):
        X_prev = X

        C_off = C - np.diag(np.diag(C))
        RHS_X = lambda_z * (Y.T @ (Y - E)) + mu * C_off - Lambda
        X = U @ ((U.T @ RHS_X) / (lambda_z * s_g + mu)[:, None])

        Q_tilde = Q - Pi_Q / sigma
        RHS_C = mu * (X + Lambda / mu) + sigma * (Q_tilde @ Db)
        C = ((RHS_C @ V) / scale) @ V.T
        np.fill_diagonal(C, 0.0)

        CDt = C @ Db.T
        Q = soft_threshold(CDt + Pi_Q / sigma, gamma_q / sigma)

        E = block_soft_threshold_segments(Y - Y @ X, lambda_e / lambda_z, bounds)

        C_off = C - np.diag(np.diag(C))
        Lambda += mu * (X - C_off)
        Pi_Q += sigma * (CDt - Q)

        primal_res = max(
            np.linalg.norm(X - C_off, "fro"),
            np.linalg.norm(CDt - Q, "fro"),
        )
        dual_res = mu * np.linalg.norm(X - X_prev, "fro")
        if primal_res < tol and dual_res < tol:
            break

        if mu < mu_max or sigma < mu_max:
            mu = min(mu_max, gamma_0 * mu)
            sigma = min(mu_max, gamma_0 * sigma)
            scale = mu + sigma * eigs

    return X, C, E, bounds