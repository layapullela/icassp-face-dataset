"""SSC-Block-TV with column-wise L2,1 on E: Uses ||E||_{2,1} instead of ||E||_{2,1}^{block}.

Objective
---------
    min   lambda_e * ||E||_{2,1}  +  (lambda_z/2) ||Y - YX - E||_F^2
          + gamma_q * ||C Db^T||_1
    s.t.  X = C_off,  C Db^T = Q,  diag(C) = 0

Differences from ``ssc_block_tv``:

1. E uses standard column-wise L2,1 sparsity (shrink individual columns) 
   instead of block-wise (contiguous segment) sparsity.

2. The TV term on C still uses the block finite-difference operator Db.
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


def col_soft_threshold(M, tau):
    """Proximal operator of tau * sum_j ||M[:, j]||_2 (column-wise L2,1).

    Each column of M is shrunk toward zero independently using its L2 norm:
        prox(M_j) = max(0, 1 - tau / ||M_j||_2) * M_j
    """
    out = np.empty_like(M)
    for j in range(M.shape[1]):
        col = M[:, j]
        norm = np.linalg.norm(col)
        scale = max(0.0, 1.0 - tau / max(norm, 1e-12))
        out[:, j] = scale * col
    return out


# ── ADMM solver ──────────────────────────────────────────────────────────────

def ssc_admm_block_tv_col21(
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
    lambda_e   : weight on the column-wise ||E||_{2,1} term
    lambda_z   : weight on reconstruction loss
    gamma_q    : TV weight for block-wise differences  gamma_q ||C Db^T||_1
    block_size : k, contiguous block length used for Db
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

        # Column-wise L2,1 proximal for E
        E = col_soft_threshold(Y - Y @ X, lambda_e / lambda_z)

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
