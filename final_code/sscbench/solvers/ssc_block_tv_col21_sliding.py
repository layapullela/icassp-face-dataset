"""SSC-Block-TV-Col21 with a SLIDING (undecimated) block-difference operator.

Objective
---------
    min   lambda_e * ||E||_{2,1}  +  (lambda_z/2) ||Y - YX - E||_F^2
          + gamma_q * ||C D^T||_1
    s.t.  X = C_off,  C D^T = Q,  diag(C) = 0

Same objective and same ADMM as ``ssc_block_tv_col21``; only ``D`` changes.
This is ``ssc_block_tv_l1_sliding`` with the E-step swapped back to the
column-wise L2,1 group penalty Col21 uses (whole frames are outliers or not),
so it is directly comparable to Col21's reported numbers with only the TV
operator differing -- see ``ssc_block_tv_l1_sliding`` for the two deliberate
differences that come with the sliding operator itself:

1. Sliding block difference.  ``ssc_block_tv_col21`` uses Db = Ds @ B, the
   difference of NON-OVERLAPPING k-bin averages, so it has only N/k - 1 rows
   pinned at local indices k, 2k, 3k, ...  Here

       (D c)_j = (1/k) sum_{i=j}^{j+k-1} c_i  -  (1/k) sum_{i=j-k}^{j-1} c_i,
       j = k, ..., N-k

   asks the same difference-of-two-contiguous-averages at EVERY position: a
   convolution (one stencil, shifted), hence translation-invariant, whose row
   set is the union of Db's rows over all k phases. At k = 1 it reduces
   exactly to Db (up to the normalization below).

2. D is rescaled to ||D||_2 = 1 by default (``normalize_d``). Db's rows have
   squared norm 2/k, so the spectrum of Db^T Db shrinks like O(1/k); without
   rescaling, large k weakens the C D^T = Q constraint through the solver
   rather than through the prior. Pass ``normalize_d=False`` to keep D's rows
   at the same per-row scale as Db (squared norm 2/k) instead, which is the
   setting that makes a reused Db-tuned gamma_q mean the same per-row thing
   in both operators -- useful when comparing against Col21 numbers that were
   tuned for Db.

``normalize_y`` (divide Y by sigma_1(Y)) is OFF by default and should
normally stay off -- see ``ssc_block_tv_l1_sliding``'s docstring point 3 for
why turning it on silently produces a degenerate solve unless lambda_z is
rescaled to compensate. Col21 itself does not normalize Y.
"""

import warnings
import numpy as np

warnings.filterwarnings('ignore', message='.*matmul.*', category=RuntimeWarning)


def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


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


# ── Sliding block operator ───────────────────────────────────────────────────

def sliding_block_diff_matrix(N, k, normalize=True):
    """D: (N-2k+1, N), row j = mean(c[j:j+k]) - mean(c[j-k:j]).

    k is clamped to [1, (N-1)//2] so at least one row exists. Returns
    (D, centers, k_eff) where centers[r] is the bin the r-th row straddles.
    """
    k = max(1, min(int(k), max((N - 1) // 2, 1)))
    centers = list(range(k, N - k + 1))
    D = np.zeros((len(centers), N))
    for r, j in enumerate(centers):
        D[r, j - k:j] = -1.0 / k
        D[r, j:j + k] = 1.0 / k
    if normalize:
        D /= np.linalg.norm(D, 2)
    return D, centers, k


# ── ADMM solver ──────────────────────────────────────────────────────────────

def ssc_admm_block_tv_col21_sliding(
    Y,
    lambda_e=1.0,
    lambda_z=0.1,
    gamma_q=0.1,
    block_size=5,
    mu=1.0,
    sigma=1.0,
    max_iter=50,
    tol=1e-4,
    mu_max=10.0,
    normalize_y=False,
    normalize_d=True,
):
    """
    Parameters
    ----------
    Y            : ndarray (n, N)
    lambda_e     : weight on the column-wise ||E||_{2,1} term
    lambda_z     : weight on reconstruction loss
    gamma_q      : TV weight for the sliding block difference  gamma_q ||C D^T||_1
    block_size   : k, half-width of the sliding average used by D
    mu, sigma    : ADMM penalties
    mu_max       : cap on the penalty continuation
    normalize_y  : divide Y by sigma_1(Y) before solving. Off by default; if
                   enabled, scale lambda_z by ~sigma_1^2 (see
                   ssc_block_tv_l1_sliding's docstring)
    normalize_d  : rescale D to unit spectral norm

    Returns
    -------
    X, C, E, info
        ``info`` has n_iter, primal_res, dual_res, converged, sigma1, k_eff,
        n_tv_rows -- so a sweep can reject configurations that did not
        converge instead of silently scoring them.
    """
    n, N = Y.shape
    gamma_0 = 1.1

    sigma1 = float(np.linalg.norm(Y, 2)) if normalize_y else 1.0
    if sigma1 <= 0:
        sigma1 = 1.0
    Y = Y / sigma1

    D, centers, k_eff = sliding_block_diff_matrix(N, block_size, normalize_d)
    K = D.T @ D
    eigs, V = np.linalg.eigh(K)
    scale = mu + sigma * eigs
    s_g, U = np.linalg.eigh(Y.T @ Y)

    n_aux = D.shape[0]
    X = np.zeros((N, N))
    C = np.zeros((N, N))
    E = np.zeros((n, N))
    Q = np.zeros((N, n_aux))

    Lambda = np.zeros((N, N))
    Pi_Q = np.zeros((N, n_aux))

    primal_res = dual_res = float("nan")
    converged = False
    for it in range(max_iter):
        X_prev = X

        C_off = C - np.diag(np.diag(C))
        RHS_X = lambda_z * (Y.T @ (Y - E)) + mu * C_off - Lambda
        X = U @ ((U.T @ RHS_X) / (lambda_z * s_g + mu)[:, None])

        Q_tilde = Q - Pi_Q / sigma
        RHS_C = mu * (X + Lambda / mu) + sigma * (Q_tilde @ D)
        C = ((RHS_C @ V) / scale) @ V.T
        np.fill_diagonal(C, 0.0)

        CDt = C @ D.T
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
            converged = True
            break

        if mu < mu_max or sigma < mu_max:
            mu = min(mu_max, gamma_0 * mu)
            sigma = min(mu_max, gamma_0 * sigma)
            scale = mu + sigma * eigs

    info = {
        "n_iter": it + 1,
        "primal_res": float(primal_res),
        "dual_res": float(dual_res),
        "converged": bool(converged),
        "sigma1": sigma1,
        "k_eff": k_eff,
        "n_tv_rows": n_aux,
        "centers": centers,
    }
    return X, C, E * sigma1, info
