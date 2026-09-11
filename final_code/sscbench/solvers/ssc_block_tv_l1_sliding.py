"""SSC-Block-TV-L1 with a SLIDING (undecimated) block-difference operator.

Objective
---------
    min   lambda_e * ||E||_1  +  (lambda_z/2) ||Y - YX - E||_F^2
          + gamma_q * ||C D^T||_1
    s.t.  X = C_off,  C D^T = Q,  diag(C) = 0

Same objective and same ADMM as ``ssc_block_tv_l1``; only ``D`` changes, plus
two scaling fixes. Three differences, all of them deliberate:

1. Sliding block difference.  ``ssc_block_tv_l1`` uses Db = Ds @ B, the
   difference of NON-OVERLAPPING k-bin averages, so it has only N/k - 1 rows
   pinned at local indices k, 2k, 3k, ...  Those indices are absolute offsets
   into a window whose start is arbitrary, so for k > 1 the same genomic
   content gets re-partitioned by every window (and a jump can only be
   located to within k bins).  Here

       (D c)_j = (1/k) sum_{i=j}^{j+k-1} c_i  -  (1/k) sum_{i=j-k}^{j-1} c_i,
       j = k, ..., N-k

   asks the same difference-of-two-contiguous-averages at EVERY position.
   It is a convolution (one stencil, shifted), hence translation-invariant,
   and its row set is the union of Db's rows over all k phases.  At k = 1 it
   reduces exactly to Db (up to the normalization in 2).

2. D is rescaled to ||D||_2 = 1.  Db's rows have squared norm 2/k, so the
   spectrum of Db^T Db shrinks like O(1/k): at large k the sigma*eigs term in
   ``scale`` vanishes and the C D^T = Q constraint is barely enforced, i.e.
   large k was being weakened by the solver rather than by the prior.
   Normalizing makes gamma_q mean the same thing at every k.

3. ``normalize_y`` divides Y by sigma_1(Y).  It is OFF by default, and
   should normally stay off.  It is an exact reparameterization -- replacing
   Y by Y/s and (lambda_e, lambda_z) by (s*lambda_e, s^2*lambda_z) leaves C
   and X unchanged -- but that is precisely the problem: mu and sigma stay at
   1, so with Y/s the eigenvalues of Y^T Y drop from ~sigma_1^2 to <= 1 and
   lambda_z * s_g no longer competes with mu.  The data term is swamped by
   the ADMM penalty, X collapses to C_off, and the iteration wanders (C
   moving >100% between iteration 50 and 100).  Using it therefore requires
   multiplying lambda_z by ~sigma_1^2 (~1e4 on 10 kb KR windows); at the
   lambda_z ~ 0.1 used for the unnormalized solver it silently produces a
   degenerate solve.  The published solvers here do not normalize the data:
   ``osc.py`` instead makes its penalty data-adaptive (rho = 1.1*sigma_1(X)^2),
   which is the right way to get scale-awareness without moving lambda out of
   data units.  E is rescaled back to the units of Y on return.

Note that lambda_e and lambda_z are not independent: minimizing over E gives
a Huber loss with knee delta = lambda_e/lambda_z, so only that ratio and the
overall weight lambda_z matter.  Sweeps should be designed on (delta,
lambda_z), not on a Cartesian grid of (lambda_e, lambda_z).
"""

import warnings
import numpy as np

warnings.filterwarnings('ignore', message='.*matmul.*', category=RuntimeWarning)


def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


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

def ssc_admm_block_tv_l1_sliding(
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
    snapshot_iter=None,
):
    """
    Parameters
    ----------
    Y            : ndarray (n, N)
    lambda_e     : weight on the element-wise ||E||_1 term
    lambda_z     : weight on reconstruction loss
    gamma_q      : TV weight for the sliding block difference  gamma_q ||C D^T||_1
    block_size   : k, half-width of the sliding average used by D
    mu, sigma    : ADMM penalties
    mu_max       : cap on the penalty continuation
    normalize_y  : divide Y by sigma_1(Y) before solving. Off by default; if
                   enabled, scale lambda_z by ~sigma_1^2 (see module docstring)
    normalize_d  : rescale D to unit spectral norm
    snapshot_iter: if set, copy C after this 1-based iteration into
                   info["C_snapshot"] so a sweep can check cut stability
                   without a second solve

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
    C_snap = None
    snap_at = None if snapshot_iter is None else int(snapshot_iter)
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

        # Element-wise L1 proximal for E
        E = soft_threshold(Y - Y @ X, lambda_e / lambda_z)

        C_off = C - np.diag(np.diag(C))
        Lambda += mu * (X - C_off)
        Pi_Q += sigma * (CDt - Q)

        primal_res = max(
            np.linalg.norm(X - C_off, "fro"),
            np.linalg.norm(CDt - Q, "fro"),
        )
        dual_res = mu * np.linalg.norm(X - X_prev, "fro")
        if snap_at is not None and (it + 1) == snap_at:
            C_snap = np.array(C, copy=True)
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
        "C_snapshot": C_snap if C_snap is not None else (
            np.array(C, copy=True) if snap_at is not None else None
        ),
    }
    return X, C, E * sigma1, info
