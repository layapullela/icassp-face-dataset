"""SSC with sparse affinity Z and sliding block-TV on C.

Objective
---------
    min   (1/2) ||Y - Y C||_F^2  +  lambda_1 ||Z||_1  +  lambda_2 ||C D^T||_1
    s.t.  C = Z,  C D^T = Q,  diag(C) = 0

Same hyperparameter shape as OSC (``osc.py``):

  lambda_1    elementwise sparsity of the coefficient / affinity matrix Z
  lambda_2    temporal TV  (OSC uses ||Z R||_{1,2} on first differences;
              here it is ||C D^T||_1 on the sliding block-difference D)
  block_size  width k of D; at k = 1 this is a first difference, like OSC's R

No sparse residual E and no extra reconstruction weight -- the data term is
ordinary least-squares self-expression, as in OSC's (1/2)||X - XZ||_F^2 after
eliminating E. Cluster from W = |Z| + |Z^T|.

D is the sliding (undecimated) block-difference from
``ssc_block_tv_l1_sliding.sliding_block_diff_matrix``, rescaled to ||D||_2 = 1
by default. This module is new (not one of the checksum-locked copies).
"""

import warnings
import numpy as np

from ssc_block_tv_l1_sliding import sliding_block_diff_matrix, soft_threshold

warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)


def ssc_admm_sparse_block_tv(
    Y,
    lambda_1=0.1,
    lambda_2=0.1,
    block_size=5,
    sigma=1.0,
    rho=1.0,
    max_iter=50,
    tol=1e-4,
    mu_max=10.0,
    normalize_d=True,
):
    """Sparse-affinity sliding block-TV ADMM.

    Parameters
    ----------
    Y           : ndarray (n, N), columns are samples
    lambda_1    : weight on ||Z||_1 (affinity sparsity)
    lambda_2    : weight on ||C D^T||_1 (sliding block-TV)
    block_size  : k, half-width of the sliding average used by D
    sigma, rho  : ADMM penalties for C D^T = Q and C = Z
    max_iter, tol, mu_max : ADMM control (penalties ramp by 1.1 up to mu_max)
    normalize_d : rescale D to unit spectral norm

    Returns
    -------
    Z, C, info
        ``Z`` is the sparse coefficient matrix used for clustering.
        ``C`` is the TV-constrained copy (C ≈ Z at convergence).
    """
    _n, N = Y.shape
    gamma_0 = 1.1
    lambda_1 = float(lambda_1)
    lambda_2 = float(lambda_2)
    rho = float(rho)
    sigma = float(sigma)

    D, centers, k_eff = sliding_block_diff_matrix(N, block_size, normalize_d)
    eigs, V = np.linalg.eigh(D.T @ D)
    s_g, U = np.linalg.eigh(Y.T @ Y)
    s_g = np.clip(s_g, 0.0, None)
    YtY = (U * s_g) @ U.T

    n_aux = D.shape[0]
    C = np.zeros((N, N))
    Z = np.zeros((N, N))
    Q = np.zeros((N, n_aux))
    Pi_Q = np.zeros((N, n_aux))
    Pi_Z = np.zeros((N, N))

    primal_res = dual_res = float("nan")
    converged = False

    def _denom():
        d = s_g[:, None] + rho + sigma * eigs[None, :]
        return np.maximum(d, 1e-12)

    denom = _denom()

    for it in range(max_iter):
        C_prev = C

        Q_tilde = Q - Pi_Q / sigma
        Z_tilde = Z - Pi_Z / rho
        RHS = YtY + sigma * (Q_tilde @ D) + rho * Z_tilde
        C = U @ ((U.T @ RHS @ V) / denom) @ V.T
        # Do NOT zero diag(C) here.  The C-subproblem is not separable across
        # entries, so projecting its unconstrained minimizer is not the
        # minimizer of the diag-constrained subproblem: hard-zeroing biases the
        # iteration and it converges (primal_res ~ 1e-12, converged=True) to a
        # point that does not minimize the stated objective.  diag(C) = 0 is
        # already enforced exactly through the Z-step -- soft_threshold followed
        # by fill_diagonal IS the prox of lambda_1||.||_1 + ind{diag = 0} -- and
        # the C = Z coupling then drives max|diag(C)| to ~1e-12 on its own.
        CDt = C @ D.T
        Q = soft_threshold(CDt + Pi_Q / sigma, lambda_2 / sigma)
        Pi_Q += sigma * (CDt - Q)

        Z = soft_threshold(C + Pi_Z / rho, lambda_1 / rho)
        np.fill_diagonal(Z, 0.0)
        Pi_Z += rho * (C - Z)

        primal_res = max(
            float(np.linalg.norm(CDt - Q, "fro")),
            float(np.linalg.norm(C - Z, "fro")),
        )
        dual_res = float(np.linalg.norm(C - C_prev, "fro"))
        if primal_res < tol and dual_res < tol:
            converged = True
            break

        if sigma < mu_max or rho < mu_max:
            sigma = min(mu_max, gamma_0 * sigma)
            rho = min(mu_max, gamma_0 * rho)
            denom = _denom()

    info = {
        "n_iter": it + 1,
        "primal_res": float(primal_res),
        "dual_res": float(dual_res),
        "converged": bool(converged),
        "k_eff": k_eff,
        "n_tv_rows": n_aux,
        "centers": centers,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "block_size": int(k_eff),
    }
    return Z, C, info
