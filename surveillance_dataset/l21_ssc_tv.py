"""SSC-TV-L21: L1 on E, L2,1 on TV auxiliaries of C.

``ssc_admm_nuc_tv`` uses both P = DC (row-wise) and Q = CD^T (column-wise).
``ssc_admm_col_tv`` keeps only Q, so adjacent columns of C are smoothed and
rows are left unregularized.
"""

import warnings
import numpy as np

from ssc_tv import cluster_from_C

warnings.filterwarnings('ignore', message='.*matmul.*', category=RuntimeWarning)


# ── Helpers ───────────────────────────────────────────────────────────────────

def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def block_soft_threshold_cols(M, tau):
    """Proximal operator of tau * ||·||_{2,1} (sum of column L2-norms).

    Each column m_j is shrunk toward zero by the group lasso rule:
        prox(m_j) = max(0, 1 - tau / ||m_j||_2) * m_j
    """
    col_norms = np.linalg.norm(M, axis=0, keepdims=True)          # (1, N)
    scale = np.maximum(1.0 - tau / np.maximum(col_norms, 1e-12), 0.0)
    return scale * M

def block_soft_threshold_rows(M, tau):
    """Proximal operator of tau * ||·||_{2,1} (sum of row L2-norms).

    Each row m_i is shrunk toward zero by the group lasso rule:
        prox(m_i) = max(0, 1 - tau / ||m_i||_2) * m_i
    """
    row_norms = np.linalg.norm(M, axis=1, keepdims=True)          # (N, 1)
    scale = np.maximum(1.0 - tau / np.maximum(row_norms, 1e-12), 0.0)
    return scale * M

def finite_diff_matrix(N):
    """First-order finite-difference operator D ∈ ℝ^{(N-1)×N}."""
    D = np.zeros((N - 1, N))
    idx = np.arange(N - 1)
    D[idx, idx]     = -1.0
    D[idx, idx + 1] =  1.0
    return D


# ── ADMM solver ──────────────────────────────────────────────────────────────

def ssc_admm_nuc_tv(
    Y,
    lambda_e=1.0,
    lambda_z=0.1,
    gamma_p=0.1,
    gamma_q=0.1,
    mu=1.0,
    sigma=1.0,
    max_iter=50,
    tol=1e-4,
):
    """
    Sparse Subspace Clustering with anisotropic Total-Variation regularisation.

    Parameters
    ----------
    Y        : ndarray (n, N)   data matrix (columns = data points)
    lambda_e : float            weight on ||E||_1
    lambda_z : float            weight on reconstruction loss
    gamma_p  : float            TV weight for row-wise differences  γ_p ||P||_{2,1}
    gamma_q  : float            TV weight for column-wise differences  γ_q ||Q||_{2,1}
    mu       : float            ADMM penalty for the X = C_off constraint
    sigma    : float            ADMM penalty for the TV auxiliary constraints
    max_iter : int
    tol      : float            convergence tolerance (max primal Frobenius residual)

    Returns
    -------
    X, C, E : ndarrays
    """
    n, N = Y.shape

    # ── Precompute static quantities ──────────────────────────────────────────
    D = finite_diff_matrix(N)                 # (N-1, N)
    K = D.T @ D                               # (N, N), symmetric PSD
    eigs, V = np.linalg.eigh(K)               # eigs ascending, V orthogonal

    # Sylvester denominator: denom[i,j] = μ + σ(λ_i + λ_j)
    denom = mu + sigma * (eigs[:, None] + eigs[None, :])          # (N, N)
    A_inv = np.linalg.inv(lambda_z * (Y.T @ Y) + mu * np.eye(N))  # for X-update

    # ── Initialise primal and dual variables ──────────────────────────────────
    X = np.zeros((N, N))
    C = np.zeros((N, N))
    E = np.zeros((n, N))
    P = np.zeros((N - 1, N))     # DC   auxiliary
    Q = np.zeros((N, N - 1))     # CD^T auxiliary

    Lambda = np.zeros((N, N))    # dual for X = C_off
    Pi_P   = np.zeros((N - 1, N))   # dual for DC = P
    Pi_Q   = np.zeros((N, N - 1))   # dual for CD^T = Q

    for it in range(max_iter):
        X_prev = X

        # 1. X-update
        C_off = C - np.diag(np.diag(C))
        X = A_inv @ (lambda_z * (Y.T @ (Y - E)) + mu * C_off - Lambda)

        # 2. C-update (Sylvester equation via eigendecomposition of K)
        P_tilde = P - Pi_P / sigma
        Q_tilde = Q - Pi_Q / sigma
        RHS_C   = mu * (X + Lambda / mu) + sigma * (D.T @ P_tilde + Q_tilde @ D)
        C = V @ ((V.T @ RHS_C @ V) / denom) @ V.T
        np.fill_diagonal(C, 0.0)

        # 3-4. P- and Q-updates: column-wise group soft threshold (L2,1)
        DC  = D @ C
        CDt = C @ D.T
        P = block_soft_threshold_rows(DC  + Pi_P / sigma, gamma_p / sigma)
        Q = block_soft_threshold_cols(CDt + Pi_Q / sigma, gamma_q / sigma)

        # 5. E-update
        E = soft_threshold(Y - Y @ X, lambda_e / lambda_z)

        # 6. Dual updates
        C_off = C - np.diag(np.diag(C))
        Lambda += mu    * (X   - C_off)
        Pi_P   += sigma * (DC  - P)
        Pi_Q   += sigma * (CDt - Q)

        # Convergence check
        primal_res = max(
            np.linalg.norm(X   - C_off, 'fro'),
            np.linalg.norm(DC  - P,     'fro'),
            np.linalg.norm(CDt - Q,     'fro'),
        )
        dual_res = mu * np.linalg.norm(X - X_prev, 'fro')
        if primal_res < tol and dual_res < tol:
            break

        mu_max, gamma_0 = 10.0, 1.1
        gamma_step = gamma_0 if max(primal_res, dual_res) < tol else 1.0
        mu = min(mu_max, gamma_step * mu)
        sigma = min(mu_max, gamma_step * sigma)

    return X, C, E


def ssc_admm_col_tv(
    Y,
    lambda_e=1.0,
    lambda_z=0.1,
    gamma_q=0.1,
    mu=1.0,
    sigma=1.0,
    max_iter=50,
    tol=1e-4,
):
    """SSC-TV-L21 with column-wise TV only: γ_q ||Q||_{2,1}, Q = CD^T.

    Drops the row-wise term γ_p ||DC||_{2,1}. Adjacent columns of C (the
    self-expressive coefficients of consecutive samples) are encouraged to
    be similar; rows of C are not.

    The C-update is the right-hand linear system
        C (μ I + σ D^T D) = μ (X + Λ/μ) + σ Q̃ D
    rather than the two-sided Sylvester equation used when both TV terms
    are present.
    """
    n, N = Y.shape

    D = finite_diff_matrix(N)
    K = D.T @ D
    eigs, V = np.linalg.eigh(K)
    scale = mu + sigma * eigs
    A_inv = np.linalg.inv(lambda_z * (Y.T @ Y) + mu * np.eye(N))

    X = np.zeros((N, N))
    C = np.zeros((N, N))
    E = np.zeros((n, N))
    Q = np.zeros((N, N - 1))

    Lambda = np.zeros((N, N))
    Pi_Q = np.zeros((N, N - 1))

    for it in range(max_iter):
        X_prev = X

        C_off = C - np.diag(np.diag(C))
        X = A_inv @ (lambda_z * (Y.T @ (Y - E)) + mu * C_off - Lambda)

        Q_tilde = Q - Pi_Q / sigma
        RHS_C = mu * (X + Lambda / mu) + sigma * (Q_tilde @ D)
        C = ((RHS_C @ V) / scale) @ V.T
        np.fill_diagonal(C, 0.0)

        CDt = C @ D.T
        Q = block_soft_threshold_cols(CDt + Pi_Q / sigma, gamma_q / sigma)

        E = soft_threshold(Y - Y @ X, lambda_e / lambda_z)

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

        mu_max, gamma_0 = 10.0, 1.1
        gamma_step = gamma_0 if max(primal_res, dual_res) < tol else 1.0
        mu = min(mu_max, gamma_step * mu)
        sigma = min(mu_max, gamma_step * sigma)

    return X, C, E


# ── Synthetic sanity check ──────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    from sklearn.metrics import adjusted_rand_score

    cluster_sizes = [20, 25, 15, 20]
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
    same = labels[:, None] == labels[None, :]
    probs = np.where(same, 0.75, 0.05)
    N = sum(cluster_sizes)
    upper = np.triu(rng.random((N, N)) < probs, k=0).astype(float)
    Y = upper + upper.T - np.diag(np.diag(upper))

    print(f"Y: {Y.shape},  clusters: {cluster_sizes}\n")
    t0 = time.perf_counter()
    X, C, E = ssc_admm_nuc_tv(Y, lambda_e=1.0, lambda_z=0.1, gamma_p=0.1, gamma_q=0.1)
    pred = cluster_from_C(X, k=len(cluster_sizes))
    print(f"PQ  ARI = {adjusted_rand_score(labels, pred):.4f}   "
          f"time = {time.perf_counter() - t0:.2f}s")
    t0 = time.perf_counter()
    X, C, E = ssc_admm_col_tv(Y, lambda_e=1.0, lambda_z=0.1, gamma_q=0.1)
    pred = cluster_from_C(X, k=len(cluster_sizes))
    print(f"col ARI = {adjusted_rand_score(labels, pred):.4f}   "
          f"time = {time.perf_counter() - t0:.2f}s")