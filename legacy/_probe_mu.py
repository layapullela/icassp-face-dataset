"""Minimal check: does the ADMM penalty mu ever grow in ssc_admm_nuc_tv?

Small synthetic problem so it runs in seconds on a login node.
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "surveillance_dataset"))
from l21_ssc_tv import finite_diff_matrix, block_soft_threshold_cols, \
    block_soft_threshold_rows, soft_threshold  # noqa: E402


def trace_mu(Y, lambda_e, lambda_z, gamma_p, gamma_q,
             mu=1.0, sigma=1.0, max_iter=50, tol=1e-4):
    n, N = Y.shape
    D = finite_diff_matrix(N)
    K = D.T @ D
    eigs, V = np.linalg.eigh(K)
    denom = mu + sigma * (eigs[:, None] + eigs[None, :])
    A_inv = np.linalg.inv(lambda_z * (Y.T @ Y) + mu * np.eye(N))

    X = np.zeros((N, N)); C = np.zeros((N, N)); E = np.zeros((n, N))
    P = np.zeros((N - 1, N)); Q = np.zeros((N, N - 1))
    Lambda = np.zeros((N, N))
    Pi_P = np.zeros((N - 1, N)); Pi_Q = np.zeros((N, N - 1))

    mus, primals = [], []
    for it in range(max_iter):
        X_prev = X
        C_off = C - np.diag(np.diag(C))
        X = A_inv @ (lambda_z * (Y.T @ (Y - E)) + mu * C_off - Lambda)
        P_tilde = P - Pi_P / sigma
        Q_tilde = Q - Pi_Q / sigma
        RHS_C = mu * (X + Lambda / mu) + sigma * (D.T @ P_tilde + Q_tilde @ D)
        C = V @ ((V.T @ RHS_C @ V) / denom) @ V.T
        np.fill_diagonal(C, 0.0)
        DC = D @ C
        CDt = C @ D.T
        P = block_soft_threshold_rows(DC + Pi_P / sigma, gamma_p / sigma)
        Q = block_soft_threshold_cols(CDt + Pi_Q / sigma, gamma_q / sigma)
        E = soft_threshold(Y - Y @ X, lambda_e / lambda_z)
        C_off = C - np.diag(np.diag(C))
        Lambda += mu * (X - C_off)
        Pi_P += sigma * (DC - P)
        Pi_Q += sigma * (CDt - Q)

        primal_res = max(np.linalg.norm(X - C_off, 'fro'),
                         np.linalg.norm(DC - P, 'fro'),
                         np.linalg.norm(CDt - Q, 'fro'))
        dual_res = mu * np.linalg.norm(X - X_prev, 'fro')
        mus.append(mu); primals.append(primal_res)
        if primal_res < tol and dual_res < tol:
            break
        mu_max, gamma_0 = 10.0, 1.1
        gamma_step = gamma_0 if max(primal_res, dual_res) < tol else 1.0
        mu = min(mu_max, gamma_step * mu)
        sigma = min(mu_max, gamma_step * sigma)

    C_off = C - np.diag(np.diag(C))
    return mus, primals, np.linalg.norm(X - C_off, 'fro') / max(np.linalg.norm(X, 'fro'), 1e-12)


rng = np.random.default_rng(0)
# 3 subspaces, 20 points each, ambient dim 40 -> N=60 columns
blocks = [rng.standard_normal((40, 4)) @ rng.standard_normal((4, 20)) for _ in range(3)]
Y = np.concatenate(blocks, axis=1)
Y = Y / np.maximum(np.linalg.norm(Y, axis=0, keepdims=True), 1e-12)
print(f"synthetic Y: {Y.shape}\n")

from l21_ssc_tv import ssc_admm_nuc_tv  # noqa: E402

PARAMS = (
    ("best known-k found", dict(lambda_e=0.02264, lambda_z=0.08313,
                                gamma_p=0.003745, gamma_q=0.6823)),
    ("old khat best     ", dict(lambda_e=6.491, lambda_z=0.6468,
                                gamma_p=5.98, gamma_q=2.61)),
)

print("--- OLD behaviour (mu pinned, reproduced by trace_mu) ---")
for label, p in PARAMS:
    mus, primals, rel = trace_mu(Y, **p)
    print(f"{label}: iters={len(mus)}  mu[0]={mus[0]:.3f}  mu[-1]={mus[-1]:.3f}  "
          f"n_distinct_mu={len(set(np.round(mus, 6)))}  "
          f"primal {primals[0]:.4g} -> {primals[-1]:.4g}  rel_gap={rel:.5f}")

print("\n--- NEW behaviour (patched ssc_admm_nuc_tv) ---")
for label, p in PARAMS:
    X, C, E = ssc_admm_nuc_tv(Y, max_iter=50, **p)
    C_off = C - np.diag(np.diag(C))
    rel = np.linalg.norm(X - C_off, 'fro') / max(np.linalg.norm(X, 'fro'), 1e-12)
    print(f"{label}: rel_gap={rel:.5f}")
