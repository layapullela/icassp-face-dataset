"""Diagnostics for the review: contiguity-prior baselines + ADMM convergence."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "surveillance_dataset"))

import cluster_experiment as ce  # noqa: E402
from dp_contiguous_partition import cluster_from_C_ordered  # noqa: E402
import l21_ssc_tv as L  # noqa: E402


def trivial_baselines(mats, tag):
    rows = {"equal-chunk (ignores data)": [], "DP-NCut on raw Gram Y^T Y": []}
    for _, _, Y, lab, nms in mats:
        k = len(nms)
        N = Y.shape[1]
        eq = np.repeat(np.arange(k), np.diff(np.linspace(0, N, k + 1, dtype=int)))
        rows["equal-chunk (ignores data)"].append(ce.metrics(lab, eq))
        g = cluster_from_C_ordered(Y.T @ Y, k)
        rows["DP-NCut on raw Gram Y^T Y"].append(ce.metrics(lab, g))
    for name, sc in rows.items():
        print(
            f"  {tag:6s} {name:28s} "
            f"ACC={np.mean([s['acc'] for s in sc]):.4f}  "
            f"ARI={np.mean([s['ari'] for s in sc]):.4f}  n={len(sc)}"
        )


def admm_residuals(Y, max_iter=50):
    """Re-run the SSC-TV ADMM loop, logging residuals (mirrors ssc_admm_nuc_tv)."""
    lambda_e, lambda_z, gamma_p, gamma_q = 1.0, 0.1, 0.1, 0.1
    mu = sigma = 1.0
    n, N = Y.shape
    D = L.finite_diff_matrix(N)
    Kmat = D.T @ D
    eigs, V = np.linalg.eigh(Kmat)
    denom = mu + sigma * (eigs[:, None] + eigs[None, :])
    A_inv = np.linalg.inv(lambda_z * (Y.T @ Y) + mu * np.eye(N))
    X = np.zeros((N, N)); C = np.zeros((N, N)); E = np.zeros((n, N))
    P = np.zeros((N - 1, N)); Q = np.zeros((N, N - 1))
    Lam = np.zeros((N, N)); PiP = np.zeros((N - 1, N)); PiQ = np.zeros((N, N - 1))
    for it in range(max_iter):
        X_prev = X
        C_off = C - np.diag(np.diag(C))
        X = A_inv @ (lambda_z * (Y.T @ (Y - E)) + mu * C_off - Lam)
        RHS = mu * (X + Lam / mu) + sigma * (D.T @ (P - PiP / sigma) + (Q - PiQ / sigma) @ D)
        C = V @ ((V.T @ RHS @ V) / denom) @ V.T
        np.fill_diagonal(C, 0.0)
        DC = D @ C; CDt = C @ D.T
        P = L.block_soft_threshold_rows(DC + PiP / sigma, gamma_p / sigma)
        Q = L.block_soft_threshold_cols(CDt + PiQ / sigma, gamma_q / sigma)
        E = L.soft_threshold(Y - Y @ X, lambda_e / lambda_z)
        C_off = C - np.diag(np.diag(C))
        Lam += mu * (X - C_off); PiP += sigma * (DC - P); PiQ += sigma * (CDt - Q)
        pr = max(np.linalg.norm(X - C_off, "fro"),
                 np.linalg.norm(DC - P, "fro"),
                 np.linalg.norm(CDt - Q, "fro"))
        dr = mu * np.linalg.norm(X - X_prev, "fro")
        if it in (0, 9, 24, 49):
            print(f"    iter {it + 1:3d}  primal={pr:.3e}  dual={dr:.3e}  (tol=1e-4)")
    return X


def main():
    loaded = ce.load_all_sequences()
    people = sorted(p.name for p in loaded[0][3])
    tr, te = ce.chunk_people(people)
    for sig in (0.0, 0.5):
        rng = np.random.default_rng(ce.SEED)
        train, test = ce.build_split_mats(loaded, tr, te, sig, rng)
        print(f"\n--- surveillance sigma={sig} ---")
        trivial_baselines(train, "train")
        trivial_baselines(test, "test")

    rng = np.random.default_rng(ce.SEED)
    train, _ = ce.build_split_mats(loaded, tr, te, 0.0, rng)
    Y = train[0][2]
    print(f"\n--- SSC-TV ADMM residuals, Y={Y.shape}, defaults, mu=sigma=1 ---")
    X = admm_residuals(Y, 50)
    print("    ||X||_F =", f"{np.linalg.norm(X, 'fro'):.4e}")
    print("\n--- same problem, 300 iterations: does the label output change? ---")
    lab = train[0][3]
    X50, _, _ = L.ssc_admm_nuc_tv(Y, 1.0, 0.1, 0.1, 0.1, max_iter=50)
    X300, _, _ = L.ssc_admm_nuc_tv(Y, 1.0, 0.1, 0.1, 0.1, max_iter=300)
    print("    ARI @50 iters :", f"{ce.metrics(lab, ce.cluster_from_C(X50, k=5))['ari']:.4f}")
    print("    ARI @300 iters:", f"{ce.metrics(lab, ce.cluster_from_C(X300, k=5))['ari']:.4f}")
    print("    rel. change in X:",
          f"{np.linalg.norm(X300 - X50, 'fro') / np.linalg.norm(X300, 'fro'):.4f}")


if __name__ == "__main__":
    main()
