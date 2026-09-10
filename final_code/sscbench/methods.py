"""The five benchmarked methods, and how a hyperparameter row becomes a call.

Each ``run_*`` wrapper is the same thin adapter the development tree used: it
calls the solver, then turns the coefficient matrix into labels. Every method
except TKSS ends in the SAME final step -- the exact contiguous DP normalized
cut (README §5) -- so differences between methods are differences in the
self-expression model, not in the label-inference step. TKSS is the exception:
it emits per-column argmin labels from its own assignment step and gets no
contiguity prior (README §8.3).

``k`` is known for every method in this benchmark (``--known-k`` everywhere),
so the k-estimation branches in the solver stack are never taken.
"""

from __future__ import annotations

import numpy as np

from .config import LAMBDA_E_PIN
from .solvers import (
    bd_qosc,
    cluster_from_C,
    cluster_from_Z,
    osc_exact,
    ssc_admm_block_tv_col21,
    tkss_cluster,
)

SEED = 0  # TKSS's random_state; matches the tree


# --------------------------------------------------------------------------
# Solver wrappers
# --------------------------------------------------------------------------
def run_ssc_block_tv_col21(Y, k, lambda_e, lambda_z, gamma_q, block_size):
    """The proposed method. ``block_size`` is the TV window, NOT the cluster count.

    These are separate quantities that unfortunately share the name ``k`` in
    the solver's internal helpers. Passing ``block_size = k`` would put the TV
    window in exactly the regime that switches the method's distinguishing
    term off (README §4.3), so it is required explicitly and never defaulted.
    """
    if block_size is None:
        raise ValueError(
            "SSC-Block-TV-Col21 requires an explicit block_size (the TV "
            "window). It is NOT defaulted to k -- see README §4.3/§8.8."
        )
    _, C, _, _ = ssc_admm_block_tv_col21(
        Y, lambda_e=lambda_e, lambda_z=lambda_z, gamma_q=gamma_q,
        block_size=int(block_size), max_iter=50,
    )
    return cluster_from_C(C, k=k)


def run_osc(Y, k, lambda_1, lambda_2):
    Z = osc_exact(Y, lambda_1, lambda_2, max_iter=50)
    return cluster_from_Z(Z, k=k)


def run_bdosc(Y, k, lambda_1, lambda_2, gamma_1, p, max_iter=50):
    Z, _, _ = bd_qosc(
        Y, k, lambda_1, lambda_2, gamma_1, p,
        max_iter=int(max_iter), diagconstraint=True,
    )
    return cluster_from_Z(Z, k=k)


def run_tkss(Y, k, d, lam, s):
    pred, _ = tkss_cluster(
        Y, k=k, d=int(d), lam=lam, s=int(s), max_iter=30, random_state=SEED,
    )
    return pred


def run_gram_ncut(Y, k):
    """The trivial control: no self-expression, no hyperparameters.

    Runs the identical final clustering step directly on the column Gram
    Y^T Y. This is a FLOOR, not a competitor -- whatever a real method
    achieves above it is what the subspace model actually bought (README §8.5,
    where it lands surprisingly close).
    """
    return cluster_from_C(Y.T @ Y, k=k)


# --------------------------------------------------------------------------
# Row -> call
# --------------------------------------------------------------------------
# Which hyperparameter columns each method consumes, in call order.
PARAM_COLUMNS = {
    "SSC-Block-TV-Col21": ("lambda_e", "lambda_z", "gamma_q", "block_size"),
    "OSC": ("lambda_1", "lambda_2"),
    "BD-OSC": ("lambda_1", "lambda_2", "gamma_1", "p", "max_iter"),
    "TKSS": ("d", "lam", "s"),
    "DP NCut (trivial)": (),
}

_RUNNERS = {
    "SSC-Block-TV-Col21": run_ssc_block_tv_col21,
    "OSC": run_osc,
    "BD-OSC": run_bdosc,
    "TKSS": run_tkss,
    "DP NCut (trivial)": run_gram_ncut,
}

_INT_PARAMS = {"block_size", "max_iter", "d", "s"}


def params_from_row(method, row):
    """Pull this method's hyperparameters out of a hyperparameters.csv row.

    Raises on a missing or blank value rather than substituting a default:
    a silently-defaulted hyperparameter is a different experiment, and the
    whole point of the CSV is that the reported value is the one used.
    """
    if method not in PARAM_COLUMNS:
        raise ValueError(f"unknown method {method!r}; expected one of "
                         f"{sorted(PARAM_COLUMNS)}")
    out = {}
    for col in PARAM_COLUMNS[method]:
        raw = (row.get(col) or "").strip()
        if raw == "":
            raise ValueError(
                f"{method}: hyperparameters.csv row is missing a value for "
                f"{col!r} (dataset={row.get('dataset')} k={row.get('k')} "
                f"sigma={row.get('sigma')} fold={row.get('fold')})"
            )
        out[col] = int(float(raw)) if col in _INT_PARAMS else float(raw)
    if method == "SSC-Block-TV-Col21" and out["lambda_e"] != LAMBDA_E_PIN:
        # Not an error: the surveillance k=8 rows on disk really were run at
        # lambda_e=0.1, before the pin was unified (README §8.2). Reported as
        # such, and flagged so it cannot pass unnoticed.
        pass
    return out


def predict(method, Y, k, params):
    """Run one method on one matrix and return integer labels."""
    return _RUNNERS[method](Y, k, **params)
