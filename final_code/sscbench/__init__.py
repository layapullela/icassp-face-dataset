"""Sequential subspace clustering benchmark -- reproduction package.

A trimmed, single-copy version of the experiment described in
``../README.md`` (the method spec). It contains exactly what is needed to
reproduce the headline figures from reported hyperparameters, and nothing
else: no Optuna, no SLURM plumbing, no ablation methods, no k-estimation
path (``k`` is known everywhere in this benchmark).
"""

__all__ = ["config", "metrics", "methods", "data_ballet", "data_surveillance"]
