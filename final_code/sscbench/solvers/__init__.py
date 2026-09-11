"""Solver stack, copied VERBATIM from the development tree.

The six benchmark modules in this directory are byte-identical to their
originals in ``surveillance_dataset/`` (see ``SOLVER_CHECKSUMS.md5``, and
``../../verify_reproduction.py --check-solvers``). They are deliberately not
reformatted, renamed, or refactored: they are the numerical core of the
benchmark, and an unmodified copy is the only version of them whose results
provably match the published numbers.

``ssc_block_tv_l1.py`` is an extra Block-TV variant (element-wise L1 on E
instead of column-wise L2,1) used by the Hi-C experiment. It is not one of
the five published methods.

They import each other with flat, non-package imports (``from ssc_tv import
cluster_from_C``), which is why this file puts its own directory on
``sys.path`` before importing them. That single ``sys.path`` insert is the
whole price of keeping the copies verbatim, and it is confined to this file --
nothing else in ``sscbench`` manipulates the import path.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from bdosc import bd_qosc                                    # noqa: E402
from osc import osc_exact, cluster_from_Z                    # noqa: E402
from ssc_block_tv_col21 import ssc_admm_block_tv_col21       # noqa: E402
from ssc_block_tv_l1 import ssc_admm_block_tv_l1             # noqa: E402
from ssc_tv import cluster_from_C                            # noqa: E402
from tkss import tkss_cluster                                # noqa: E402

__all__ = [
    "bd_qosc",
    "osc_exact",
    "cluster_from_Z",
    "ssc_admm_block_tv_col21",
    "ssc_admm_block_tv_l1",
    "cluster_from_C",
    "tkss_cluster",
]
