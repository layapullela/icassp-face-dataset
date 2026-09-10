"""Constants and dataset locations. Every number here is load-bearing.

These values define the experiment. Changing one silently changes what is
being measured, so they live in one place rather than being repeated across
scripts. See ``../README.md`` (§2 datasets, §4.3 pins) for the justification
behind each.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Where the raw frames live
# --------------------------------------------------------------------------
# The image corpora (24 surveillance sequences of .pgm crops, 44 ballet clips
# of .jpg frames) are large and are NOT copied into this package -- it points
# at them in place. Override with SSCBENCH_DATA_ROOT if you move them.
_DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("SSCBENCH_DATA_ROOT", _DEFAULT_DATA_ROOT))

SURVEILLANCE_ROOTS = (DATA_ROOT / "surveillance_dataset" / "P1E",
                      DATA_ROOT / "surveillance_dataset" / "P1L")
BALLET_FRAMES_DIR = DATA_ROOT / "ballet_dataset" / "frames_tracked"
BALLET_LABELS_PATH = DATA_ROOT / "ballet_dataset" / "labels.mat"

# --------------------------------------------------------------------------
# Shared experiment constants
# --------------------------------------------------------------------------
SEED = 0                    # every RNG in the benchmark derives from this
N_FRAMES_LO = 50            # frames per cluster ~ Unif{LO, ..., HI}
N_FRAMES_HI = 100           # (the "balanced" regime; the imbalance variant
                            #  used LO=10 and is not part of this package)
SIGMAS = (0.0, 0.25, 0.5)   # noise levels, added in [0,1] pixel units

SURV_DOWN_HW = 24           # surveillance frames -> 24x24 -> n = 576
BALLET_DOWN_HW = 30         # ballet frames      -> 30x30 -> n = 900

BALLET_EXCLUDED_DANCERS = {8}        # 14 short cameo segments; see README §2.2
BALLET_MATRICES_PER_DANCER = 12      # --n-matrices-per-dancer default

SURV_TUNE_POOL_GROUPS = 10           # tuning-pool size (tuning only; unused
                                     # for inference, recorded for provenance)

# --------------------------------------------------------------------------
# The pinned block_size (the TV window; NOT the cluster count k)
# --------------------------------------------------------------------------
# block_size is pinned rather than searched (README §4.3): 2 is the smallest
# window for which the block-difference operator D_b is defined at all, and
# the TV term stops binding as the window grows.
#
# NOTE the surveillance k=8 exception. The runs on disk -- and therefore the
# hyperparameters reported in hyperparameters.csv -- used block_size=3 there.
# The block-size-searched probe (README §7e) later showed 2 beats 3 at k=8 at
# every sigma, so the pin *should* become a uniform 2. It is left at 3 here
# because that is what actually produced the reported numbers; the CSV carries
# the value per row, so this dict is only a fallback.
BLOCK_SIZE_BY_K = {
    "surveillance": {3: 2, 5: 2, 8: 3},
    "ballet": {3: 2, 5: 2},
}

# lambda_e is pinned to one a-priori constant on both datasets (README §4.3):
# it enters the solver only as tau = lambda_e / lambda_z, and 0.01 is the only
# grid value that keeps the error term E alive rather than thresholding it to
# exactly zero.
LAMBDA_E_PIN = 0.01

# --------------------------------------------------------------------------
# The five benchmarked methods, in figure order
# --------------------------------------------------------------------------
METHODS = ["SSC-Block-TV-Col21", "OSC", "BD-OSC", "TKSS", "DP NCut (trivial)"]

# Short slug used in result filenames and in hyperparameters.csv
METHOD_SLUG = {
    "SSC-Block-TV-Col21": "ssc_block_tv_col21",
    "OSC": "osc",
    "BD-OSC": "bdosc",
    "TKSS": "tkss",
    "DP NCut (trivial)": "gram_ncut",
}
SLUG_TO_METHOD = {v: k for k, v in METHOD_SLUG.items()}

# Which k values each dataset reports (README §4.1)
K_VALUES = {"surveillance": (3, 5, 8), "ballet": (3, 5)}


def sigma_tag(sigma: float) -> str:
    """0.25 -> '0p25'. The filename convention used throughout the tree."""
    return f"{sigma:g}".replace(".", "p")
