"""Sequential clustering on ballet dataset.

Each sequence folder (seq_000001, ...) is one temporal cluster. All
sequences are shuffled (seed=SEED). The last N_TEST sequences are the
test pool; the rest are the train pool. Within each pool we sample many
distinct K-sequence combinations to form clustering examples Y, so the
same sequence can appear in multiple matrices but train and test never mix.

Per sequence, n ~ Unif{N_FRAMES_LO, ..., N_FRAMES_HI} and a start index
are sampled once; the n consecutive frames from that start are kept and
reused in every combination that includes that sequence.

Each frame is downsampled to DOWN_HW x DOWN_HW (default 30) and vectorized.
All methods see the same Y.

Optuna tunes on pooled train matrices only (clean / hetero modes). Train
and test are then evaluated. Homogeneous --sigmas tunes a separate param
set per σ on that level's train matrices only, then evaluates train+test
at each σ.

Methods: OSC, TKSS, SSC-TV-L21, SSC-TV-L21-col, BDOSC (fixed params),
Gram-NCut (Gram of Y, eigengap, DP NCut; no tuning).
"""

from pathlib import Path
import argparse
import csv
import json
import time
import sys
from math import comb
import numpy as np
import optuna
from optuna.samplers import TPESampler
from PIL import Image

# Add surveillance_dataset to path for imports
HERE = Path(__file__).resolve().parent
SURVEILLANCE_DIR = HERE.parent / "surveillance_dataset"
sys.path.insert(0, str(SURVEILLANCE_DIR))

from bdosc import bd_qosc
from l21_ssc_tv import ssc_admm_col_tv, ssc_admm_nuc_tv
from osc import osc_exact, cluster_from_Z
from ssc_block_tv import ssc_admm_block_tv
from ssc_tv import cluster_from_C, estimate_k_from_data
from tkss import tkss_cluster

FRAMES_DIR = HERE / "frames_tracked"
CSV_PATH = HERE / "ballet_cluster_khat_scaled_results.csv"
PARAMS_PATH = HERE / "ballet_cluster_khat_scaled_params.json"
HETERO_CSV_PATH = HERE / "ballet_cluster_hetero_results.csv"
HETERO_PARAMS_PATH = HERE / "ballet_cluster_hetero_params.json"
SIGMAS_CSV_PATH = HERE / "ballet_cluster_sigmas_results.csv"
SIGMAS_PARAMS_PATH = HERE / "ballet_cluster_sigmas_params.json"

N_FRAMES_LO = 10
N_FRAMES_HI = 100
N_TRIALS = 100
SIGMAS_N_TRIALS = 10
SEED = 0
SIGMAS = (0.0, 0.25, 0.5, 0.75)
K_DEFAULT = 5
N_TEST = 15
N_TRAIN_COMBOS = 24
N_TEST_COMBOS = 18
DOWN_HW = 30

# OSC-like start: no sparse error, no row TV; column TV matches OSC λ₂=0.1.
# lambda_e and gamma_p cannot be 0 on a log scale, so they sit at the search floor.
SSC_DEFAULTS = dict(lambda_e=0.01, lambda_z=0.1, gamma_p=0.001, gamma_q=0.1)
SSC_COL_DEFAULTS = dict(lambda_e=1.0, lambda_z=0.1, gamma_q=0.1)
# Block length k for Db and E's groups; not searched.
SSC_BLOCK_SIZE_FIXED = 5
SSC_BLOCK_DEFAULTS = dict(lambda_e=1.0, lambda_z=0.1, gamma_q=0.1)
OSC_DEFAULTS = dict(lambda_1=0.1, lambda_2=0.1)
BDOSC_DEFAULTS = dict(lambda_1=0.2, lambda_2=1.0, gamma_1=0.01, p=1.1, max_iter=50)
TKSS_DEFAULTS = dict(d=5, lam=1.0, s=2)
GRAM_NCUT_DEFAULTS = dict()


def n_tune_trials(defaults, n_trials, ref=OSC_DEFAULTS):
    """Scale Optuna trials with the number of hyperparameters.

    ``n_trials`` is OSC's budget (2 params). SSC-TV-L21 (4 params) gets 2×.
    """
    return max(1, int(round(n_trials * len(defaults) / max(len(ref), 1))))

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_image(path, size=DOWN_HW):
    """Load grayscale image, downsample to size x size, return flattened."""
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('L')
    img = img.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float64)
    return arr.reshape(-1)


def sequence_dirs(frames_dir=FRAMES_DIR):
    return sorted(p for p in Path(frames_dir).iterdir() if p.is_dir())


def split_pools(seq_dirs, n_test=N_TEST, seed=SEED):
    """Shuffle sequences; last n_test names are the test pool, the rest train."""
    names = np.asarray(sorted(p.name for p in seq_dirs))
    rng = np.random.default_rng(seed)
    names = names[rng.permutation(len(names))]
    if len(names) < n_test:
        raise ValueError(f"need at least n_test={n_test} sequences, got {len(names)}")
    test_pool = [str(x) for x in names[-n_test:]]
    train_pool = [str(x) for x in names[:-n_test]]
    return train_pool, test_pool


def sample_combos(pool, k, n_combos, rng):
    """Sample n_combos distinct unsorted-as-sorted k-subsets from pool."""
    pool = list(pool)
    if len(pool) < k:
        raise ValueError(f"pool size {len(pool)} < k={k}")
    n_all = comb(len(pool), k)
    n_combos = min(int(n_combos), n_all)
    seen = set()
    groups = []
    attempts = 0
    max_attempts = max(n_combos * 1000, 1000)
    while len(groups) < n_combos:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"could not sample {n_combos} distinct {k}-subsets from "
                f"{len(pool)} items after {attempts} draws"
            )
        idx = rng.choice(len(pool), size=k, replace=False)
        g = tuple(sorted(pool[i] for i in idx))
        if g in seen:
            continue
        seen.add(g)
        groups.append(list(g))
    return groups


def load_sequence(seq_dir, rng):
    """Keep n consecutive frames from a random start in one sequence.

    n ~ Unif{N_FRAMES_LO, ..., N_FRAMES_HI}, capped at the folder length.
    """
    frames = sorted(seq_dir.glob("*.jpg"))
    n = int(rng.integers(N_FRAMES_LO, N_FRAMES_HI + 1))
    n = min(n, len(frames))
    start = int(rng.integers(0, len(frames) - n + 1))
    frames = frames[start:start + n]
    images = [load_image(p) for p in frames]
    return np.stack(images, axis=1), frames


def load_group(seq_dirs, rng):
    """Load one concatenated matrix from a list of sequence directories."""
    images, labels, paths, n_kept = [], [], [], []
    for seq_idx, seq_dir in enumerate(seq_dirs):
        Yi, frames = load_sequence(seq_dir, rng)
        n_kept.append(Yi.shape[1])
        images.append(Yi)
        labels.append(np.full(Yi.shape[1], seq_idx, dtype=int))
        paths.extend(frames)
    Y = np.concatenate(images, axis=1)
    labels = np.concatenate(labels)
    return Y, labels, list(seq_dirs), paths, n_kept


def concat_group(tracks, names):
    """Build Y from pre-sampled per-sequence tracks. ``tracks[name]`` is Y01."""
    cols, labs = [], []
    n_kept = []
    for i, name in enumerate(names):
        Yi = tracks[name]
        cols.append(Yi)
        labs.append(np.full(Yi.shape[1], i, dtype=int))
        n_kept.append(Yi.shape[1])
    Y = np.concatenate(cols, axis=1)
    labels = np.concatenate(labs)
    return Y, labels, n_kept


def load_ballet_sequences(frames_dir=FRAMES_DIR, k=K_DEFAULT, rng=None):
    """Load the first k sequences (visualization / backward compatible)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    dirs = sequence_dirs(frames_dir)[:k]
    return load_group(dirs, rng)


def column_normalize(Y):
    norms = np.linalg.norm(Y, axis=0, keepdims=True)
    return Y / np.maximum(norms, 1e-12)


def contingency(y_true, y_pred):
    _, yt = np.unique(y_true, return_inverse=True)
    _, yp = np.unique(y_pred, return_inverse=True)
    table = np.zeros((yt.max() + 1, yp.max() + 1), dtype=np.int64)
    np.add.at(table, (yt, yp), 1)
    return table


def adjusted_rand_score(y_true, y_pred):
    table = contingency(y_true, y_pred)
    n = len(y_true)
    sum_comb = np.sum(table * (table - 1)) / 2.0
    sum_rows = np.sum(table.sum(axis=1) * (table.sum(axis=1) - 1)) / 2.0
    sum_cols = np.sum(table.sum(axis=0) * (table.sum(axis=0) - 1)) / 2.0
    n_comb = n * (n - 1) / 2.0
    expected = sum_rows * sum_cols / n_comb
    max_index = 0.5 * (sum_rows + sum_cols)
    if max_index == expected:
        return 1.0
    return (sum_comb - expected) / (max_index - expected)


def normalized_mutual_info(y_true, y_pred):
    table = contingency(y_true, y_pred).astype(np.float64)
    n = table.sum()
    pi = table / n
    py = pi.sum(axis=1, keepdims=True)
    pk = pi.sum(axis=0, keepdims=True)
    nz = pi > 0
    mi = np.sum(pi[nz] * np.log(pi[nz] / (py * pk)[nz]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    hk = -np.sum(pk[pk > 0] * np.log(pk[pk > 0]))
    return float(2.0 * mi / (hy + hk)) if (hy + hk) else 1.0


def _hungarian(cost):
    """Min-cost assignment. Extra rows or columns may stay unmatched."""
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0 or min(cost.shape) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    n, m = cost.shape
    if n > m:
        cols, rows = _hungarian(cost.T)
        return rows, cols
    a = np.zeros((n + 1, m + 1))
    a[1:, 1:] = cost
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        found = False
        for _ in range(m + 1):
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0, j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            if j1 == 0:
                break
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                found = True
                break
        if not found:
            break
        for _ in range(m + 1):
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    col_of_row = np.full(n, -1, dtype=int)
    for j in range(1, m + 1):
        if p[j] != 0:
            col_of_row[p[j] - 1] = j - 1
    rows = np.where(col_of_row >= 0)[0]
    return rows, col_of_row[rows]


def clustering_accuracy(y_true, y_pred):
    table = contingency(y_true, y_pred)
    rows, cols = _hungarian(-table.astype(float))
    return float(table[rows, cols].sum() / table.sum())


def metrics(y_true, y_pred):
    return {
        "acc": clustering_accuracy(y_true, y_pred),
        "nmi": normalized_mutual_info(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
    }


def run_ssc_tv(Y, lambda_e, lambda_z, gamma_p, gamma_q, k=None):
    _, C, _ = ssc_admm_nuc_tv(
        Y, lambda_e=lambda_e, lambda_z=lambda_z, gamma_p=gamma_p, gamma_q=gamma_q, max_iter=50,
    )
    return cluster_from_C(C, k=k)


def run_ssc_tv_col(Y, lambda_e, lambda_z, gamma_q, k=None):
    _, C, _ = ssc_admm_col_tv(
        Y, lambda_e=lambda_e, lambda_z=lambda_z, gamma_q=gamma_q, max_iter=50,
    )
    return cluster_from_C(C, k=k)


def run_osc(Y, lambda_1, lambda_2, k=None):
    Z = osc_exact(Y, lambda_1, lambda_2, max_iter=50)
    return cluster_from_Z(Z, k=k)


def run_bdosc(Y, lambda_1, lambda_2, gamma_1, p, max_iter=50, k=None):
    if k is None:
        k = estimate_k_from_data(Y)
    Z, _, _ = bd_qosc(
        Y, k, lambda_1, lambda_2, gamma_1, p,
        max_iter=max_iter, diagconstraint=True,
    )
    return cluster_from_Z(Z, k=k)


def run_tkss(Y, d, lam, s, k=None):
    pred, _ = tkss_cluster(Y, k=k, d=d, lam=lam, s=s, max_iter=30, random_state=SEED)
    return pred


def run_gram_ncut(Y, k=None):
    """Eigengap on the Gram YᵀY, then contiguous DP NCut on that Gram."""
    G = Y.T @ Y
    if k is None:
        k = estimate_k_from_data(Y)
    return cluster_from_C(G, k=k)


def suggest_ssc(trial):
    # Full 4D search; ranges from good trials (ARI > 0.74 across 100 trials).
    # No single parameter sits in a tight enough band to justify fixing it —
    # the landscape has two modes:
    #   (a) near-zero regularization (trial 38, best: ARI=0.776, all params ~0.005)
    #   (b) moderate lambda_z/gamma_p with tiny gamma_q (trials 73,83,85: ARI~0.76)
    # gamma_q < 0.15 in every top trial except one; range tightened from
    # [0.001, 10] to [0.001, 1.0] to concentrate samples in the productive zone.
    # lambda_z and gamma_p cover near-zero through mode (b) on log scale.
    return dict(
        lambda_e=trial.suggest_float("lambda_e", 0.01, 10.0, log=True),
        lambda_z=trial.suggest_float("lambda_z", 0.001, 1.0, log=True),
        gamma_p=trial.suggest_float("gamma_p", 0.001, 10.0, log=True),
        gamma_q=trial.suggest_float("gamma_q", 0.001, 1.0, log=True),
    )


def suggest_ssc_col(trial):
    return dict(
        lambda_e=trial.suggest_float("lambda_e", 1e-2, 10.0, log=True),
        lambda_z=trial.suggest_float("lambda_z", 1e-3, 10.0, log=True),
        gamma_q=trial.suggest_float("gamma_q", 1e-3, 10.0, log=True),
    )


def suggest_osc(trial):
    return dict(
        lambda_1=trial.suggest_float("lambda_1", 1e-3, 10.0, log=True),
        lambda_2=trial.suggest_float("lambda_2", 1e-3, 10.0, log=True),
    )


def suggest_tkss(trial):
    return dict(
        d=trial.suggest_int("d", 1, 15),
        lam=trial.suggest_float("lam", 1e-2, 10.0, log=True),
        s=trial.suggest_int("s", 1, 6),
    )


def apply_noise(Y01, sigma, rng):
    """Add Gaussian noise. ``sigma`` is a scalar or a length-n per-column vector."""
    noise = rng.standard_normal(Y01.shape)
    scale = np.asarray(sigma, dtype=float)
    if scale.ndim == 0:
        Y = Y01 + scale * noise
    else:
        Y = Y01 + noise * scale.reshape(1, -1)
    return column_normalize(Y)


def apply_hetero_noise(Y01, rng, sigma_lo=0.0, sigma_hi=1.0, label=""):
    """Each column independently draws σ ~ Unif[sigma_lo, sigma_hi]."""
    sigma_j = rng.uniform(sigma_lo, sigma_hi, size=Y01.shape[1])
    Y = apply_noise(Y01, sigma_j, rng)
    qs = np.quantile(sigma_j, [0.25, 0.5, 0.75])
    prefix = f"{label} " if label else ""
    print(
        f"{prefix}hetero noise over {sigma_j.size} frames:  "
        f"σ ~ Unif[{sigma_lo:g}, {sigma_hi:g}]  "
        f"mean={sigma_j.mean():.3f}  std={sigma_j.std():.3f}  "
        f"min={sigma_j.min():.3f}  max={sigma_j.max():.3f}  "
        f"q25={qs[0]:.3f}  q50={qs[1]:.3f}  q75={qs[2]:.3f}"
    )
    return Y


def _fmt_params(params):
    parts = []
    for key, val in params.items():
        if isinstance(val, float):
            parts.append(f"{key}={val:.4g}")
        else:
            parts.append(f"{key}={val}")
    return ", ".join(parts)


def tune_over(name, suggest, run, mats, n_trials=N_TRIALS, enqueue=None, known_k=False):
    """Maximize mean ARI over a list of (Y, y_true, k)."""

    def objective(trial):
        params = suggest(trial)
        t0 = time.perf_counter()
        aris = []
        last_scores = None
        for Y, y_true, kk in mats:
            try:
                pred = run(Y, k=(kk if known_k else None), **params)
            except Exception as exc:
                print(f"  {name} trial {trial.number} failed: {exc}")
                return -1.0
            last_scores = metrics(y_true, pred)
            aris.append(last_scores["ari"])
        elapsed = time.perf_counter() - t0
        mean_ari = float(np.mean(aris))
        if len(aris) == 1:
            print(
                f"  {name} trial {trial.number}: ARI={last_scores['ari']:.4f}  "
                f"ACC={last_scores['acc']:.4f}  NMI={last_scores['nmi']:.4f}  "
                f"{elapsed:.1f}s  {_fmt_params(params)}"
            )
        else:
            print(
                f"  {name} trial {trial.number}: mean ARI={mean_ari:.4f}  "
                f"n={len(aris)}  {elapsed:.1f}s  {_fmt_params(params)}"
            )
        return mean_ari

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if enqueue:
        study.enqueue_trial(enqueue)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    params = dict(study.best_params)
    return {
        "params": params,
        "best_ari": float(study.best_value),
        "tune_s": time.perf_counter() - t0,
    }


def eval_method(name, pred_fn, y_true):
    t0 = time.perf_counter()
    pred = pred_fn()
    elapsed = time.perf_counter() - t0
    scores = metrics(y_true, pred)
    scores["k_pred"] = int(len(np.unique(pred)))
    print(
        f"  {name}: k_hat={scores['k_pred']}  ACC={scores['acc']:.4f}  "
        f"NMI={scores['nmi']:.4f}  ARI={scores['ari']:.4f}  {elapsed:.1f}s"
    )
    return scores, elapsed


def _jsonable(params):
    out = {}
    for key, val in params.items():
        if isinstance(val, (np.floating, float)):
            out[key] = float(val)
        elif isinstance(val, (np.integer, int)):
            out[key] = int(val)
        else:
            out[key] = val
    return out


def write_params(tuned, path):
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text())
    for name, result in tuned.items():
        existing[name] = {
            "params": _jsonable(result["params"]),
            "best_ari": result["best_ari"],
            "tune_s": result["tune_s"],
        }
    path.write_text(json.dumps(existing, indent=2))
    print(f"  wrote {path}")


def write_sigma_params(tuned_by_sigma, path):
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text())
    for sigma, methods in tuned_by_sigma.items():
        key = str(sigma)
        block = existing.setdefault(key, {})
        for name, result in methods.items():
            block[name] = {
                "params": _jsonable(result["params"]),
                "best_ari": result["best_ari"],
                "tune_s": result["tune_s"],
            }
    path.write_text(json.dumps(existing, indent=2))
    print(f"  wrote {path}")


def print_means_from_csv(path):
    from collections import defaultdict
    agg = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "")
            sigma = row.get("sigma", row.get("noise_type", ""))
            agg[(row["method"], split, sigma)].append({
                "acc": float(row["acc"]),
                "nmi": float(row["nmi"]),
                "ari": float(row["ari"]),
            })
    print("\n=== mean over matrices ===")
    for (method, split, sigma), scores_list in agg.items():
        acc = np.mean([s["acc"] for s in scores_list])
        nmi = np.mean([s["nmi"] for s in scores_list])
        ari = np.mean([s["ari"] for s in scores_list])
        print(
            f"  {method} {split} σ={sigma}: "
            f"ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  n={len(scores_list)}"
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--k", type=int, default=K_DEFAULT,
        help="Number of sequences per clustering example (default: 5)",
    )
    p.add_argument(
        "--n-test", type=int, default=N_TEST,
        help="Number of sequences held out for test (default: 15)",
    )
    p.add_argument(
        "--n-train-combos", type=int, default=N_TRAIN_COMBOS,
        help="Distinct k-sequence combinations sampled from the train pool",
    )
    p.add_argument(
        "--n-test-combos", type=int, default=N_TEST_COMBOS,
        help="Distinct k-sequence combinations sampled from the test pool",
    )
    p.add_argument(
        "--n-trials", type=int, default=N_TRIALS,
        help="Optuna trials for OSC (2 params). Other methods scale by "
             "n_params/2, so SSC-TV-L21 (4 params) gets 2× this value.",
    )
    p.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to run. Default: OSC TKSS SSC-TV-L21 BDOSC Gram-NCut",
    )
    p.add_argument(
        "--append", action="store_true",
        help="Append eval rows to the existing CSV instead of overwriting",
    )
    p.add_argument(
        "--hetero-noise", action="store_true",
        help="Add heterogeneous noise: per-column σ ~ Unif[0, 1]",
    )
    p.add_argument(
        "--sigmas", nargs="+", type=float, default=None,
        help="Homogeneous noise levels. Tune a separate param set per σ on "
             "that level's train matrices, then evaluate train+test at each σ. "
             f"Default {SIGMAS_N_TRIALS} trials per σ unless --n-trials is set. "
             "Example: --sigmas 0 0.25 0.5 0.75",
    )
    p.add_argument(
        "--no-tune", action="store_true",
        help="Skip Optuna; load hyperparameters from the params JSON",
    )
    p.add_argument(
        "--known-k", action="store_true",
        help="Use the true number of sequences as k (skip eigengap)",
    )
    p.add_argument(
        "--out-tag", default=None,
        help="Append _{tag} to the CSV/JSON stems so this run does not "
             "overwrite the default result files.",
    )
    p.add_argument(
        "--full-data", action="store_true",
        help="Concatenate every sequence into one matrix (no train/test split). "
             "Tune and eval in-sample. --k is ignored.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    k = args.k
    n_test = args.n_test
    n_train_combos = args.n_train_combos
    n_test_combos = args.n_test_combos
    if args.hetero_noise and args.sigmas is not None:
        raise ValueError("use either --hetero-noise or --sigmas, not both")
    sigmas = tuple(args.sigmas) if args.sigmas is not None else None
    n_trials = args.n_trials
    if sigmas is not None and n_trials == N_TRIALS:
        n_trials = SIGMAS_N_TRIALS

    if args.hetero_noise:
        csv_path = HETERO_CSV_PATH
        params_path = HETERO_PARAMS_PATH
    elif sigmas is not None:
        # One job per σ → separate files so parallel submits do not collide.
        if len(sigmas) == 1:
            tag = f"{sigmas[0]:g}"
            csv_path = HERE / f"ballet_cluster_sigmas_{tag}_results.csv"
            params_path = HERE / f"ballet_cluster_sigmas_{tag}_params.json"
        else:
            csv_path = SIGMAS_CSV_PATH
            params_path = SIGMAS_PARAMS_PATH
    else:
        csv_path = CSV_PATH
        params_path = PARAMS_PATH
    if args.known_k:
        csv_path = csv_path.with_name(csv_path.name.replace("khat", "knownk"))
        params_path = params_path.with_name(params_path.name.replace("khat", "knownk"))
        if csv_path == CSV_PATH:
            csv_path = HERE / "ballet_cluster_knownk_scaled_results.csv"
            params_path = HERE / "ballet_cluster_knownk_scaled_params.json"
    if args.out_tag:
        csv_path = csv_path.with_name(f"{csv_path.stem}_{args.out_tag}{csv_path.suffix}")
        params_path = params_path.with_name(
            f"{params_path.stem}_{args.out_tag}{params_path.suffix}"
        )

    print("=== Ballet Clustering Experiment ===")
    print(
        f"k={k}  n_test={n_test}  n_train_combos={n_train_combos}  "
        f"n_test_combos={n_test_combos}  n_trials={n_trials}  "
        f"down_hw={DOWN_HW}  full_data={args.full_data}  "
        f"n_frames ~ Unif[{N_FRAMES_LO}, {N_FRAMES_HI}]  seed={SEED}"
    )
    print(
        f"hetero_noise={args.hetero_noise}  sigmas={sigmas}  "
        f"append={args.append}  no_tune={args.no_tune}  "
        f"known_k={args.known_k}  csv={csv_path.name}"
    )

    all_dirs = sequence_dirs()
    name_to_dir = {p.name: p for p in all_dirs}
    if args.full_data:
        train_groups = [sorted(name_to_dir)]
        test_groups = []
        print(
            f"full-data: {len(all_dirs)} sequences in one matrix  "
            f"k={len(train_groups[0])}"
        )
        print(f"sequences={train_groups[0]}")
    else:
        train_pool, test_pool = split_pools(all_dirs, n_test=n_test, seed=SEED)
        combo_rng = np.random.default_rng(SEED)
        train_groups = sample_combos(train_pool, k, n_train_combos, combo_rng)
        if test_pool and len(test_pool) >= k and n_test_combos > 0:
            test_groups = sample_combos(test_pool, k, n_test_combos, combo_rng)
        else:
            test_groups = []
        print(
            f"sequences={len(all_dirs)}  train_pool={len(train_pool)}  "
            f"test_pool={len(test_pool)}  "
            f"train_combos={len(train_groups)}  test_combos={len(test_groups)}"
        )
        print(f"train pool: {train_pool}")
        print(f"test  pool: {test_pool}")
        for i, names in enumerate(train_groups):
            print(f"train {i} k={len(names)}  sequences={names}")
        for i, names in enumerate(test_groups):
            print(f"test  {i} k={len(names)}  sequences={names}")

    frame_rng = np.random.default_rng(SEED + 1)
    used_names = sorted({n for g in train_groups + test_groups for n in g})
    print(f"\nLoading {len(used_names)} sequences once (reused across combinations)")
    tracks = {}
    for name in used_names:
        Y_raw, _ = load_sequence(name_to_dir[name], frame_rng)
        tracks[name] = Y_raw / 255.0
        print(f"  {name}: frames={Y_raw.shape[1]}  dim={Y_raw.shape[0]}")

    def build_loaded(groups, tag):
        loaded = []
        for i, names in enumerate(groups):
            Y01, labels, n_kept = concat_group(tracks, names)
            print(
                f"  {tag}[{i}]: frames={Y01.shape[1]}  dim={Y01.shape[0]}  "
                f"n_kept={n_kept}  seqs={names}"
            )
            loaded.append((i, Y01, labels, names, n_kept))
        return loaded

    print("\nBuilding train combinations")
    loaded_train = build_loaded(train_groups, "train")
    print("Building test combinations")
    loaded_test = build_loaded(test_groups, "test")

    def noisify(loaded, rng, sigma=None, hetero=False, tag=""):
        mats = []
        for i, Y01, labels, names, n_kept in loaded:
            if hetero:
                Y = apply_hetero_noise(
                    Y01, rng, label=f"{tag}[{i}]",
                )
            elif sigma is not None:
                Y = apply_noise(Y01, sigma, rng)
            else:
                Y = column_normalize(Y01)
            mats.append((i, Y, labels, names))
        return mats

    noise_rng = np.random.default_rng(SEED)
    by_sigma = {}
    if args.hetero_noise:
        train_mats = noisify(loaded_train, noise_rng, hetero=True, tag="train")
        test_mats = noisify(loaded_test, noise_rng, hetero=True, tag="test")
        by_sigma["mixed"] = (train_mats, test_mats)
    elif sigmas is not None:
        for sigma in sigmas:
            sigma_rng = np.random.default_rng(SEED)
            train_mats = noisify(loaded_train, sigma_rng, sigma=sigma)
            test_mats = noisify(loaded_test, sigma_rng, sigma=sigma)
            by_sigma[sigma] = (train_mats, test_mats)
            print(
                f"σ={sigma}: train mats={len(train_mats)}  test mats={len(test_mats)}"
            )
    else:
        train_mats = noisify(loaded_train, noise_rng)
        test_mats = noisify(loaded_test, noise_rng)
        by_sigma["clean"] = (train_mats, test_mats)

    sample_Y = next(iter(by_sigma.values()))[0][0][1]
    print(f"clustering Y dim={sample_Y.shape[0]}  (downsampled {DOWN_HW}x{DOWN_HW})")

    run_fns = {
        "OSC": run_osc,
        "SSC-TV-L21": run_ssc_tv,
        "SSC-TV-L21-col": run_ssc_tv_col,
        "BDOSC": run_bdosc,
        "TKSS": run_tkss,
        "Gram-NCut": run_gram_ncut,
    }
    tune_specs = {
        "OSC": (suggest_osc, run_osc, OSC_DEFAULTS),
        "SSC-TV-L21": (suggest_ssc, run_ssc_tv, SSC_DEFAULTS),
        "SSC-TV-L21-col": (suggest_ssc_col, run_ssc_tv_col, SSC_COL_DEFAULTS),
        "TKSS": (suggest_tkss, run_tkss, TKSS_DEFAULTS),
    }
    fixed_defaults = {"BDOSC": BDOSC_DEFAULTS, "Gram-NCut": GRAM_NCUT_DEFAULTS}

    if args.methods:
        selected = args.methods
    elif sigmas is not None:
        selected = ["OSC", "TKSS", "SSC-TV-L21", "SSC-TV-L21-col", "BDOSC", "Gram-NCut"]
    else:
        selected = ["OSC", "TKSS", "SSC-TV-L21", "BDOSC", "Gram-NCut"]

    unknown = [n for n in selected if n not in run_fns]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {list(run_fns)}")
    print(f"\nMethods: {selected}")

    tuned = {}
    tuned_by_sigma = {}
    saved = json.loads(params_path.read_text()) if params_path.exists() else {}

    def load_method_params(saved_block, name):
        if name in saved_block and saved_block[name].get("params"):
            return {
                "params": dict(saved_block[name]["params"]),
                "best_ari": saved_block[name].get("best_ari"),
                "tune_s": saved_block[name].get("tune_s") or 0.0,
            }
        if name in fixed_defaults:
            return {
                "params": dict(fixed_defaults[name]),
                "best_ari": None,
                "tune_s": 0.0,
            }
        return None

    if sigmas is not None:
        print(
            f"per-σ tuning: {len(sigmas)} levels × {n_trials} trials  "
            f"({len(next(iter(by_sigma.values()))[0])} train mats per σ)"
        )
        if args.no_tune:
            for sigma in sigmas:
                tuned_by_sigma[sigma] = {}
                sigma_saved = saved.get(str(sigma), {})
                for name in selected:
                    result = load_method_params(sigma_saved, name)
                    if result is None:
                        raise ValueError(
                            f"--no-tune requires saved params for {name} at "
                            f"σ={sigma} in {params_path}"
                        )
                    tuned_by_sigma[sigma][name] = result
                    print(
                        f"\nσ={sigma} {name}: "
                        f"loaded {_fmt_params(result['params'])}"
                    )
        else:
            for sigma in sigmas:
                tuned_by_sigma[sigma] = {}
                train_mats, _ = by_sigma[sigma]
                tune_mats = [
                    (Y, labels, len(names))
                    for _, Y, labels, names in train_mats
                ]
                for name in selected:
                    result = load_method_params({}, name)
                    if result is not None and name in fixed_defaults:
                        tuned_by_sigma[sigma][name] = result
                        print(
                            f"\nσ={sigma} {name}: fixed "
                            f"{_fmt_params(result['params'])}"
                        )
                        continue
                    if name not in tune_specs:
                        continue
                    suggest, run, defaults = tune_specs[name]
                    n_method = n_tune_trials(defaults, n_trials)
                    print(f"\n{'=' * 60}")
                    print(
                        f"--- Tuning {name} at σ={sigma} over "
                        f"{len(tune_mats)} train mats  "
                        f"n_trials={n_method} ({len(defaults)} params) ---"
                    )
                    result = tune_over(
                        name, suggest, run, tune_mats,
                        n_trials=n_method, enqueue=defaults,
                        known_k=args.known_k,
                    )
                    tuned_by_sigma[sigma][name] = result
                    print(
                        f"  best mean ARI={result['best_ari']:.4f}  "
                        f"tune {result['tune_s']:.1f}s  "
                        f"{_fmt_params(result['params'])}"
                    )
                    write_sigma_params(tuned_by_sigma, params_path)
    else:
        tune_mats = []
        for train_mats, _ in by_sigma.values():
            for _, Y, labels, names in train_mats:
                tune_mats.append((Y, labels, len(names)))
        print(f"pooled train mats for tuning={len(tune_mats)}")

        if args.no_tune:
            for name in selected:
                result = load_method_params(saved, name)
                if result is None:
                    raise ValueError(
                        f"--no-tune requires saved params for {name} in {params_path}"
                    )
                tuned[name] = result
                print(f"\n{name}: loaded {_fmt_params(tuned[name]['params'])}")
        else:
            for name in selected:
                result = load_method_params({}, name)
                if result is not None and name in fixed_defaults:
                    tuned[name] = result
                    print(f"\n{name}: fixed {_fmt_params(tuned[name]['params'])}")

            for name in selected:
                if name not in tune_specs:
                    continue
                suggest, run, defaults = tune_specs[name]
                n_method = n_tune_trials(defaults, n_trials)
                print(f"\n{'=' * 60}")
                print(
                    f"--- Tuning {name} over {len(tune_mats)} train mats  "
                    f"n_trials={n_method} ({len(defaults)} params) ---"
                )
                result = tune_over(
                    name, suggest, run, tune_mats,
                    n_trials=n_method, enqueue=defaults,
                    known_k=args.known_k,
                )
                tuned[name] = result
                print(
                    f"  best mean ARI={result['best_ari']:.4f}  "
                    f"tune {result['tune_s']:.1f}s  "
                    f"{_fmt_params(result['params'])}"
                )
                write_params(tuned, params_path)

    fieldnames = [
        "method", "split", "example", "k", "k_pred", "n_frames", "sequences",
        "noise_type", "sigma", "params", "acc", "nmi", "ari", "seconds",
    ]
    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    mode = "a" if args.append and csv_exists else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        f.flush()

        for sigma, (train_mats, test_mats) in by_sigma.items():
            if args.hetero_noise:
                noise_type = "hetero"
            elif sigmas is not None:
                noise_type = "homogeneous"
            else:
                noise_type = "clean"
            for split_name, mats in (("train", train_mats), ("test", test_mats)):
                for ex_idx, Y, labels, names in mats:
                    print(
                        f"\n=== {split_name}[{ex_idx}] k={len(names)}  "
                        f"dim={Y.shape[0]}  frames={Y.shape[1]}  σ={sigma}  "
                        f"seqs={names} ==="
                    )
                    for name in selected:
                        run = run_fns[name]
                        if sigmas is not None:
                            params = tuned_by_sigma[sigma][name]["params"]
                        else:
                            params = tuned[name]["params"]
                        scores, elapsed = eval_method(
                            name,
                            lambda run=run, Y=Y, params=params, kk=len(names):
                                run(Y, k=(kk if args.known_k else None), **params),
                            labels,
                        )
                        writer.writerow({
                            "method": name,
                            "split": split_name,
                            "example": ex_idx,
                            "k": len(names),
                            "k_pred": scores["k_pred"],
                            "n_frames": Y.shape[1],
                            "sequences": " ".join(names),
                            "noise_type": noise_type,
                            "sigma": sigma,
                            "params": _fmt_params(params),
                            "acc": f"{scores['acc']:.6f}",
                            "nmi": f"{scores['nmi']:.6f}",
                            "ari": f"{scores['ari']:.6f}",
                            "seconds": f"{elapsed:.2f}",
                        })
                        f.flush()

    print_means_from_csv(csv_path)
    print(f"\nwrote {csv_path}")
    print(f"wrote {params_path}")


if __name__ == "__main__":
    main()
