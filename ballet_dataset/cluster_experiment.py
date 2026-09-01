"""Sequential clustering on ballet dataset.

Each sequence folder (seq_000001, seq_000002, ...) is treated as a separate
temporal sequence. We uniformly sample between 10-60 frames from each sequence,
keeping the images in temporal order.

The data matrix Y has columns organized by:
[seq_000001_img1, seq_000001_img10, ..., seq_000002_img3, seq_000002_img15, ...]

Given k sequences, we cluster to recover which frames belong to which sequence.

Two experiments:
1. Clean clustering (no noise)
2. Heterogeneous noise: each image column draws σ ~ Unif[0, 1]

Methods tested: OSC, TKSS, SSC-TV-L21, and BD-OSC (inference only, fixed params).
"""

from pathlib import Path
import argparse
import csv
import json
import time
import sys
import numpy as np
import optuna
from optuna.samplers import TPESampler
from PIL import Image

# Add surveillance_dataset to path for imports
HERE = Path(__file__).resolve().parent
SURVEILLANCE_DIR = HERE.parent / "surveillance_dataset"
sys.path.insert(0, str(SURVEILLANCE_DIR))

from bdosc import bd_qosc
from l21_ssc_tv import ssc_admm_nuc_tv
from osc import osc_exact, cluster_from_Z
from ssc_tv import cluster_from_C
from tkss import tkss

FRAMES_DIR = HERE / "frames_tracked"
CSV_PATH = HERE / "ballet_cluster_results.csv"
PARAMS_PATH = HERE / "ballet_cluster_params.json"
HETERO_CSV_PATH = HERE / "ballet_cluster_hetero_results.csv"
HETERO_PARAMS_PATH = HERE / "ballet_cluster_hetero_params.json"

N_FRAMES_LO = 10
N_FRAMES_HI = 60
N_TRIALS = 40
TKSS_PCA = 150
SEED = 0

SSC_DEFAULTS = dict(lambda_e=1.0, lambda_z=0.1, gamma_p=0.1, gamma_q=0.1)
OSC_DEFAULTS = dict(lambda_1=0.1, lambda_2=0.1)
BDOSC_DEFAULTS = dict(lambda_1=0.2, lambda_2=1.0, gamma_1=0.01, p=1.1, max_iter=50)
TKSS_DEFAULTS = dict(d=5, lam=1.0, s=2)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_image(path):
    """Load image and convert to grayscale if needed, return flattened."""
    img = Image.open(path)
    if img.mode != 'L':
        img = img.convert('L')
    arr = np.array(img, dtype=np.float64)
    return arr.reshape(-1)


def load_ballet_sequences(frames_dir=FRAMES_DIR, k=5, rng=None):
    """Load k ballet sequences with uniform sampling.
    
    For each sequence, sample n ~ Unif{N_FRAMES_LO, ..., N_FRAMES_HI} frames,
    keeping them in temporal order.
    
    Returns:
        Y: data matrix (d x total_frames)
        labels: sequence index for each column
        seq_dirs: list of selected sequence directories
        n_kept: list of number of frames kept per sequence
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    
    # Get all sequence directories
    seq_dirs = sorted([p for p in frames_dir.iterdir() if p.is_dir()])
    
    # Select k sequences
    if len(seq_dirs) < k:
        raise ValueError(f"Only {len(seq_dirs)} sequences available, but k={k} requested")
    
    # For now, just take the first k sequences (could randomize if desired)
    seq_dirs = seq_dirs[:k]
    
    images = []
    labels = []
    paths = []
    n_kept = []
    
    for seq_idx, seq_dir in enumerate(seq_dirs):
        # Get all frames in temporal order
        frames = sorted(seq_dir.glob("*.jpg"))
        
        # Sample n frames uniformly
        n = int(rng.integers(N_FRAMES_LO, N_FRAMES_HI + 1))
        n = min(n, len(frames))
        
        # Select n frames uniformly from available frames
        if n < len(frames):
            indices = np.sort(rng.choice(len(frames), size=n, replace=False))
            frames = [frames[i] for i in indices]
        else:
            frames = frames[:n]
        
        n_kept.append(n)
        
        # Load frames
        for frame_path in frames:
            img_vec = load_image(frame_path)
            images.append(img_vec)
            labels.append(seq_idx)
            paths.append(frame_path)
    
    Y = np.stack(images, axis=1)
    labels = np.asarray(labels, dtype=int)
    
    return Y, labels, seq_dirs, paths, n_kept


def column_normalize(Y):
    norms = np.linalg.norm(Y, axis=0, keepdims=True)
    return Y / np.maximum(norms, 1e-12)


def pca_reduce(Y, n_comp):
    Yc = Y - Y.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(Yc, full_matrices=False)
    n_comp = min(n_comp, U.shape[1])
    return U[:, :n_comp].T @ Yc


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


def run_ssc_tv(Y, k, lambda_e, lambda_z, gamma_p, gamma_q):
    X, _, _ = ssc_admm_nuc_tv(
        Y, lambda_e=lambda_e, lambda_z=lambda_z, gamma_p=gamma_p, gamma_q=gamma_q, max_iter=50,
    )
    return cluster_from_C(X, k=k)


def run_osc(Y, k, lambda_1, lambda_2):
    Z = osc_exact(Y, lambda_1, lambda_2, max_iter=50)
    return cluster_from_Z(Z, k=k)


def run_bdosc(Y, k, lambda_1, lambda_2, gamma_1, p, max_iter=50):
    Z, _, _ = bd_qosc(
        Y, k, lambda_1, lambda_2, gamma_1, p,
        max_iter=max_iter, diagconstraint=True,
    )
    return cluster_from_Z(Z, k=k)


def run_tkss(Y_pca, k, d, lam, s):
    _, pred = tkss(Y_pca, K=k, d=d, lam=lam, s=s, max_iter=30, random_state=SEED)
    return pred


def suggest_ssc(trial):
    return dict(
        lambda_e=trial.suggest_float("lambda_e", 1e-2, 10.0, log=True),
        lambda_z=trial.suggest_float("lambda_z", 1e-3, 10.0, log=True),
        gamma_p=trial.suggest_float("gamma_p", 1e-3, 10.0, log=True),
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


def apply_hetero_noise(Y01, rng, sigma_lo=0.0, sigma_hi=1.0):
    """Each column independently draws σ ~ Unif[sigma_lo, sigma_hi]."""
    sigma_j = rng.uniform(sigma_lo, sigma_hi, size=Y01.shape[1])
    Y = apply_noise(Y01, sigma_j, rng)
    
    qs = np.quantile(sigma_j, [0.25, 0.5, 0.75])
    print(
        f"hetero noise over {sigma_j.size} frames:  "
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


def tune_over(name, suggest, run, Y, labels, k, n_trials=N_TRIALS, enqueue=None):
    """Tune on a single dataset to maximize ARI."""
    def objective(trial):
        params = suggest(trial)
        t0 = time.perf_counter()
        try:
            pred = run(Y, k, **params)
            scores = metrics(labels, pred)
        except Exception as exc:
            print(f"  {name} trial {trial.number} failed: {exc}")
            return -1.0
        elapsed = time.perf_counter() - t0
        print(
            f"  {name} trial {trial.number}: ARI={scores['ari']:.4f}  "
            f"ACC={scores['acc']:.4f}  NMI={scores['nmi']:.4f}  "
            f"{elapsed:.1f}s  {_fmt_params(params)}"
        )
        return scores["ari"]

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if enqueue:
        study.enqueue_trial(enqueue)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    
    return {
        "params": dict(study.best_params),
        "best_ari": float(study.best_value),
        "tune_s": time.perf_counter() - t0,
    }


def eval_method(name, pred_fn, y_true):
    t0 = time.perf_counter()
    pred = pred_fn()
    elapsed = time.perf_counter() - t0
    scores = metrics(y_true, pred)
    print(
        f"  {name}: ACC={scores['acc']:.4f}  NMI={scores['nmi']:.4f}  "
        f"ARI={scores['ari']:.4f}  {elapsed:.1f}s"
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--k", type=int, default=5,
        help="Number of sequences to cluster (default: 5)",
    )
    p.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to run. Default: OSC TKSS SSC-TV-L21 BDOSC",
    )
    p.add_argument(
        "--hetero-noise", action="store_true",
        help="Add heterogeneous noise: per-column σ ~ Unif[0, 1]",
    )
    p.add_argument(
        "--no-tune", action="store_true",
        help="Skip Optuna; load hyperparameters from the params JSON",
    )
    return p.parse_args()


def main():
    args = parse_args()
    k = args.k
    
    if args.hetero_noise:
        csv_path = HETERO_CSV_PATH
        params_path = HETERO_PARAMS_PATH
    else:
        csv_path = CSV_PATH
        params_path = PARAMS_PATH
    
    print(f"=== Ballet Clustering Experiment ===")
    print(f"k={k}  n_frames ~ Unif[{N_FRAMES_LO}, {N_FRAMES_HI}]  seed={SEED}")
    print(f"hetero_noise={args.hetero_noise}  no_tune={args.no_tune}")
    
    # Load data
    rng = np.random.default_rng(SEED)
    Y_raw, labels, seq_dirs, paths, n_kept = load_ballet_sequences(k=k, rng=rng)
    
    print(f"\nLoaded {len(seq_dirs)} sequences:")
    for i, (seq_dir, n) in enumerate(zip(seq_dirs, n_kept)):
        print(f"  {seq_dir.name}: {n} frames")
    print(f"Total frames: {Y_raw.shape[1]}")
    print(f"Image dimension: {Y_raw.shape[0]}")
    
    # Normalize to [0, 1]
    Y01 = Y_raw / 255.0
    
    # Apply noise if requested
    if args.hetero_noise:
        Y = apply_hetero_noise(Y01, rng)
    else:
        Y = column_normalize(Y01)
    
    # Prepare for TKSS (PCA reduction)
    Y_pca = pca_reduce(Y, TKSS_PCA)
    
    # Define methods
    tune_specs = {
        "OSC": (suggest_osc, run_osc, OSC_DEFAULTS),
        "SSC-TV-L21": (suggest_ssc, run_ssc_tv, SSC_DEFAULTS),
        "TKSS": (suggest_tkss, run_tkss, TKSS_DEFAULTS),
    }
    eval_specs = {
        "OSC": (run_osc, Y),
        "SSC-TV-L21": (run_ssc_tv, Y),
        "BDOSC": (run_bdosc, Y),
        "TKSS": (run_tkss, Y_pca),
    }
    fixed_defaults = {"BDOSC": BDOSC_DEFAULTS}
    
    if args.methods:
        selected = args.methods
    else:
        selected = ["OSC", "TKSS", "SSC-TV-L21", "BDOSC"]
    
    unknown = [n for n in selected if n not in eval_specs]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {list(eval_specs)}")
    
    print(f"\nMethods: {selected}")
    
    # Tune hyperparameters
    tuned = {}
    saved = json.loads(params_path.read_text()) if params_path.exists() else {}
    
    if args.no_tune:
        for name in selected:
            if name in saved and saved[name].get("params"):
                tuned[name] = {
                    "params": dict(saved[name]["params"]),
                    "best_ari": saved[name].get("best_ari"),
                    "tune_s": saved[name].get("tune_s") or 0.0,
                }
                print(f"\n{name}: loaded {_fmt_params(tuned[name]['params'])}")
            elif name in fixed_defaults:
                tuned[name] = {
                    "params": dict(fixed_defaults[name]),
                    "best_ari": None,
                    "tune_s": 0.0,
                }
                print(f"\n{name}: fixed {_fmt_params(tuned[name]['params'])}")
            else:
                raise ValueError(
                    f"--no-tune requires saved params for {name} in {params_path}"
                )
    else:
        # Fixed methods
        for name in selected:
            if name in fixed_defaults:
                tuned[name] = {
                    "params": dict(fixed_defaults[name]),
                    "best_ari": None,
                    "tune_s": 0.0,
                }
                print(f"\n{name}: fixed {_fmt_params(tuned[name]['params'])}")
        
        # Tune others
        for name in selected:
            if name not in tune_specs:
                continue
            
            suggest, run, defaults = tune_specs[name]
            print(f"\n{'=' * 60}")
            print(f"--- Tuning {name} ---")
            
            Y_tune = Y_pca if name == "TKSS" else Y
            result = tune_over(name, suggest, run, Y_tune, labels, k, enqueue=defaults)
            tuned[name] = result
            print(
                f"  best ARI={result['best_ari']:.4f}  "
                f"tune {result['tune_s']:.1f}s  {_fmt_params(result['params'])}"
            )
            write_params(tuned, params_path)
    
    # Evaluate all methods
    print(f"\n{'=' * 60}")
    print("=== Evaluation ===")
    
    fieldnames = [
        "method", "k", "n_frames", "noise_type", 
        "params", "acc", "nmi", "ari", "seconds",
    ]
    
    mode = "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for name in selected:
            run, Y_eval = eval_specs[name]
            params = tuned[name]["params"]
            scores, elapsed = eval_method(
                name,
                lambda: run(Y_eval, k, **params),
                labels,
            )
            writer.writerow({
                "method": name,
                "k": k,
                "n_frames": Y.shape[1],
                "noise_type": "hetero" if args.hetero_noise else "clean",
                "params": _fmt_params(params),
                "acc": f"{scores['acc']:.6f}",
                "nmi": f"{scores['nmi']:.6f}",
                "ari": f"{scores['ari']:.6f}",
                "seconds": f"{elapsed:.2f}",
            })
            f.flush()
    
    print(f"\nwrote {csv_path}")
    print(f"wrote {params_path}")


if __name__ == "__main__":
    main()
