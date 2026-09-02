"""Sequential clustering on P1E and P1L.

Each sequence has 25 people. They are shuffled, then split into 5 groups of
K=5. The first 4 groups are train matrices; the last group is the held-out
test matrix. Train and test both cluster with k=5.

Per person, n is drawn uniformly from {N_FRAMES_LO, ..., N_FRAMES_HI}
(default 30–60) and the first min(n, available frames) images are kept.
People with fewer than N_FRAMES_LO frames keep all available frames.

Optuna tunes SSC-TV-L21, SSC-TV-L21-col, OSC, and TKSS once, maximizing
mean train ARI pooled over all sequences, train groups, and noise levels
in SIGMAS. BDOSC is not tuned; it uses a fixed parameter set. Those
hyperparameters are then evaluated on train and test at each noise level.

SSC-TV-L21-col is the L2,1 model with only column-wise TV (Q = CD^T);
the row-wise term γ_p ||DC||_{2,1} is dropped.

With --hetero-noise, each image column draws its own σ ~ Unif[0, 1]
instead of one σ per matrix. Tuning and eval use that single mixed-noise
draw; OSC is included.
"""

from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import time
import numpy as np
import optuna
from optuna.samplers import TPESampler

from bdosc import bd_qosc
from l21_ssc_tv import ssc_admm_col_tv, ssc_admm_nuc_tv
from osc import osc_exact, cluster_from_Z
from ssc_tv import cluster_from_C
from tkss import tkss

HERE = Path(__file__).resolve().parent
DATA_ROOTS = (HERE / "P1E", HERE / "P1L")
K = 5
N_TRIALS = 10
TKSS_PCA = 150
SEED = 0
SIGMAS = (0.0, 0.25, 0.5, 0.75)
N_FRAMES_LO = 30
N_FRAMES_HI = 60


def _result_paths(n_trials=N_TRIALS):
    tag = f"min{N_FRAMES_LO}_trials{n_trials}"
    return {
        "sigmas": (HERE / f"split_k5_sigmas_{tag}.csv", HERE / f"split_k5_sigmas_{tag}_params.json"),
        "clean": (HERE / f"split_k5_clean_{tag}.csv", HERE / f"split_k5_clean_{tag}_params.json"),
        "hetero": (HERE / f"split_k5_hetero_{tag}.csv", HERE / f"split_k5_hetero_{tag}_params.json"),
    }

SSC_DEFAULTS = dict(lambda_e=1.0, lambda_z=0.1, gamma_p=0.1, gamma_q=0.1)
SSC_COL_DEFAULTS = dict(lambda_e=1.0, lambda_z=0.1, gamma_q=0.1)
OSC_DEFAULTS = dict(lambda_1=0.1, lambda_2=0.1)
BDOSC_DEFAULTS = dict(lambda_1=0.2, lambda_2=1.0, gamma_1=0.01, p=1.1, max_iter=50)
TKSS_DEFAULTS = dict(d=5, lam=1.0, s=2)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def read_pgm(path):
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"{path}: expected P5 PGM, got {magic!r}")
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        width, height = map(int, line.split())
        maxval = int(f.readline())
        dtype = np.uint8 if maxval < 256 else np.uint16
        pixels = np.frombuffer(f.read(), dtype=dtype)
    if pixels.size != width * height:
        raise ValueError(f"{path}: expected {width * height} pixels, got {pixels.size}")
    return pixels.reshape(height, width)


def sequence_dirs(roots=DATA_ROOTS):
    seqs = []
    for root in roots:
        seqs.extend(sorted(p for p in root.iterdir() if p.is_dir()))
    return seqs


def load_sequence(root, rng=None):
    """Columns of Y are flattened frames, grouped by person id then frame number.

    For each person, n ~ Unif{N_FRAMES_LO, ..., N_FRAMES_HI}; keep the first
    min(n, available) frames so temporal order is preserved.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    person_dirs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    images, labels, paths = [], [], []
    n_kept = []
    for person_idx, person_dir in enumerate(person_dirs):
        frames = sorted(person_dir.glob("*.pgm"), key=lambda p: p.name)
        n = int(rng.integers(N_FRAMES_LO, N_FRAMES_HI + 1))
        n = min(n, len(frames))
        frames = frames[:n]
        n_kept.append(n)
        for frame_path in frames:
            images.append(read_pgm(frame_path).reshape(-1).astype(np.float64))
            labels.append(person_idx)
            paths.append(frame_path)
    Y = np.stack(images, axis=1)
    labels = np.asarray(labels, dtype=int)
    return Y, labels, person_dirs, paths, n_kept


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
    """Min-cost assignment. Extra rows or columns may stay unmatched.

    The e-maxx implementation needs n <= m (a free column for every row).
    If there are more rows than columns, transpose, solve, and swap back.
    """
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


def run_ssc_tv_col(Y, k, lambda_e, lambda_z, gamma_q):
    X, _, _ = ssc_admm_col_tv(
        Y, lambda_e=lambda_e, lambda_z=lambda_z, gamma_q=gamma_q, max_iter=50,
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


def suggest_bdosc(trial):
    return dict(
        lambda_1=trial.suggest_float("lambda_1", 1e-3, 10.0, log=True),
        lambda_2=trial.suggest_float("lambda_2", 1e-3, 10.0, log=True),
        gamma_1=trial.suggest_float("gamma_1", 1e-3, 10.0, log=True),
        p=trial.suggest_float("p", 1.01, 1.5),
    )


def suggest_tkss(trial):
    return dict(
        d=trial.suggest_int("d", 1, 15),
        lam=trial.suggest_float("lam", 1e-2, 10.0, log=True),
        s=trial.suggest_int("s", 1, 6),
    )


def chunk_people(names, k=K, seed=SEED):
    """Shuffle identities, split into groups of k. Last group is test."""
    names = np.asarray(sorted(names))
    rng = np.random.default_rng(seed)
    names = names[rng.permutation(len(names))]
    n_groups = len(names) // k
    names = names[: n_groups * k]
    groups = [sorted(g.tolist()) for g in names.reshape(n_groups, k)]
    return groups[:-1], groups[-1]


def subset_people(Y, labels, person_dirs, keep_names):
    name_to_old = {p.name: i for i, p in enumerate(person_dirs)}
    ids = [name_to_old[n] for n in keep_names]
    mask = np.isin(labels, ids)
    remap = {int(old): new for new, old in enumerate(ids)}
    y = Y[:, mask]
    lab = np.array([remap[int(x)] for x in labels[mask]], dtype=int)
    return y, lab, list(keep_names)


def _fmt_params(params):
    parts = []
    for key, val in params.items():
        if isinstance(val, float):
            parts.append(f"{key}={val:.4g}")
        else:
            parts.append(f"{key}={val}")
    return ", ".join(parts)


def tune(name, suggest, run, y_true, n_trials=N_TRIALS, enqueue=None):
    cache = {"ari": -np.inf, "pred": None, "params": None}

    def objective(trial):
        params = suggest(trial)
        t0 = time.perf_counter()
        try:
            pred = run(**params)
            scores = metrics(y_true, pred)
        except Exception as exc:
            print(f"  {name} trial {trial.number} failed: {exc}")
            return -1.0
        elapsed = time.perf_counter() - t0
        print(
            f"  {name} trial {trial.number}: ARI={scores['ari']:.4f}  "
            f"ACC={scores['acc']:.4f}  NMI={scores['nmi']:.4f}  "
            f"{elapsed:.1f}s  {_fmt_params(params)}"
        )
        if scores["ari"] > cache["ari"]:
            cache.update(ari=scores["ari"], pred=pred, params=params, scores=scores)
        return scores["ari"]

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if enqueue:
        study.enqueue_trial(enqueue)
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    cache["tune_s"] = time.perf_counter() - t0
    cache["best_ari"] = float(study.best_value)
    return cache


def tune_over(name, suggest, run, mats, n_trials=N_TRIALS, enqueue=None):
    """Maximize mean ARI over a list of (Y, y_true, k) train matrices."""

    def objective(trial):
        params = suggest(trial)
        t0 = time.perf_counter()
        aris = []
        for Y, y_true, k in mats:
            try:
                pred = run(Y, k, **params)
            except Exception as exc:
                print(f"  {name} trial {trial.number} failed: {exc}")
                return -1.0
            aris.append(metrics(y_true, pred)["ari"])
        elapsed = time.perf_counter() - t0
        mean_ari = float(np.mean(aris))
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


def load_all_sequences():
    rng = np.random.default_rng(SEED)
    loaded = []
    for seq in sequence_dirs():
        print(f"Loading {seq.name}")
        Y_raw, labels, person_dirs, paths, n_kept = load_sequence(seq, rng=rng)
        loaded.append((seq.name, Y_raw / 255.0, labels, person_dirs, paths))
        n_short = sum(n < N_FRAMES_LO for n in n_kept)
        print(
            f"  Y: {Y_raw.shape}  people={len(person_dirs)}  "
            f"frames/person={n_kept}  min={min(n_kept)}"
        )
        if n_short:
            print(
                f"  warning: {n_short} people have fewer than "
                f"{N_FRAMES_LO} available frames (kept all)"
            )
    return loaded


def apply_noise(Y01, sigma, rng):
    """Add Gaussian noise. ``sigma`` is a scalar or a length-n per-column vector."""
    noise = rng.standard_normal(Y01.shape)
    scale = np.asarray(sigma, dtype=float)
    if scale.ndim == 0:
        Y = Y01 + scale * noise
    else:
        Y = Y01 + noise * scale.reshape(1, -1)
    return column_normalize(Y)


def build_split_mats(loaded, train_groups, test_group, sigma, rng):
    train_mats, test_mats = [], []
    for ds_name, Y01, labels, person_dirs, paths in loaded:
        Y = apply_noise(Y01, sigma, rng)
        for i, names in enumerate(train_groups):
            Ys, labs, nms = subset_people(Y, labels, person_dirs, names)
            train_mats.append((ds_name, i, Ys, labs, nms))
        Ys, labs, nms = subset_people(Y, labels, person_dirs, test_group)
        test_mats.append((ds_name, 0, Ys, labs, nms))
    return train_mats, test_mats


def build_split_mats_hetero(
    loaded, train_groups, test_group, rng, sigma_lo=0.0, sigma_hi=1.0,
):
    """Each column independently draws σ ~ Unif[sigma_lo, sigma_hi]."""
    train_mats, test_mats = [], []
    all_sigmas = []
    for ds_name, Y01, labels, person_dirs, paths in loaded:
        sigma_j = rng.uniform(sigma_lo, sigma_hi, size=Y01.shape[1])
        Y = apply_noise(Y01, sigma_j, rng)
        all_sigmas.append(sigma_j)
        for i, names in enumerate(train_groups):
            Ys, labs, nms = subset_people(Y, labels, person_dirs, names)
            train_mats.append((ds_name, i, Ys, labs, nms))
        Ys, labs, nms = subset_people(Y, labels, person_dirs, test_group)
        test_mats.append((ds_name, 0, Ys, labs, nms))
    all_sigmas = np.concatenate(all_sigmas)
    qs = np.quantile(all_sigmas, [0.25, 0.5, 0.75])
    print(
        f"hetero mix over {all_sigmas.size} frames:  "
        f"σ ~ Unif[{sigma_lo:g}, {sigma_hi:g}]  "
        f"mean={all_sigmas.mean():.3f}  std={all_sigmas.std():.3f}  "
        f"min={all_sigmas.min():.3f}  max={all_sigmas.max():.3f}  "
        f"q25={qs[0]:.3f}  q50={qs[1]:.3f}  q75={qs[2]:.3f}"
    )
    return train_mats, test_mats


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to run. Default: SSC-TV-L21 SSC-TV-L21-col BDOSC TKSS "
             "(+ OSC when --hetero-noise)",
    )
    p.add_argument(
        "--append", action="store_true",
        help="Append eval rows to the existing CSV instead of overwriting",
    )
    p.add_argument(
        "--hetero-noise", action="store_true",
        help="Per-column σ ~ Unif[0, 1] (mixed clean/noisy frames)",
    )
    p.add_argument(
        "--sigmas", nargs="+", type=float, default=None,
        help="Noise levels to tune/evaluate (default: 0 0.25 0.5 0.75)",
    )
    p.add_argument(
        "--skip-tune", nargs="+", default=None,
        help="Skip Optuna for these methods; load params from the params JSON",
    )
    p.add_argument(
        "--no-tune", action="store_true",
        help="Skip Optuna; load hyperparameters from the params JSON",
    )
    p.add_argument(
        "--n-trials", type=int, default=N_TRIALS,
        help=f"Optuna trials per tuned method (default: {N_TRIALS})",
    )
    return p.parse_args()


def write_params(tuned, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def print_means_from_csv(path):
    agg = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            agg[(row["method"], row["split"], row["sigma"])].append({
                "acc": float(row["acc"]),
                "nmi": float(row["nmi"]),
                "ari": float(row["ari"]),
            })
    print(f"\n=== mean over matrices ===")
    for (method, split_name, sigma), scores_list in agg.items():
        acc = np.mean([s["acc"] for s in scores_list])
        nmi = np.mean([s["nmi"] for s in scores_list])
        ari = np.mean([s["ari"] for s in scores_list])
        print(
            f"  {method} {split_name} σ={sigma}: "
            f"ACC={acc:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}  n={len(scores_list)}"
        )


def main():
    args = parse_args()
    sigmas = tuple(args.sigmas) if args.sigmas is not None else SIGMAS
    n_trials = args.n_trials
    paths = _result_paths(n_trials)
    if args.hetero_noise:
        csv_path, params_path = paths["hetero"]
    elif sigmas == (0.0,):
        csv_path, params_path = paths["clean"]
    else:
        csv_path, params_path = paths["sigmas"]
    loaded = load_all_sequences()
    people = sorted(p.name for p in loaded[0][3])
    train_groups, test_group = chunk_people(people)
    print(f"sequences={len(loaded)}  people={len(people)}  k={K}")
    print(f"frames/person ~ Unif[{N_FRAMES_LO}, {N_FRAMES_HI}]")
    print(f"sigmas={sigmas}  n_trials={n_trials}  csv={csv_path.name}")
    if args.hetero_noise:
        print("hetero-noise: per-column σ ~ Unif[0, 1]")
    for i, names in enumerate(train_groups):
        print(f"train {i} k={len(names)}  people={names}")
    print(f"test    k={len(test_group)}  people={test_group}")

    tune_specs = {
        "OSC": (suggest_osc, run_osc, OSC_DEFAULTS),
        "SSC-TV-L21": (suggest_ssc, run_ssc_tv, SSC_DEFAULTS),
        "SSC-TV-L21-col": (suggest_ssc_col, run_ssc_tv_col, SSC_COL_DEFAULTS),
        "TKSS": (suggest_tkss, run_tkss, TKSS_DEFAULTS),
    }
    eval_specs = {
        "OSC": run_osc,
        "SSC-TV-L21": run_ssc_tv,
        "SSC-TV-L21-col": run_ssc_tv_col,
        "BDOSC": run_bdosc,
        "TKSS": run_tkss,
    }
    fixed_defaults = {"BDOSC": BDOSC_DEFAULTS}
    if args.methods:
        selected = args.methods
    elif args.hetero_noise:
        selected = ["SSC-TV-L21", "SSC-TV-L21-col", "OSC", "BDOSC", "TKSS"]
    else:
        selected = ["SSC-TV-L21", "SSC-TV-L21-col", "BDOSC", "TKSS"]
    unknown = [n for n in selected if n not in eval_specs]
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {list(eval_specs)}")
    print(
        f"methods={selected}  append={args.append}  "
        f"hetero_noise={args.hetero_noise}  no_tune={args.no_tune}"
    )

    fieldnames = [
        "method", "dataset", "split", "example", "sigma", "k", "n_people",
        "n_frames", "people", "params", "acc", "nmi", "ari", "seconds",
    ]

    by_sigma = {}
    tune_mats = []
    if args.hetero_noise:
        rng = np.random.default_rng(SEED)
        train_mats, test_mats = build_split_mats_hetero(
            loaded, train_groups, test_group, rng,
        )
        by_sigma["mixed"] = (train_mats, test_mats)
        for _, _, Ys, labs, nms in train_mats:
            tune_mats.append((Ys, labs, len(nms)))
        print(
            f"σ=mixed: train mats={len(train_mats)}  "
            f"test mats={len(test_mats)}"
        )
    else:
        for sigma in sigmas:
            rng = np.random.default_rng(SEED)
            train_mats, test_mats = build_split_mats(
                loaded, train_groups, test_group, sigma, rng,
            )
            by_sigma[sigma] = (train_mats, test_mats)
            for _, _, Ys, labs, nms in train_mats:
                tune_mats.append((Ys, labs, len(nms)))
            print(
                f"σ={sigma}: train mats={len(train_mats)}  "
                f"test mats={len(test_mats)}"
            )
    print(f"pooled train mats for tuning={len(tune_mats)}")

    tuned = {}
    saved = json.loads(params_path.read_text()) if params_path.exists() else {}
    skip_tune = set(args.skip_tune or ())
    if skip_tune:
        missing = [n for n in skip_tune if n not in saved or not saved[n].get("params")]
        if missing:
            raise ValueError(
                f"--skip-tune requires saved params for {missing} in {params_path}"
            )
        print(f"skip_tune={sorted(skip_tune)}")
    if args.no_tune:
        for name in selected:
            if name in saved and saved[name].get("params"):
                tuned[name] = {
                    "params": dict(saved[name]["params"]),
                    "best_ari": saved[name].get("best_ari"),
                    "tune_s": saved[name].get("tune_s") or 0.0,
                }
                print(f"\n{'=' * 60}")
                print(
                    f"{name}: skip tuning, loaded "
                    f"{_fmt_params(tuned[name]['params'])}"
                )
            elif name in fixed_defaults:
                tuned[name] = {
                    "params": dict(fixed_defaults[name]),
                    "best_ari": None,
                    "tune_s": 0.0,
                }
                print(f"\n{'=' * 60}")
                print(
                    f"{name}: skip tuning, fixed "
                    f"{_fmt_params(tuned[name]['params'])}"
                )
            else:
                raise ValueError(
                    f"--no-tune requires saved params for {name} in {params_path}"
                )
    else:
        for name in selected:
            if name in fixed_defaults:
                tuned[name] = {
                    "params": dict(fixed_defaults[name]),
                    "best_ari": None,
                    "tune_s": 0.0,
                }
                print(f"\n{'=' * 60}")
                print(f"{name}: skip tuning, fixed {_fmt_params(tuned[name]['params'])}")
                write_params(tuned, params_path)

        for name in selected:
            if name not in tune_specs:
                continue
            if name in skip_tune:
                tuned[name] = {
                    "params": dict(saved[name]["params"]),
                    "best_ari": saved[name].get("best_ari"),
                    "tune_s": saved[name].get("tune_s") or 0.0,
                }
                print(f"\n{'=' * 60}")
                print(
                    f"{name}: skip tuning, loaded "
                    f"{_fmt_params(tuned[name]['params'])}"
                )
                continue
            suggest, run, defaults = tune_specs[name]
            print(f"\n{'=' * 60}")
            print(f"--- tune {name} over {len(tune_mats)} train mats ---")
            result = tune_over(
                name, suggest, run, tune_mats, n_trials=n_trials, enqueue=defaults,
            )
            tuned[name] = result
            print(
                f"  best mean ARI={result['best_ari']:.4f}  "
                f"tune {result['tune_s']:.1f}s  {_fmt_params(result['params'])}"
            )
            write_params(tuned, params_path)

    csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
    mode = "a" if args.append and csv_exists else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        f.flush()

        for sigma, (train_mats, test_mats) in by_sigma.items():
            for split_name, mats in (("train", train_mats), ("test", test_mats)):
                for ds_name, ex_idx, Ys, labs, nms in mats:
                    k = len(nms)
                    print(f"\n=== {ds_name} {split_name}[{ex_idx}] k={k}  "
                          f"dim={Ys.shape[0]}  frames={Ys.shape[1]}  σ={sigma} ===")
                    for name in selected:
                        run = eval_specs[name]
                        params = tuned[name]["params"]
                        scores, elapsed = eval_method(
                            name,
                            lambda Ys=Ys, k=k, run=run, params=params: run(Ys, k, **params),
                            labs,
                        )
                        writer.writerow({
                            "method": name,
                            "dataset": ds_name,
                            "split": split_name,
                            "example": ex_idx,
                            "sigma": sigma,
                            "k": k,
                            "n_people": len(nms),
                            "n_frames": Ys.shape[1],
                            "people": " ".join(nms),
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
