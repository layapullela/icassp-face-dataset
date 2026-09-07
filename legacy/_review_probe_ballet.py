import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "surveillance_dataset"))
sys.path.insert(0, str(HERE / "ballet_dataset"))

import cluster_experiment as be  # noqa: E402  (ballet, first on path)
from dp_contiguous_partition import cluster_from_C_ordered  # noqa: E402

all_dirs = be.sequence_dirs()
name_to_dir = {p.name: p for p in all_dirs}
train_pool, test_pool = be.split_pools(all_dirs, n_test=be.N_TEST, seed=be.SEED)
combo_rng = np.random.default_rng(be.SEED)
train_groups = be.sample_combos(train_pool, 5, be.N_TRAIN_COMBOS, combo_rng)
test_groups = be.sample_combos(test_pool, 5, be.N_TEST_COMBOS, combo_rng)

frame_rng = np.random.default_rng(be.SEED + 1)
used = sorted({n for g in train_groups + test_groups for n in g})
tracks = {n: be.load_sequence(name_to_dir[n], frame_rng)[0] / 255.0 for n in used}

for tag, groups in (("train", train_groups), ("test", test_groups)):
    eq_s, gram_s = [], []
    for names in groups:
        Y01, labels, _ = be.concat_group(tracks, names)
        Y = be.column_normalize(Y01)
        k, N = len(names), Y.shape[1]
        eq = np.repeat(np.arange(k), np.diff(np.linspace(0, N, k + 1, dtype=int)))
        eq_s.append(be.metrics(labels, eq))
        gram_s.append(be.metrics(labels, cluster_from_C_ordered(Y.T @ Y, k)))
    for nm, sc in (("equal-chunk (ignores data)", eq_s),
                   ("DP-NCut on raw Gram Y^T Y", gram_s)):
        print(f"  ballet {tag:5s} {nm:28s} "
              f"ACC={np.mean([s['acc'] for s in sc]):.4f}  "
              f"ARI={np.mean([s['ari'] for s in sc]):.4f}  n={len(sc)}")
