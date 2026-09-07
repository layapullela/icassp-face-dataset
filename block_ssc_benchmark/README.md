# Block SSC Benchmark Bundle

Known-k, clean (σ=0) runs comparing SSC-Block-TV variants to OSC / TKSS / BDOSC.

## Layout

- `results/{surveillance,ballet}/` — evaluation CSVs (`*_variants.csv`, `*_baselines.csv`)
- `params/{surveillance,ballet}/` — tuned / fixed hyperparameters
- `logs/{surveillance,ballet}/` — SLURM stdout/stderr for those jobs
- `solvers/` — copies of `ssc_block_tv*.py` (canonical sources remain in `../surveillance_dataset/`)
- `scripts/` — submit / launch helpers used for these runs

## Job IDs

| Suite | Dataset | k | Job |
|-------|---------|---|-----|
| variants | surveillance | 5/8/12 | 60370471–73 |
| variants | ballet | 5/8/12 | 60370409–11 |
| baselines | surveillance | 5/8/12 | 60371420–22 |
| baselines | ballet | 5/8/12 | 60371423–25 |

Variants: SSC-Block-TV, SSC-Block-TV-Col21, SSC-Block-TV-L1  
Baselines: OSC, TKSS, BDOSC  

## Hold-out CV benchmark (surveillance only)

Rotating k-fold hold-out CV instead of the single fixed split above. The
25 people are shuffled and cut into equal groups of k (remainder dropped);
each group is rotated through as the held-out test set, and hyperparameters
are **retuned per fold** on that fold's train groups only (no leakage of
the held-out group into tuning). Fold counts follow from 25 people: k=3 ->
8 folds, k=5 -> 5 folds, k=8 -> 3 folds (1 person unused).

**k=12 was dropped.** It only supports 2 disjoint groups (1 person
unused), and once one is held out as test, only C(13,12)=13 *distinct*
size-12 subsets exist among the remaining 13 people — not enough
combinatorial room for a real tuning pool. This surfaced from a targeted
ablation (`cv/deprecated/analysis/overlap_ablation.md`): enlarging the
tuning pool helped SSC-Block-TV-Col21 substantially at k=12 under noise
(e.g. σ=0.5 test ARI 0.575→0.618) while leaving OSC/TKSS essentially
unchanged — consistent with SSC-Block-TV-Col21 (4 hyperparameters) being
starved for tuning data relative to OSC (2) and TKSS (3) — but the k=12
fold-to-fold variance stayed high regardless, because with only 2 folds
the *variance estimate itself* is unreliable no matter how well tuning
goes. k=3 was added in its place; the old k=12 run and the ablation that
diagnosed it are preserved in `cv/deprecated/`.

Methods: SSC-Block-TV-Col21, OSC, BDOSC (fixed defaults, not tuned), TKSS,
Gram-NCut (trivial DP NCut baseline, `run_gram_ncut`, not tuned).
Sigmas: 0, 0.25, 0.5. Optuna: 20 base trials (scaled per method by
param count, e.g. SSC-Block-TV-Col21 gets 40, TKSS gets 30).

**Tuning pool (standard for every k here, not an ablation):** every
tuned method uses `--tune-pool-groups 20` — each fold's hyperparameter
search runs on 20 randomly drawn, *distinct*, possibly-overlapping
k-sized subsets of that fold's held-in people (24 sequences each, so
20×24=480 matrices), instead of just the fold's 1–7 fixed partition
groups. `sample_overlapping_groups()` in `cluster_experiment.py` raises
if the held-in pool can't combinatorially support 20 distinct subsets —
this is exactly the check that rules out k=12. Reported train/test eval
rows still come from the original fixed partition groups.

**Exactly what "train" and "test" mean per fold** (`cluster_experiment.py`,
`--cv` branch of `main()`):

- **Test.** One fixed k-person group from the original partition (the
  same group every time this fold is evaluated) x 24 sequences = 24 test
  matrices. This is a single people-composition shot across sequences,
  *not* a combinatorial sample of test groups — it's coincidentally
  >=20 only because there happen to be 24 sequences.
- **Held-in pool.** `held_in = all 25 people - this fold's test group`
  (`held_in = sorted(set(people) - set(test_group))`) — computed fresh
  per fold, so it's exactly "everyone not currently being tested."
- **Train (reported rows).** The fold's other 1–7 fixed partition
  groups (disjoint from `held_in`'s complement by construction, since
  `chunk_folds` partitions all 25 people up front) x 24 sequences.
  These are the matrices written to the CSV as `split=train`.
- **Tune (hyperparameter search objective only, not reported as rows).**
  20 *distinct* k-sized subsets sampled from `held_in` (not the full
  C(len(held_in), k) space — just 20 random draws out of it, rejecting
  repeats) x 24 sequences = 480 matrices. Individual people recur across
  these 20 groups (that's the intended overlap for variety); no two of
  the 20 groups are the exact same set of people; and every person in
  every one of them comes from `held_in`, so none of them is ever a
  person from that fold's test group.

Net effect: a person in a given fold's test group never appears in that
fold's train or tune matrices, in either construction.

One SLURM job per (method, k, sigma) = 5 x 3 x 3 = 45 jobs, submitted via
`scripts/submit_cv_surveillance_slurm.py`. Each job calls
`surveillance_dataset/cluster_experiment.py --cv --tune-pool-groups 20
--out-dir cv/` and everything lands together under **`cv/`** (kept
separate from the single-split `results/params/logs` trees above so the
two don't mix):

- `cv/results/surveillance/*_cv_{method}_k{k}_sigma{sigma}.csv` —
  per-matrix eval rows (adds a `fold` column) for both `train` and `test`
  splits
- `cv/results/surveillance/*_cv_{method}_k{k}_sigma{sigma}_timing.json` —
  mean/std/total inference seconds per method (excludes tuning time)
- `cv/params/surveillance/*_cv_..._sigma{s}_fold{i}.json` — tuned/fixed
  hyperparameters per (sigma, fold)
- `cv/logs/surveillance/cv_{method}_k{k}_s{sigma}_%j.{out,err}`

### Validity notes

- Per-fold retuning means each fold can select different hyperparameters;
  the CSV `params` column records what was actually used per row so this
  is auditable, but it also means "the model" isn't fixed across folds
  the way a deployed model would be — this is standard nested CV, not a
  claim that one hyperparameter setting generalizes.
- BDOSC and Gram-NCut use fixed/no hyperparameters (per the original
  benchmark), so CV only changes which data they're scored on, not what
  they tune against.

### History

- **v1** (job IDs 60400185–60400230, k={5,8,12}, no tuning-pool
  enlargement) and the **overlap ablation** (job IDs
  60418877–60418894, k={8,12}, `--tune-pool-groups 4` on
  SSC-Block-TV-Col21/OSC/TKSS only) are archived in `cv/deprecated/`.
  The ablation is what motivated dropping k=12 and making an enlarged
  tuning pool (N=20) standard.
- **Current** (this section): k={3,5,8}, `--tune-pool-groups 20`
  standard for every tuned method. Submitted via
  `scripts/submit_cv_surveillance_slurm.py`; see `squeue -u $USER` /
  `sacct` for live job status and
  `scripts/cv_slurm_scripts/run_cv_{method}_k{k}_s{sigma}.sh` for the
  exact job -> config mapping.

## Hold-out CV benchmark (ballet)

Same design as the surveillance CV benchmark above, ported to
`ballet_dataset/cluster_experiment.py` (`--cv` / `--tune-pool-groups` /
`--out-dir` flags added there for this). The unit that gets partitioned
into folds is a *ballet sequence* (`seq_000001`, ...) instead of a
*person* — each ballet sequence is already one temporal cluster, so a
fold's k-sequence group directly becomes one clustering matrix (no
per-video-camera repetition the way surveillance has 24 matrices per
group; ballet has exactly 1).

44 ballet sequences total, shuffled and cut into equal groups of k
(remainder dropped), each group rotated through as the held-out test
set: k=3 -> 14 folds (2 sequences unused), k=5 -> 8 folds (4 unused),
k=8 -> 5 folds (4 unused). Unlike surveillance's 25-person pool, every
held-in pool here (30-41 sequences) comfortably supports 20 distinct
k-sized tuning subsets, so no k value needed to be dropped for
combinatorial reasons — the same K_VALUES={3,5,8} were used for both
datasets from the start.

Config mirrors the surveillance CV benchmark exactly: methods =
SSC-Block-TV-Col21, OSC, BDOSC (fixed defaults), TKSS, Gram-NCut
(not tuned); sigmas = 0, 0.25, 0.5; Optuna 20 base trials (scaled per
method by param count); `--tune-pool-groups 20` for every tuned method.

**Train/test/tune per fold** (`cluster_experiment.py --cv` branch):

- **Test.** The fold's held-out k-sequence group -> 1 test matrix.
- **Held-in pool.** All 44 (minus any dropped short sequences) sequences
  minus this fold's test group.
- **Train (reported rows).** The fold's other fixed partition groups
  (one matrix each).
- **Tune (hyperparameter search objective only).** 20 distinct k-sized
  subsets sampled from the held-in pool (`sample_combos`), not the
  reported train rows.

One SLURM job per (method, k, sigma) = 5 x 3 x 3 = 45 jobs, submitted
via `scripts/submit_cv_ballet_slurm.py`. Everything lands under
**`cv/`** alongside the surveillance CV output, in a parallel `ballet/`
subtree so the two datasets never collide:

- `cv/results/ballet/*_cv_{method}_k{k}_sigma{sigma}.csv` — per-matrix
  eval rows (adds a `fold` column) for both `train` and `test` splits
- `cv/results/ballet/*_cv_{method}_k{k}_sigma{sigma}_timing.json`
- `cv/params/ballet/*_cv_..._sigma{s}_fold{i}.json`
- `cv/logs/ballet/cvb_{method}_k{k}_s{sigma}_%j.{out,err}`
