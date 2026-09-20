# PrivaSchema

Multi-table database synthesis under a **single** differential-privacy budget.

PrivaSchema allocates a global privacy budget across tables in a relational schema (workload-aware linear program), synthesizes tables in parent-first order with foreign-key conditioning, and repairs residual referential violations with minimum-cost transport under Rényi composition.

**Public GitHub** ([nathanielkang/PrivaSchema](https://github.com/nathanielkang/PrivaSchema)) is **code-only** and ships the **proposed PrivaSchema pipeline** plus documented optional extras. Competitor clones are not bundled.

Paper title: *One Privacy Budget to Rule Them All: Synthesizing Multi-Table Databases Under Differential Privacy*.

This tree is the **IEEE TIFS** retarget of that method. The prior Big Data venue is closed. Manuscript TeX/PDF stay off this remote.

## Setup

```bash
conda create -n privaschema python=3.10 -y
conda activate privaschema
pip install -r requirements.txt
```

## Layout

```
configs/     Experiment YAML
src/         Core library (proposed method)
scripts/     Runners
results/     Local output
```

## Quick start

```bash
python scripts/run_experiments.py --config configs/quick_test.yaml --output results/
python scripts/run_ablation.py --config configs/default.yaml --output results/ablation/
```

Default public-facing methods: `privaschema`, plus simple ablations `equal_split` / `independent`.

TIFS missing-cell runners (no precomputed numbers in the repo):

```bash
# ε grid {0.1, 0.5, 1, 2, 4}, three seeds, mean + sample std
python scripts/run_experiments.py --config configs/tifs_grid.yaml --output results/tifs_grid/

# F / aim_parent_first_equal — parent-first AIM, ε_i = ε/√K
# U / aim_parent_first_uniform_fk — parent-first + uniform key sampling
# C / convex_no_parent_first — convex allocator, independent tables
# P / privlava — official extra only (skip stub JSON if the clone is missing)
python scripts/run_tifs_missing.py --output results/tifs_missing/
```

Accountant knobs (scientific names; see YAML comments): quadratic foreign-key share 0.25 / synthesis share 0.75; join-cardinality clip 10× the real join. Writers emit per-seed rows and `*_seed_stats.*` (mean and sample std over 3 seeds). Dataset dumps report K, L, n_i, FK edges, and fanout. Optional membership-inference hook: real-joined vs synth-joined random-forest holdout (`evaluation.membership_inference` in the YAML). Multi-parent children (IMDB `cast`, TPC-H `partsupp`) use joint junction sampling.

Optional extras: see `extras/README.md`. Method P writes `results/.../skipped/*.json` and does **not** invent a numeric row.

## Datasets

Loaders generate synthetic proxies when CSVs are absent under `data/`. Names: berka, imdb, tpch, rossmann, walmart, university, synthetic_star, synthetic_tiny.

## License

MIT
