# PrivaSchema

Multi-table database synthesis under a **single** differential-privacy budget.

PrivaSchema allocates a global privacy budget across tables in a relational schema (workload-aware linear program), synthesizes tables in parent-first order with foreign-key conditioning, and repairs residual referential violations with minimum-cost transport under Rényi composition.

**Public GitHub** ([nathanielkang/PrivaSchema](https://github.com/nathanielkang/PrivaSchema)) is **code-only** and ships the **proposed PrivaSchema pipeline** plus documented optional extras. Competitor clones are not bundled.

Paper title: *One Privacy Budget to Rule Them All: Synthesizing Multi-Table Databases Under Differential Privacy* (IEEE TIFS prep pack).

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
# ε grid {0.1, 0.5, 1, 2, 4}, three seeds, F/U/C/P + proposed
python scripts/run_experiments.py --config configs/tifs_grid.yaml --output results/tifs_grid/

# F = parent-first AIM + equal ε/√K
# U = convex split + uniform FK (histogram ablation)
# C = convex allocator, no parent-first
# P = optional PrivLava extra (fails unless extras/PrivLava is installed)
python scripts/run_tifs_missing.py --output results/tifs_missing/
```

JSON/CSV writers emit per-seed rows and `*_seed_stats.*` with mean and sample std. Dataset dumps (`dataset_stats.json`) report K, L, n_i, fanout, and the 10× join-cardinality clip. Multi-parent children (IMDB `cast`, TPC-H `partsupp`) use joint junction sampling.

Optional extras: see `extras/README.md`. Method P does not write a stand-in row.

## Datasets

Loaders generate synthetic proxies when CSVs are absent under `data/`. Names: berka, imdb, tpch, rossmann, walmart, university, synthetic_star.

## License

MIT
