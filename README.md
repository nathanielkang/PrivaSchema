# PrivaSchema

Multi-table database synthesis under a single differential-privacy budget.

PrivaSchema allocates a global privacy budget across tables in a relational schema, synthesizes tables in parent-first order with foreign-key conditioning, and repairs residual referential violations with minimum-cost transport.

Paper: *One Privacy Budget to Rule Them All: Synthesizing Multi-Table Databases Under Differential Privacy*.

## Setup

```bash
conda create -n privaschema python=3.10 -y
conda activate privaschema
pip install -r requirements.txt
```

## Layout

```
configs/     Experiment YAML
src/         Core library
scripts/     Runners
results/     Local output
```

## Quick start

```bash
python scripts/run_experiments.py --config configs/quick_test.yaml --output results/
python scripts/run_ablation.py --config configs/default.yaml --output results/ablation/
python scripts/run_extra_methods.py --output results/extra/
python scripts/run_experiments.py --config configs/eps_grid.yaml --output results/eps_grid/
```

Default methods: `privaschema`, plus simple ablations `equal_split` / `independent`.

Extra methods `F` / `U` / `C` / `P`: parent-first AIM with equal split, convex split with uniform FK, convex allocation without parent-first, and optional PrivLava. If PrivLava is not installed, the runner writes a skip JSON. See `extras/README.md`.

The epsilon-grid config uses ε ∈ {0.1, 0.5, 1, 2, 4} and three seeds. Writers emit per-run CSV/JSON plus seed mean and sample standard deviation. The quadratic FK split is 0.25 / 0.75. Synthetic joins are clipped at 10× the real join cardinality. Membership inference is an optional random-forest presence test on the clipped join.

## Datasets

Loaders generate synthetic proxies when CSVs are absent under `data/`. Names: berka, imdb, tpch, rossmann, walmart, university, synthetic_star, synthetic_tiny.

## License

MIT
