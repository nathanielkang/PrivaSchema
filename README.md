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
```

Default methods: `privaschema`, plus simple ablations `equal_split` / `independent`.

Optional extras (AIM, PrivLava): see `extras/README.md`.

## Datasets

Loaders generate synthetic proxies when CSVs are absent under `data/`. Names: berka, imdb, tpch, rossmann, walmart, university, synthetic_star, synthetic_tiny.

## License

MIT
