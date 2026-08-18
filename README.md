# PrivaSchema

Multi-table database synthesis under a **single** differential-privacy budget.

PrivaSchema allocates a global privacy budget across tables in a relational schema (workload-aware linear program), synthesizes tables in parent-first order with foreign-key conditioning, and repairs residual referential violations with minimum-cost transport under Rényi composition.

This public mirror contains the **proposed method** only (allocator, sequential synthesizer, integrity repair, evaluation helpers). Competitor / baseline runners are not included.

Paper title: *One Privacy Budget to Rule Them All: Synthesizing Multi-Table Databases Under Differential Privacy*.

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
scripts/     Runners (proposed method + simple ablations)
results/     Local output directory (empty or user-generated)
```

## Quick start

```bash
python scripts/run_experiments.py --config configs/quick_test.yaml --output results/
```

Default methods in the public scripts: `privaschema`, plus same-family ablations `equal_split` and `independent`.

## Datasets

Loaders can synthesize schema-compatible proxies when raw CSVs are absent. Names include berka, imdb, tpch, rossmann, walmart, university, and synthetic_star.

## License

MIT
