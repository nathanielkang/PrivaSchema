# PrivaSchema: Multi-Table DP Synthesis Under a Single Privacy Budget

Experiment code for SIGMOD 2027 Paper B.

## Setup

```bash
conda create -n privaschema python=3.10 -y
conda activate privaschema
pip install -r requirements.txt
```

## Directory Structure

```
configs/          Configuration YAML files
src/              Core library
  schema.py                Relational schema representation
  budget_allocator.py      LP-based privacy budget allocation
  sequential_synthesizer.py  Parent-first DP table synthesis
  integrity_enforcer.py    FK integrity enforcement via optimal transport
  data/datasets.py         Multi-table dataset loaders
  evaluation/metrics.py    Evaluation metrics (query accuracy, FK consistency, ML utility)
scripts/          Experiment runners
  run_experiments.py       Main experiments (vary epsilon, baselines)
  run_ablation.py          Ablation studies
```

## Running Experiments

```bash
# Full experiment suite
python scripts/run_experiments.py --config configs/default.yaml --output results/

# Ablation studies
python scripts/run_ablation.py --config configs/default.yaml --output results/ablation/
```

## Datasets

- **Berka Financial** (8 tables): auto-downloaded or synthetic proxy generated
- **Rossmann Store Sales** (3 tables): auto-downloaded or synthetic proxy generated
- **Synthetic Star Schema**: configurable number of tables for scalability testing
