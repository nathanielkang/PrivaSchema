"""Main experiment runner: PrivaSchema pipeline + baselines across epsilon values."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.budget_allocator import BudgetAllocator
from src.data.datasets import load_dataset
from src.evaluation.metrics import evaluate_all
from src.integrity_enforcer import MinCostRepairer
from src.schema import RelationalSchema
from src.sequential_synthesizer import SequentialSynthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Pipeline runners
# ======================================================================

def run_privaschema(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Full PrivaSchema pipeline: allocate -> synthesise -> enforce."""
    table_sizes = {tname: len(df) for tname, df in real_data.items()}
    allocator = BudgetAllocator(
        schema=schema,
        epsilon_total=epsilon,
        delta=cfg["privacy"]["delta"],
        epsilon_min=cfg["privacy"]["epsilon_min_per_table"],
        method=cfg["allocator"]["method"],
        query_weight_decay=cfg["allocator"].get("query_weight_decay", 0.9),
        table_sizes=table_sizes,
    )
    workload = allocator.generate_workload(
        query_types=cfg["evaluation"]["query_types"],
        num_queries=cfg["evaluation"]["num_queries"],
    )
    eps_alloc = allocator.allocate(workload)
    logger.info("Budget allocation (eps_total=%.2f): %s", epsilon, eps_alloc)

    synthesizer = SequentialSynthesizer(
        schema=schema,
        epsilon_alloc=eps_alloc,
        num_bins=cfg["synthesis"].get("num_bins", 32),
        max_parents_bn=cfg["synthesis"].get("max_parents_bn", 3),
        seed=seed,
    )
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(real_data)

    repairer = MinCostRepairer(
        schema=schema,
        method=cfg["enforcer"]["method"],
        max_iterations=cfg["enforcer"]["max_repair_iterations"],
        ot_reg=cfg["enforcer"].get("ot_reg", 0.01),
    )
    synthetic_data = repairer.repair(synthetic_data)

    return synthetic_data


def run_equal_split(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Baseline: equal budget split, parent-first synthesis, with repair."""
    allocator = BudgetAllocator(
        schema=schema,
        epsilon_total=epsilon,
        delta=cfg["privacy"]["delta"],
        epsilon_min=cfg["privacy"]["epsilon_min_per_table"],
        method="equal",
    )
    eps_alloc = allocator.allocate()

    synthesizer = SequentialSynthesizer(
        schema=schema, epsilon_alloc=eps_alloc,
        num_bins=cfg["synthesis"].get("num_bins", 32),
        max_parents_bn=cfg["synthesis"].get("max_parents_bn", 3),
        seed=seed,
    )
    synthesizer.fit(real_data)
    synthetic_data = synthesizer.sample(real_data)

    repairer = MinCostRepairer(schema=schema, method="greedy", max_iterations=50)
    synthetic_data = repairer.repair(synthetic_data)
    return synthetic_data


def run_independent(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Baseline: independent per-table synthesis (no FK conditioning, no repair)."""
    from src.sequential_synthesizer import SingleTableSynthesizer

    n_tables = schema.num_tables
    eps_per_table = epsilon / max(n_tables, 1)
    synthetic_data: dict[str, pd.DataFrame] = {}

    for tname in schema.topological_order():
        df = real_data[tname]
        table_meta = schema.tables[tname]
        synth = SingleTableSynthesizer(
            epsilon=eps_per_table,
            num_bins=cfg["synthesis"].get("num_bins", 32),
            max_parents=cfg["synthesis"].get("max_parents_bn", 3),
            seed=seed,
        )
        synth.fit(df, table_meta.dtypes)
        synthetic_data[tname] = synth.sample(len(df))

    return synthetic_data


def run_non_private_sdv(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Non-private baseline using SDV's GaussianCopula per table."""
    try:
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import Metadata
    except ImportError:
        logger.warning("SDV not installed; skipping non-private baseline.")
        return {}

    synthetic_data: dict[str, pd.DataFrame] = {}

    for tname in schema.topological_order():
        df = real_data[tname]
        table_meta = schema.tables[tname]

        metadata = Metadata()
        metadata.add_table(tname)

        if table_meta.primary_key and table_meta.primary_key in df.columns:
            metadata.update_column(
                table_name=tname,
                column_name=table_meta.primary_key,
                sdtype="id",
            )

        for col in df.columns:
            if col == table_meta.primary_key:
                continue
            dtype = table_meta.dtypes.get(col, "categorical")
            sdtype = "numerical" if dtype == "numerical" else "categorical"
            metadata.update_column(table_name=tname, column_name=col, sdtype=sdtype)

        try:
            synth = GaussianCopulaSynthesizer(metadata, table_name=tname)
            synth.fit(df)
            synthetic_data[tname] = synth.sample(len(df))
        except Exception as e:
            logger.warning("SDV failed for %s: %s. Using simple sample instead.", tname, e)
            synthetic_data[tname] = df.sample(n=len(df), replace=True, random_state=seed).reset_index(drop=True)

    return synthetic_data


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run PrivaSchema experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset list")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else cfg["experiment"]["datasets"]
    epsilons = cfg["experiment"]["epsilons"]
    num_runs = cfg["experiment"]["num_runs"]
    base_seed = cfg["experiment"]["seed"]

    all_results: list[dict] = []

    for ds_name in datasets:
        logger.info("=" * 60)
        logger.info("Dataset: %s", ds_name)
        logger.info("=" * 60)

        real_data, schema = load_dataset(ds_name)
        logger.info(
            "Loaded %d tables: %s",
            schema.num_tables,
            list(schema.tables.keys()),
        )

        methods = {
            "privaschema": run_privaschema,
            "equal_split": run_equal_split,
            "independent": run_independent,
        }

        for epsilon in epsilons:
            for run_idx in range(num_runs):
                seed = base_seed + run_idx

                for method_name, method_fn in methods.items():
                    logger.info(
                        "  [%s] eps=%.2f run=%d/%d",
                        method_name, epsilon, run_idx + 1, num_runs,
                    )
                    t0 = time.time()
                    synthetic_data = method_fn(real_data, schema, cfg, epsilon, seed)
                    elapsed = time.time() - t0

                    eval_result = evaluate_all(
                        real_data, synthetic_data, schema,
                        num_queries=cfg["evaluation"]["num_queries"],
                        query_types=cfg["evaluation"]["query_types"],
                        test_fraction=cfg["evaluation"].get("ml_test_fraction", 0.3),
                        seed=seed,
                    )

                    record = {
                        "dataset": ds_name,
                        "method": method_name,
                        "epsilon": epsilon,
                        "run": run_idx,
                        "time_sec": round(elapsed, 2),
                        **_flatten(eval_result),
                    }
                    all_results.append(record)
                    logger.info("    -> %s", {k: v for k, v in record.items() if k not in ("dataset", "method")})

        logger.info("  Running non-private SDV baseline...")
        t0 = time.time()
        sdv_syn = run_non_private_sdv(real_data, schema, seed=base_seed)
        elapsed = time.time() - t0

        if sdv_syn:
            eval_result = evaluate_all(
                real_data, sdv_syn, schema,
                num_queries=cfg["evaluation"]["num_queries"],
                query_types=cfg["evaluation"]["query_types"],
                seed=base_seed,
            )
            record = {
                "dataset": ds_name,
                "method": "sdv_non_private",
                "epsilon": float("inf"),
                "run": 0,
                "time_sec": round(elapsed, 2),
                **_flatten(eval_result),
            }
            all_results.append(record)

    results_df = pd.DataFrame(all_results)
    out_path = output_dir / "experiment_results.csv"
    results_df.to_csv(out_path, index=False)
    logger.info("Results saved to %s", out_path)

    json_path = output_dir / "experiment_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("JSON results saved to %s", json_path)

    _print_summary(results_df)


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict for tabular storage."""
    flat: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


def _print_summary(df: pd.DataFrame) -> None:
    """Print a summary table using only metrics common across all datasets."""
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 80)

    UNIVERSAL_METRICS = [
        "fk_consistency.overall",
        "query_accuracy.query_error_avg",
        "query_accuracy.query_error_1way",
        "query_accuracy.query_error_2way",
        "marginal_tv_avg",
        "ml_utility.accuracy_syn",
        "ml_utility.f1_syn",
    ]

    key_cols = ["dataset", "method", "epsilon"]
    metric_cols = [c for c in UNIVERSAL_METRICS if c in df.columns]

    if not metric_cols:
        non_key = [c for c in df.columns if c not in key_cols + ["run", "time_sec"]]
        for c in non_key:
            if df[c].notna().any():
                metric_cols.append(c)
            if len(metric_cols) >= 6:
                break

    if not metric_cols:
        logger.info("No metrics available for summary.")
        return

    summary = df.groupby(key_cols)[metric_cols].mean().round(4)
    logger.info("\n%s", summary.to_string())

    per_method = df.groupby("method")[metric_cols].mean().round(4)
    logger.info("\n--- Averaged across datasets and epsilons ---")
    logger.info("\n%s", per_method.to_string())


if __name__ == "__main__":
    main()
