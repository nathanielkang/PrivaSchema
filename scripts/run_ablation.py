"""Ablation studies for PrivaSchema components."""

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
from src.data.datasets import generate_star_schema, load_dataset
from src.evaluation.metrics import evaluate_all
from src.integrity_enforcer import MinCostRepairer
from src.schema import RelationalSchema
from src.sequential_synthesizer import SequentialSynthesizer, SingleTableSynthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# Ablation variants
# ======================================================================

def ablation_full(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Full PrivaSchema (LP allocator + parent-first + enforcer)."""
    allocator = BudgetAllocator(
        schema=schema, epsilon_total=epsilon,
        delta=cfg["privacy"]["delta"],
        epsilon_min=cfg["privacy"]["epsilon_min_per_table"],
        method="lp",
        query_weight_decay=cfg["allocator"].get("query_weight_decay", 0.9),
    )
    workload = allocator.generate_workload(
        query_types=cfg["evaluation"]["query_types"],
        num_queries=cfg["evaluation"]["num_queries"],
    )
    eps_alloc = allocator.allocate(workload)

    synth = SequentialSynthesizer(
        schema=schema, epsilon_alloc=eps_alloc,
        num_bins=cfg["synthesis"].get("num_bins", 32),
        max_parents_bn=cfg["synthesis"].get("max_parents_bn", 3),
        seed=seed,
    )
    synth.fit(real_data)
    syn_data = synth.sample(real_data)

    repairer = MinCostRepairer(
        schema=schema,
        method=cfg["enforcer"]["method"],
        max_iterations=cfg["enforcer"]["max_repair_iterations"],
    )
    return repairer.repair(syn_data)


def ablation_no_lp(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Without LP allocator: use equal split instead."""
    eps_alloc = {t: epsilon / schema.num_tables for t in schema.tables}

    synth = SequentialSynthesizer(
        schema=schema, epsilon_alloc=eps_alloc,
        num_bins=cfg["synthesis"].get("num_bins", 32),
        max_parents_bn=cfg["synthesis"].get("max_parents_bn", 3),
        seed=seed,
    )
    synth.fit(real_data)
    syn_data = synth.sample(real_data)

    repairer = MinCostRepairer(
        schema=schema, method=cfg["enforcer"]["method"],
        max_iterations=cfg["enforcer"]["max_repair_iterations"],
    )
    return repairer.repair(syn_data)


def ablation_no_conditioning(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Without parent-first conditioning: independent per-table, then enforce."""
    n_tables = schema.num_tables
    eps_per_table = epsilon / max(n_tables, 1)
    syn_data: dict[str, pd.DataFrame] = {}

    for tname in schema.topological_order():
        df = real_data[tname]
        table_meta = schema.tables[tname]
        s = SingleTableSynthesizer(
            epsilon=eps_per_table,
            num_bins=cfg["synthesis"].get("num_bins", 32),
            max_parents=cfg["synthesis"].get("max_parents_bn", 3),
            seed=seed,
        )
        s.fit(df, table_meta.dtypes)
        syn_data[tname] = s.sample(len(df))

    repairer = MinCostRepairer(
        schema=schema, method=cfg["enforcer"]["method"],
        max_iterations=cfg["enforcer"]["max_repair_iterations"],
    )
    return repairer.repair(syn_data)


def ablation_no_enforcer(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Without integrity enforcer: LP + parent-first but no repair."""
    allocator = BudgetAllocator(
        schema=schema, epsilon_total=epsilon,
        delta=cfg["privacy"]["delta"],
        epsilon_min=cfg["privacy"]["epsilon_min_per_table"],
        method="lp",
        query_weight_decay=cfg["allocator"].get("query_weight_decay", 0.9),
    )
    workload = allocator.generate_workload(
        query_types=cfg["evaluation"]["query_types"],
        num_queries=cfg["evaluation"]["num_queries"],
    )
    eps_alloc = allocator.allocate(workload)

    synth = SequentialSynthesizer(
        schema=schema, epsilon_alloc=eps_alloc,
        num_bins=cfg["synthesis"].get("num_bins", 32),
        max_parents_bn=cfg["synthesis"].get("max_parents_bn", 3),
        seed=seed,
    )
    synth.fit(real_data)
    return synth.sample(real_data)


# ======================================================================
# Scalability study
# ======================================================================

def scalability_study(
    cfg: dict,
    table_counts: list[int] | None = None,
    epsilon: float = 1.0,
    seed: int = 42,
) -> list[dict]:
    """Vary number of tables in a star schema and measure runtime + quality."""
    if table_counts is None:
        table_counts = [2, 4, 8, 16, 32]

    results: list[dict] = []

    for n_dim in table_counts:
        logger.info("Scalability: %d dimension tables (%d total)", n_dim, n_dim + 1)
        real_data, schema = generate_star_schema(
            num_dimension_tables=n_dim,
            dim_rows=100,
            fact_rows=2000,
            seed=seed,
        )

        t0 = time.time()
        syn_data = ablation_full(real_data, schema, cfg, epsilon, seed)
        elapsed = time.time() - t0

        eval_result = evaluate_all(
            real_data, syn_data, schema,
            num_queries=50,
            query_types=["1way", "2way", "cross_table"],
            seed=seed,
        )

        record = {
            "num_dim_tables": n_dim,
            "total_tables": n_dim + 1,
            "epsilon": epsilon,
            "time_sec": round(elapsed, 2),
            "fk_consistency": eval_result["fk_consistency"].get("overall", 0),
            "query_error_avg": eval_result["query_accuracy"].get("query_error_avg", float("nan")),
        }
        results.append(record)
        logger.info("  -> time=%.2fs, fk=%.4f, qerr=%.4f",
                     elapsed, record["fk_consistency"], record["query_error_avg"])

    return results


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run PrivaSchema ablation studies")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="results/ablation/")
    parser.add_argument("--dataset", type=str, default="berka")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg["experiment"]["seed"]
    epsilon = cfg["privacy"]["epsilon_total"]

    # --- Component ablation ---
    logger.info("=" * 60)
    logger.info("COMPONENT ABLATION on %s (eps=%.2f)", args.dataset, epsilon)
    logger.info("=" * 60)

    real_data, schema = load_dataset(args.dataset)

    ablation_methods = {
        "full": ablation_full,
        "no_lp_allocator": ablation_no_lp,
        "no_conditioning": ablation_no_conditioning,
        "no_enforcer": ablation_no_enforcer,
    }

    ablation_results: list[dict] = []

    for name, fn in ablation_methods.items():
        logger.info("  Ablation: %s", name)
        t0 = time.time()
        syn_data = fn(real_data, schema, cfg, epsilon, seed)
        elapsed = time.time() - t0

        eval_result = evaluate_all(
            real_data, syn_data, schema,
            num_queries=cfg["evaluation"]["num_queries"],
            query_types=cfg["evaluation"]["query_types"],
            seed=seed,
        )

        record = {
            "dataset": args.dataset,
            "ablation": name,
            "epsilon": epsilon,
            "time_sec": round(elapsed, 2),
            "fk_consistency": eval_result["fk_consistency"].get("overall", 0),
            "query_error_avg": eval_result["query_accuracy"].get("query_error_avg", float("nan")),
            "ml_accuracy_syn": eval_result["ml_utility"].get("accuracy_syn", 0),
            "ml_accuracy_real": eval_result["ml_utility"].get("accuracy_real", 0),
            "ml_f1_syn": eval_result["ml_utility"].get("f1_syn", 0),
        }
        ablation_results.append(record)
        logger.info("    -> %s", record)

    abl_df = pd.DataFrame(ablation_results)
    abl_df.to_csv(output_dir / "ablation_components.csv", index=False)
    logger.info("Component ablation saved to %s", output_dir / "ablation_components.csv")

    # --- Scalability ---
    logger.info("=" * 60)
    logger.info("SCALABILITY STUDY")
    logger.info("=" * 60)

    scale_results = scalability_study(
        cfg=cfg,
        table_counts=[2, 4, 8, 16],
        epsilon=epsilon,
        seed=seed,
    )
    scale_df = pd.DataFrame(scale_results)
    scale_df.to_csv(output_dir / "scalability.csv", index=False)
    logger.info("Scalability results saved to %s", output_dir / "scalability.csv")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    logger.info("\nComponent ablation:\n%s", abl_df.to_string(index=False))
    logger.info("\nScalability:\n%s", scale_df.to_string(index=False))

    combined = {
        "component_ablation": ablation_results,
        "scalability": scale_results,
    }
    with open(output_dir / "ablation_all.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)


if __name__ == "__main__":
    main()
