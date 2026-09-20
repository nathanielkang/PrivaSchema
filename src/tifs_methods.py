"""TIFS missing-cell runners: F, U, C (P lives in optional_privlava).

Codes match the TIFS estimate ledger:
  F — parent-first AIM + equal ε/√K
  U — parent-first + convex split + uniform FK (no degree hist)
  C — convex allocator, no parent-first
  P — optional PrivLava extra (official stack only)
"""

from __future__ import annotations

import logging

import pandas as pd

from src.budget_allocator import BudgetAllocator
from src.integrity_enforcer import MinCostRepairer
from src.optional_aim import require_aim_stack, synthesize_one_table_aim
from src.optional_privlava import run_privlava as run_privlava_optional
from src.schema import RelationalSchema
from src.sequential_synthesizer import SequentialSynthesizer, SingleTableSynthesizer

logger = logging.getLogger(__name__)


def _allocator(
    schema: RelationalSchema,
    real_data: dict[str, pd.DataFrame],
    cfg: dict,
    epsilon: float,
    method: str,
) -> dict[str, float]:
    table_sizes = {t: len(df) for t, df in real_data.items()}
    allocator = BudgetAllocator(
        schema=schema,
        epsilon_total=epsilon,
        delta=cfg["privacy"]["delta"],
        epsilon_min=cfg["privacy"]["epsilon_min_per_table"],
        method=method,
        query_weight_decay=cfg.get("allocator", {}).get("query_weight_decay", 0.9),
        table_sizes=table_sizes,
        workload_mode=cfg.get("allocator", {}).get("workload_mode", "paper"),
    )
    workload = allocator.generate_workload(
        query_types=cfg.get("evaluation", {}).get("query_types", ["1way", "cross_table"]),
        num_queries=cfg.get("evaluation", {}).get("num_queries", 100),
    )
    return allocator.allocate(workload)


def _repair(
    schema: RelationalSchema,
    synthetic: dict[str, pd.DataFrame],
    cfg: dict,
) -> dict[str, pd.DataFrame]:
    enc = cfg.get("enforcer", {})
    repairer = MinCostRepairer(
        schema=schema,
        method=enc.get("method", "optimal_transport"),
        max_iterations=enc.get("max_repair_iterations", 100),
        ot_reg=enc.get("ot_reg", 0.01),
    )
    return repairer.repair(synthetic)


def _fk_rdp_share(cfg: dict) -> float:
    syn = cfg.get("synthesis", {})
    if "fk_rdp_share" in syn:
        return float(syn["fk_rdp_share"])
    return float(syn.get("fk_hist_frac", 0.25))


def run_parent_first_aim_equal(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """F: AIM per table under ε_i = ε/√K, parent-first FK hist, then repair."""
    require_aim_stack()
    eps_alloc = _allocator(schema, real_data, cfg, epsilon, method="equal_rdp")
    logger.info("F equal_rdp alloc (eps_total=%.2f): %s", epsilon, eps_alloc)
    delta = float(cfg.get("privacy", {}).get("delta", 1e-5))
    share = _fk_rdp_share(cfg)

    from src.fk_degree import fit_child_fk_hists, split_table_and_fk_eps
    from src.sequential_synthesizer import condition_fk_columns

    synthetic: dict[str, pd.DataFrame] = {}
    for tname in schema.topological_order():
        eps_i = float(eps_alloc.get(tname, epsilon))
        eps_synth, eps_fk = split_table_and_fk_eps(eps_i, fk_rdp_share=share)
        hists = fit_child_fk_hists(schema, tname, real_data[tname], eps_fk, seed)
        syn_df = synthesize_one_table_aim(real_data[tname], eps_synth, delta, seed)
        syn_df = condition_fk_columns(
            schema,
            tname,
            syn_df,
            synthetic,
            seed,
            fk_hists=hists,
            fk_sampling="degree",
        )
        synthetic[tname] = syn_df
        logger.info("F table=%s eps_synth=%.4f eps_fk=%.4f", tname, eps_synth, eps_fk)
    return _repair(schema, synthetic, cfg)


def run_uniform_fk(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """U: convex (lp_rdp) + parent-first + uniform FK (histogram ablation)."""
    eps_alloc = _allocator(schema, real_data, cfg, epsilon, method="lp_rdp")
    logger.info("U lp_rdp alloc (eps_total=%.2f): %s", epsilon, eps_alloc)
    synth = SequentialSynthesizer(
        schema=schema,
        epsilon_alloc=eps_alloc,
        num_bins=cfg.get("synthesis", {}).get("num_bins", 32),
        max_parents_bn=cfg.get("synthesis", {}).get("max_parents_bn", 3),
        seed=seed,
        fk_rdp_share=_fk_rdp_share(cfg),
        fk_sampling="uniform",
    )
    synth.fit(real_data)
    return _repair(schema, synth.sample(real_data), cfg)


def run_convex_no_parent_first(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """C: convex lp_rdp allocation, independent tables (no parent-first)."""
    eps_alloc = _allocator(schema, real_data, cfg, epsilon, method="lp_rdp")
    logger.info("C lp_rdp alloc, no parent-first (eps_total=%.2f): %s", epsilon, eps_alloc)
    synthetic: dict[str, pd.DataFrame] = {}
    for tname in schema.topological_order():
        df = real_data[tname]
        table_meta = schema.tables[tname]
        engine = SingleTableSynthesizer(
            epsilon=float(eps_alloc.get(tname, epsilon)),
            num_bins=cfg.get("synthesis", {}).get("num_bins", 32),
            max_parents=cfg.get("synthesis", {}).get("max_parents_bn", 3),
            seed=seed,
        )
        engine.fit(df, table_meta.dtypes)
        synthetic[tname] = engine.sample(len(df))
    return _repair(schema, synthetic, cfg)


def run_privlava_p(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """P: official PrivLava extra only."""
    return run_privlava_optional(real_data, schema, cfg, epsilon, seed)


TIFS_RUNNERS = {
    "F": run_parent_first_aim_equal,
    "parent_first_aim_equal": run_parent_first_aim_equal,
    "aim_parent_first_equal": run_parent_first_aim_equal,
    "U": run_uniform_fk,
    "uniform_fk": run_uniform_fk,
    "aim_parent_first_uniform_fk": run_uniform_fk,
    "C": run_convex_no_parent_first,
    "convex_no_parent_first": run_convex_no_parent_first,
    "P": run_privlava_p,
    "privlava": run_privlava_p,
    "privlava_optional": run_privlava_p,
}
