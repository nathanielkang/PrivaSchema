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
from src.errors import OptionalExtraError
from src.evaluation.dataset_stats import dump_dataset_stats
from src.evaluation.metrics import evaluate_all
from src.evaluation.writers import write_results_with_seed_std
from src.integrity_enforcer import MinCostRepairer
from src.schema import RelationalSchema
from src.sequential_synthesizer import SequentialSynthesizer
from src.tifs_methods import TIFS_RUNNERS

try:
    from src.baselines import run_named as _run_named_baselines
    from src.baselines.common import BaselineNotAvailableError
    from src.baselines.privaschema_aim import run_privaschema_aim
except ImportError:  # public tree does not ship competitor clones
    _run_named_baselines = None  # type: ignore[assignment]
    BaselineNotAvailableError = OptionalExtraError  # type: ignore[misc,assignment]
    run_privaschema_aim = None  # type: ignore[assignment]

FAKE_NAMED_ROWS = frozenset({"privpetal", "grdm"})

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
        method=cfg["allocator"].get("method", "lp_rdp"),
        query_weight_decay=cfg["allocator"].get("query_weight_decay", 0.9),
        table_sizes=table_sizes,
        workload_mode=cfg["allocator"].get("workload_mode", "paper"),
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
        fk_rdp_share=cfg["synthesis"].get(
            "fk_rdp_share", cfg["synthesis"].get("fk_hist_frac", 0.25)
        ),
        fk_sampling=cfg["synthesis"].get("fk_sampling", "degree"),
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
        method="equal_rdp",
    )
    eps_alloc = allocator.allocate()

    synthesizer = SequentialSynthesizer(
        schema=schema, epsilon_alloc=eps_alloc,
        num_bins=cfg["synthesis"].get("num_bins", 32),
        max_parents_bn=cfg["synthesis"].get("max_parents_bn", 3),
        seed=seed,
        fk_rdp_share=cfg["synthesis"].get(
            "fk_rdp_share", cfg["synthesis"].get("fk_hist_frac", 0.25)
        ),
        fk_sampling=cfg["synthesis"].get("fk_sampling", "degree"),
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
    eps_per_table = epsilon / np.sqrt(max(n_tables, 1))
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


def _dispatch(
    method_name: str,
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int,
    local_runners: dict,
):
    """Local F/U/C/P first, then optional competitor registry."""
    if method_name in TIFS_RUNNERS:
        return TIFS_RUNNERS[method_name](real_data, schema, cfg, epsilon, seed)
    if method_name in local_runners and local_runners[method_name] is not None:
        return local_runners[method_name](real_data, schema, cfg, epsilon, seed)
    if _run_named_baselines is not None:
        return _run_named_baselines(
            method_name, real_data, schema, cfg, epsilon, seed,
            local_runners=local_runners,
        )
    raise OptionalExtraError(
        f"Unknown method {method_name!r} in the public tree. "
        f"Known: {sorted(set(TIFS_RUNNERS) | set(local_runners))}"
    )


def _eval_bundle(real_data, synthetic_data, schema, cfg, seed):
    ev = cfg.get("evaluation", {})
    return evaluate_all(
        real_data, synthetic_data, schema,
        num_queries=ev.get("num_queries", 100),
        query_types=ev.get("query_types", ["1way", "2way", "cross_table"]),
        test_fraction=ev.get("ml_test_fraction", 0.2),
        seed=seed,
        ml_model=ev.get("ml_model", "lightgbm"),
        join_clip_mult=float(ev.get("join_clip_mult", 10.0)),
        run_membership_inference=bool(ev.get("membership_inference", False)),
    )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run PrivaSchema experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output", type=str, default="results/")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset list")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method names (default: baselines.methods in YAML)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (dataset,method,epsilon,run) already present in experiment_results.csv",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "experiment_results.csv"
    json_path = output_dir / "experiment_results.json"

    datasets = [args.dataset] if args.dataset else cfg["experiment"]["datasets"]
    epsilons = cfg["experiment"]["epsilons"]
    num_runs = cfg["experiment"]["num_runs"]
    base_seed = cfg["experiment"]["seed"]

    local_runners = {
        "privaschema": run_privaschema,
        "equal_split": run_equal_split,
        "independent": run_independent,
        **{k: v for k, v in TIFS_RUNNERS.items()},
    }
    if run_privaschema_aim is not None:
        local_runners["privaschema_aim"] = run_privaschema_aim
    if args.methods:
        method_names = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        method_names = list(
            cfg.get("baselines", {}).get(
                "methods",
                ["privaschema", "equal_split", "independent", "sdv_hma", "aim_repair"],
            )
        )

    # Non-private ceilings ignore epsilon sweep (run once per dataset)
    non_private = set(cfg.get("baselines", {}).get("non_private_methods", ["sdv_hma"]))
    allow_fake = bool(cfg.get("baselines", {}).get("allow_named_fallbacks", False))
    if not allow_fake:
        blocked = [m for m in method_names if m in FAKE_NAMED_ROWS]
        if blocked:
            logger.warning(
                "Dropping named fallback methods %s (allow_named_fallbacks=false). "
                "Real clones only, or omit from Table I.",
                blocked,
            )
            method_names = [m for m in method_names if m not in FAKE_NAMED_ROWS]

    all_results: list[dict] = []
    done_keys: set[tuple] = set()
    if args.resume and out_path.is_file():
        prev = pd.read_csv(out_path)
        all_results = prev.to_dict(orient="records")
        for row in all_results:
            done_keys.add(
                (
                    str(row["dataset"]),
                    str(row["method"]),
                    float(row["epsilon"]) if np.isfinite(float(row["epsilon"])) else float("inf"),
                    int(row["run"]),
                )
            )
        logger.info("Resume: loaded %d existing rows from %s", len(all_results), out_path)

    def _save() -> None:
        write_results_with_seed_std(all_results, output_dir, stem="experiment_results")

    def _write_skip_stub(
        method_name: str,
        dataset: str,
        epsilon: float,
        run_idx: int,
        err: Exception,
    ) -> Path:
        """Record a missing optional extra. Never invent a numeric cell."""
        stub_dir = output_dir / "skipped"
        stub_dir.mkdir(parents=True, exist_ok=True)
        safe_eps = "inf" if not np.isfinite(float(epsilon)) else f"{float(epsilon):g}"
        stub_path = stub_dir / f"{method_name}_{dataset}_eps{safe_eps}_run{run_idx}.json"
        payload = {
            "status": "skipped",
            "method": method_name,
            "dataset": dataset,
            "epsilon": epsilon if np.isfinite(float(epsilon)) else None,
            "run": run_idx,
            "reason": str(err),
            "metrics": None,
            "note": (
                "Optional extra is not installed. No invented numbers. "
                "Estimates belong in the local manuscript ledger, not here. "
                "Install the official stack and rerun to measure this cell."
            ),
        }
        stub_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.warning("Skip stub written to %s", stub_path)
        return stub_path

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
        ds_stats = dump_dataset_stats(
            real_data, schema, ds_name,
            join_clip_mult=float(cfg.get("evaluation", {}).get("join_clip_mult", 10.0)),
        )
        stats_path = output_dir / "dataset_stats.json"
        prev_stats: list[dict] = []
        if stats_path.is_file():
            try:
                prev_stats = json.loads(stats_path.read_text(encoding="utf-8"))
                if not isinstance(prev_stats, list):
                    prev_stats = [prev_stats]
            except json.JSONDecodeError:
                prev_stats = []
        prev_stats = [s for s in prev_stats if s.get("dataset") != ds_name]
        prev_stats.append(ds_stats)
        stats_path.write_text(json.dumps(prev_stats, indent=2, default=str), encoding="utf-8")
        flat_stats = []
        for row in prev_stats:
            flat = {k: v for k, v in row.items() if k != "n_i"}
            for t, n in row.get("n_i", {}).items():
                flat[f"n_{t}"] = n
            flat_stats.append(flat)
        pd.DataFrame(flat_stats).to_csv(output_dir / "dataset_stats.csv", index=False)
        logger.info(
            "Dataset stats %s: K=%s L=%s rows=%s fanout=%s",
            ds_name, ds_stats["K"], ds_stats["L"], ds_stats["total_rows"],
            ds_stats.get("max_fanout"),
        )

        # Non-private methods: one seed, epsilon recorded as inf
        for method_name in method_names:
            if method_name not in non_private:
                continue
            key = (ds_name, method_name, float("inf"), 0)
            if key in done_keys:
                logger.info("  [%s] skip (resume)", method_name)
                continue
            logger.info("  [%s] non-private ceiling", method_name)
            t0 = time.time()
            try:
                synthetic_data = _dispatch(
                    method_name, real_data, schema, cfg, float("inf"),
                    base_seed, local_runners,
                )
            except (BaselineNotAvailableError, OptionalExtraError) as e:
                logger.warning("Skipping %s: %s", method_name, e)
                _write_skip_stub(method_name, ds_name, float("inf"), 0, e)
                continue
            except Exception as e:  # noqa: BLE001
                logger.exception("Failed %s: %s", method_name, e)
                continue
            if synthetic_data is None:
                continue
            elapsed = time.time() - t0
            eval_result = _eval_bundle(real_data, synthetic_data, schema, cfg, base_seed)
            all_results.append({
                "dataset": ds_name,
                "method": method_name,
                "epsilon": float("inf"),
                "run": 0,
                "time_sec": round(elapsed, 2),
                **_flatten(eval_result),
            })
            done_keys.add(key)
            _save()

        dp_methods = [m for m in method_names if m not in non_private]
        for epsilon in epsilons:
            for run_idx in range(num_runs):
                seed = base_seed + run_idx
                for method_name in dp_methods:
                    key = (ds_name, method_name, float(epsilon), int(run_idx))
                    if key in done_keys:
                        logger.info(
                            "  [%s] eps=%.2f run=%d skip (resume)",
                            method_name, epsilon, run_idx + 1,
                        )
                        continue
                    logger.info(
                        "  [%s] eps=%.2f run=%d/%d",
                        method_name, epsilon, run_idx + 1, num_runs,
                    )
                    t0 = time.time()
                    try:
                        synthetic_data = _dispatch(
                            method_name, real_data, schema, cfg, epsilon, seed,
                            local_runners,
                        )
                    except (BaselineNotAvailableError, OptionalExtraError) as e:
                        logger.warning("Skipping %s: %s", method_name, e)
                        _write_skip_stub(method_name, ds_name, float(epsilon), int(run_idx), e)
                        continue
                    except Exception as e:  # noqa: BLE001
                        logger.exception("Failed %s eps=%.2f: %s", method_name, epsilon, e)
                        continue
                    if synthetic_data is None:
                        continue
                    elapsed = time.time() - t0

                    eval_result = _eval_bundle(
                        real_data, synthetic_data, schema, cfg, seed,
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
                    done_keys.add(key)
                    _save()
                    logger.info(
                        "    -> %s",
                        {k: v for k, v in record.items() if k not in ("dataset", "method")},
                    )

    _save()
    results_df = pd.DataFrame(all_results)
    logger.info("Results saved to %s", out_path)
    logger.info("JSON results saved to %s", json_path)
    _print_summary(results_df)
    (output_dir / "CAMPAIGN_DONE").write_text("ok\n", encoding="utf-8")



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
        "query_accuracy.query_error_cross_table",
        "query_accuracy.query_error_marginal",
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
