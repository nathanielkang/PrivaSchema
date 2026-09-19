"""CSV / JSON writers with three-seed mean and standard deviation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and s.name not in {
        "run",
        "seed",
        "time_sec",
    }


def aggregate_seed_stats(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...] = ("dataset", "method", "epsilon"),
) -> pd.DataFrame:
    """Mean and sample std over ``run`` / seed for each (dataset, method, ε)."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    group_cols = [c for c in keys if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    metric_cols = [c for c in df.columns if c not in group_cols and _is_numeric_series(df[c])]
    recs: list[dict[str, Any]] = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        rec = {col: key[i] for i, col in enumerate(group_cols)}
        rec["n_seeds"] = int(len(g))
        if "run" in g.columns:
            rec["n_seeds"] = int(g["run"].nunique())
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            rec[f"{col}_mean"] = float(np.nanmean(vals)) if len(vals) else float("nan")
            rec[f"{col}_std"] = (
                float(np.nanstd(vals, ddof=1)) if np.sum(np.isfinite(vals)) >= 2 else 0.0
            )
        recs.append(rec)
    return pd.DataFrame(recs)


def write_results_with_seed_std(
    rows: list[dict[str, Any]],
    output_dir: Path,
    stem: str = "experiment_results",
) -> tuple[Path, Path, Path, Path]:
    """Write per-run CSV/JSON plus aggregated mean/std CSV/JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    json_path = output_dir / f"{stem}.json"
    agg_csv = output_dir / f"{stem}_seed_stats.csv"
    agg_json = output_dir / f"{stem}_seed_stats.json"

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    agg = aggregate_seed_stats(rows)
    agg.to_csv(agg_csv, index=False)
    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump(agg.to_dict(orient="records"), f, indent=2, default=str)

    return csv_path, json_path, agg_csv, agg_json
