"""Schema / volume dump: K, L, n_i, fanout, join-clip."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.schema import RelationalSchema


def longest_fk_path_edges(schema: RelationalSchema) -> int:
    """L = number of FK edges on the longest root-to-leaf path."""
    children: dict[str, list[str]] = {t: [] for t in schema.tables}
    for fk in schema.foreign_keys:
        if fk.child_table not in children[fk.parent_table]:
            children[fk.parent_table].append(fk.child_table)

    memo: dict[str, int] = {}

    def _depth(node: str) -> int:
        if node in memo:
            return memo[node]
        kids = children.get(node, [])
        memo[node] = (1 + max(_depth(k) for k in kids)) if kids else 0
        return memo[node]

    if not schema.tables:
        return 0
    return max(_depth(t) for t in schema.tables)


def max_fanout(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
) -> dict[str, float]:
    """Per-FK mean/max child rows per parent PK, plus a global max."""
    out: dict[str, float] = {}
    global_max = 0.0
    for fk in schema.foreign_keys:
        child = real_data.get(fk.child_table)
        parent = real_data.get(fk.parent_table)
        if child is None or parent is None:
            continue
        if fk.child_col not in child.columns or fk.parent_col not in parent.columns:
            continue
        counts = child[fk.child_col].value_counts()
        mx = float(counts.max()) if len(counts) else 0.0
        mean = float(counts.mean()) if len(counts) else 0.0
        key = f"{fk.child_table}.{fk.child_col}->{fk.parent_table}"
        out[f"{key}.max"] = mx
        out[f"{key}.mean"] = mean
        global_max = max(global_max, mx)
    out["max_fanout"] = global_max
    return out


def dump_dataset_stats(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    dataset: str,
    join_clip_mult: float = 10.0,
) -> dict[str, Any]:
    """K tables, L join depth, per-table n_i, FK edges, fanout, join clip."""
    n_i = {t: int(len(real_data[t])) for t in schema.tables if t in real_data}
    fact_name = max(n_i, key=n_i.get) if n_i else ""
    stats: dict[str, Any] = {
        "dataset": dataset,
        "K": int(schema.num_tables),
        "L": int(longest_fk_path_edges(schema)),
        "fk_edges": int(len(schema.foreign_keys)),
        "total_rows": int(sum(n_i.values())),
        "fact_table": fact_name,
        "fact_rows": int(n_i.get(fact_name, 0)),
        "n_i": n_i,
        "join_clip_mult": float(join_clip_mult),
        "join_clip": f"{join_clip_mult:g}x real join",
        "multi_parent_children": sorted(
            {
                t
                for t in schema.tables
                if len(schema.parent_tables(t)) >= 2
            }
        ),
    }
    stats.update(max_fanout(real_data, schema))
    return stats
