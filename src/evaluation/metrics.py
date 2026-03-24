"""Evaluation metrics for multi-table synthetic data quality."""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.integrity_enforcer import IntegrityChecker, fk_consistency_metrics
from src.schema import RelationalSchema

logger = logging.getLogger(__name__)


# ======================================================================
# Cross-table SQL-style query accuracy
# ======================================================================

class QueryAccuracyEvaluator:
    """Generate random aggregate queries and compare results on real vs. synthetic data."""

    def __init__(
        self,
        schema: RelationalSchema,
        num_queries: int = 100,
        query_types: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.schema = schema
        self.num_queries = num_queries
        self.query_types = query_types or ["1way", "2way", "cross_table"]
        self.rng = np.random.default_rng(seed)

    def evaluate(
        self,
        real_data: dict[str, pd.DataFrame],
        synthetic_data: dict[str, pd.DataFrame],
    ) -> dict[str, float]:
        """Return average relative error across random queries per type."""
        results: dict[str, list[float]] = {qt: [] for qt in self.query_types}

        for _ in range(self.num_queries):
            qtype = self.rng.choice(self.query_types)
            error = self._run_query(qtype, real_data, synthetic_data)
            if error is not None:
                results[qtype].append(error)

        metrics: dict[str, float] = {}
        all_errors: list[float] = []
        for qt, errors in results.items():
            if errors:
                metrics[f"query_error_{qt}"] = float(np.mean(errors))
                all_errors.extend(errors)
            else:
                metrics[f"query_error_{qt}"] = float("nan")

        metrics["query_error_avg"] = float(np.mean(all_errors)) if all_errors else float("nan")
        return metrics

    def _run_query(
        self,
        qtype: str,
        real_data: dict[str, pd.DataFrame],
        synthetic_data: dict[str, pd.DataFrame],
    ) -> float | None:
        """Run a single random query and return relative error."""
        tables = list(self.schema.tables.keys())

        if qtype in ("1way", "2way", "3way"):
            return self._single_table_marginal_query(qtype, tables, real_data, synthetic_data)
        elif qtype == "cross_table":
            return self._cross_table_query(tables, real_data, synthetic_data)
        return None

    def _single_table_marginal_query(
        self,
        qtype: str,
        tables: list[str],
        real_data: dict[str, pd.DataFrame],
        synthetic_data: dict[str, pd.DataFrame],
    ) -> float | None:
        """k-way marginal count on a single table."""
        order = {"1way": 1, "2way": 2, "3way": 3}[qtype]
        tname = self.rng.choice(tables)
        table_meta = self.schema.tables[tname]
        cat_cols = table_meta.categorical_columns()

        if len(cat_cols) < order:
            return None

        chosen = list(self.rng.choice(cat_cols, size=min(order, len(cat_cols)), replace=False))

        real_df = real_data.get(tname)
        syn_df = synthetic_data.get(tname)
        if real_df is None or syn_df is None:
            return None

        valid_cols = [c for c in chosen if c in real_df.columns and c in syn_df.columns]
        if not valid_cols:
            return None

        real_counts = real_df.groupby(valid_cols).size()
        syn_counts = syn_df.groupby(valid_cols).size()

        all_keys = real_counts.index.union(syn_counts.index, sort=False)
        real_vec = np.array([real_counts.get(k, 0) for k in all_keys], dtype=float)
        syn_vec = np.array([syn_counts.get(k, 0) for k in all_keys], dtype=float)

        real_total = real_vec.sum()
        syn_total = syn_vec.sum()
        if real_total == 0:
            return None

        real_freq = real_vec / real_total
        syn_freq = syn_vec / max(syn_total, 1)

        return float(np.mean(np.abs(real_freq - syn_freq)))

    def _cross_table_query(
        self,
        tables: list[str],
        real_data: dict[str, pd.DataFrame],
        synthetic_data: dict[str, pd.DataFrame],
    ) -> float | None:
        """Cross-table join + aggregate query along a FK relationship."""
        if not self.schema.foreign_keys:
            return None

        fk = self.schema.foreign_keys[int(self.rng.integers(0, len(self.schema.foreign_keys)))]
        child, parent = fk.child_table, fk.parent_table

        real_child = real_data.get(child)
        real_parent = real_data.get(parent)
        syn_child = synthetic_data.get(child)
        syn_parent = synthetic_data.get(parent)

        if any(d is None for d in [real_child, real_parent, syn_child, syn_parent]):
            return None

        if fk.child_col not in real_child.columns or fk.parent_col not in real_parent.columns:
            return None

        parent_meta = self.schema.tables[parent]
        num_cols = [c for c in parent_meta.numerical_columns()
                    if c in real_parent.columns and c != fk.parent_col]

        if not num_cols:
            real_joined = real_child.merge(
                real_parent[[fk.parent_col]], left_on=fk.child_col, right_on=fk.parent_col, how="inner",
            )
            syn_joined = syn_child.merge(
                syn_parent[[fk.parent_col]], left_on=fk.child_col, right_on=fk.parent_col, how="inner",
            )
            real_val = len(real_joined)
            syn_val = len(syn_joined)
        else:
            agg_col = str(self.rng.choice(num_cols))
            parent_cols = [fk.parent_col]
            if agg_col != fk.parent_col:
                parent_cols.append(agg_col)
            real_joined = real_child.merge(
                real_parent[parent_cols],
                left_on=fk.child_col, right_on=fk.parent_col, how="inner",
            )
            syn_joined = syn_child.merge(
                syn_parent[parent_cols],
                left_on=fk.child_col, right_on=fk.parent_col, how="inner",
            )
            if agg_col not in real_joined.columns or agg_col not in syn_joined.columns:
                return None
            real_val = float(pd.to_numeric(real_joined[agg_col], errors="coerce").mean()) if len(real_joined) > 0 else 0.0
            syn_val = float(pd.to_numeric(syn_joined[agg_col], errors="coerce").mean()) if len(syn_joined) > 0 else 0.0

        denom = abs(real_val) if abs(real_val) > 1e-9 else 1.0
        return float(abs(real_val - syn_val) / denom)


# ======================================================================
# Single-table statistical fidelity
# ======================================================================

def marginal_comparison(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    dtypes: dict[str, str],
    num_bins: int = 50,
) -> dict[str, float]:
    """Compare 1-way marginals between real and synthetic tables.

    Returns per-column total-variation distance and an average.
    """
    tv_distances: dict[str, float] = {}
    for col in real_df.columns:
        if col not in syn_df.columns:
            continue

        if dtypes.get(col) == "numerical":
            r = real_df[col].dropna().values.astype(float)
            s = syn_df[col].dropna().values.astype(float)
            lo = min(r.min(), s.min()) if len(r) > 0 and len(s) > 0 else 0
            hi = max(r.max(), s.max()) if len(r) > 0 and len(s) > 0 else 1
            edges = np.linspace(lo, hi, num_bins + 1)
            real_hist, _ = np.histogram(r, bins=edges, density=True)
            syn_hist, _ = np.histogram(s, bins=edges, density=True)
            bin_width = edges[1] - edges[0] if edges[1] - edges[0] > 0 else 1.0
            tv = 0.5 * np.sum(np.abs(real_hist - syn_hist)) * bin_width
        else:
            real_counts = real_df[col].value_counts(normalize=True)
            syn_counts = syn_df[col].value_counts(normalize=True)
            all_cats = set(real_counts.index) | set(syn_counts.index)
            tv = 0.5 * sum(
                abs(real_counts.get(c, 0) - syn_counts.get(c, 0)) for c in all_cats
            )

        tv_distances[col] = float(tv)

    tv_distances["average"] = float(np.mean(list(tv_distances.values()))) if tv_distances else 0.0
    return tv_distances


# ======================================================================
# ML utility (train-on-synthetic, test-on-real)
# ======================================================================

def ml_utility(
    real_data: dict[str, pd.DataFrame],
    synthetic_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    target_table: str | None = None,
    target_column: str | None = None,
    test_fraction: float = 0.3,
    seed: int = 42,
) -> dict[str, float]:
    """Train-on-synthetic, test-on-real evaluation.

    Joins all tables via FK relationships, picks a categorical target column,
    trains a RandomForest on synthetic joined data, tests on real joined data.
    """
    real_joined = _join_all(real_data, schema)
    syn_joined = _join_all(synthetic_data, schema)

    if real_joined.empty or syn_joined.empty:
        return {"accuracy_syn": 0.0, "accuracy_real": 0.0, "f1_syn": 0.0, "f1_real": 0.0}

    if target_table and target_column:
        target = target_column
    else:
        target = _pick_target(real_joined, schema)

    if target is None or target not in real_joined.columns or target not in syn_joined.columns:
        logger.warning("No suitable target column found for ML utility.")
        return {"accuracy_syn": 0.0, "accuracy_real": 0.0, "f1_syn": 0.0, "f1_real": 0.0}

    feature_cols = [c for c in real_joined.columns if c != target and c in syn_joined.columns]
    if not feature_cols:
        return {"accuracy_syn": 0.0, "accuracy_real": 0.0, "f1_syn": 0.0, "f1_real": 0.0}

    le = LabelEncoder()
    real_y = le.fit_transform(real_joined[target].astype(str).values)
    syn_y_raw = syn_joined[target].astype(str).values
    valid_mask = np.isin(syn_y_raw, le.classes_)
    syn_joined_filtered = syn_joined[valid_mask].copy()
    syn_y = le.transform(syn_joined_filtered[target].astype(str).values)

    real_X = _encode_features(real_joined[feature_cols])
    syn_X = _encode_features(syn_joined_filtered[feature_cols])

    if len(syn_X) < 10 or len(real_X) < 10:
        return {"accuracy_syn": 0.0, "accuracy_real": 0.0, "f1_syn": 0.0, "f1_real": 0.0}

    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        real_X, real_y, test_size=test_fraction, random_state=seed, stratify=real_y
    )

    clf_syn = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf_syn.fit(syn_X, syn_y)
    pred_syn = clf_syn.predict(X_test_real)

    clf_real = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf_real.fit(X_train_real, y_train_real)
    pred_real = clf_real.predict(X_test_real)

    return {
        "accuracy_syn": float(accuracy_score(y_test_real, pred_syn)),
        "accuracy_real": float(accuracy_score(y_test_real, pred_real)),
        "f1_syn": float(f1_score(y_test_real, pred_syn, average="weighted", zero_division=0)),
        "f1_real": float(f1_score(y_test_real, pred_real, average="weighted", zero_division=0)),
    }


# ======================================================================
# Aggregate evaluation
# ======================================================================

def evaluate_all(
    real_data: dict[str, pd.DataFrame],
    synthetic_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    num_queries: int = 100,
    query_types: list[str] | None = None,
    test_fraction: float = 0.3,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all evaluation metrics and return a combined report."""
    results: dict[str, Any] = {}

    fk_metrics = fk_consistency_metrics(schema, synthetic_data)
    results["fk_consistency"] = fk_metrics

    qeval = QueryAccuracyEvaluator(schema, num_queries, query_types, seed)
    results["query_accuracy"] = qeval.evaluate(real_data, synthetic_data)

    marginals: dict[str, dict[str, float]] = {}
    for tname, table in schema.tables.items():
        if tname in real_data and tname in synthetic_data:
            marginals[tname] = marginal_comparison(
                real_data[tname], synthetic_data[tname], table.dtypes
            )
    results["marginal_tv"] = marginals

    table_avgs = [m.get("average", 0.0) for m in marginals.values()]
    results["marginal_tv_avg"] = float(np.mean(table_avgs)) if table_avgs else 0.0

    results["ml_utility"] = ml_utility(
        real_data, synthetic_data, schema,
        test_fraction=test_fraction, seed=seed,
    )

    return results


# ======================================================================
# Internal helpers
# ======================================================================

def _join_all(
    data: dict[str, pd.DataFrame], schema: RelationalSchema,
    max_joined_rows: int = 50000,
) -> pd.DataFrame:
    """Join tables along a single FK chain to produce a flat table for ML.

    Follows one child per parent (the largest child table) to avoid
    combinatorial explosion from multiple one-to-many relationships.
    """
    order = schema.topological_order()
    if not order:
        return pd.DataFrame()

    chain = _find_longest_chain(schema, data)
    if not chain:
        first = order[0]
        return data[first].copy().head(max_joined_rows)

    joined = data[chain[0]].copy()
    if len(joined) > max_joined_rows:
        joined = joined.sample(n=max_joined_rows, random_state=42).reset_index(drop=True)

    for tname in chain[1:]:
        child_df = data.get(tname)
        if child_df is None:
            continue

        fk_match = None
        for fk in schema.foreign_keys:
            if fk.child_table == tname and fk.parent_col in joined.columns:
                fk_match = fk
                break
        if fk_match is None:
            continue

        child_sample = child_df
        if len(child_df) > max_joined_rows:
            child_sample = child_df.sample(n=max_joined_rows, random_state=42)

        suffix = f"_{tname}"
        overlap = set(joined.columns) & set(child_sample.columns) - {fk_match.child_col}
        child_renamed = child_sample.rename(columns={c: c + suffix for c in overlap})

        joined = joined.merge(
            child_renamed,
            left_on=fk_match.parent_col,
            right_on=fk_match.child_col,
            how="inner",
        )
        if len(joined) > max_joined_rows:
            joined = joined.sample(n=max_joined_rows, random_state=42).reset_index(drop=True)

    return joined


def _find_longest_chain(
    schema: RelationalSchema, data: dict[str, pd.DataFrame]
) -> list[str]:
    """Find the longest root-to-leaf FK chain in the schema."""
    order = schema.topological_order()
    if not order:
        return []

    children_of: dict[str, list[str]] = {t: [] for t in schema.tables}
    for fk in schema.foreign_keys:
        if fk.child_table not in children_of[fk.parent_table]:
            children_of[fk.parent_table].append(fk.child_table)

    roots = [t for t in order if not schema.parent_tables(t)]
    if not roots:
        return [order[0]]

    best_chain: list[str] = []
    for root in roots:
        chain = _dfs_longest(root, children_of, data)
        if len(chain) > len(best_chain):
            best_chain = chain

    return best_chain


def _dfs_longest(
    node: str,
    children_of: dict[str, list[str]],
    data: dict[str, pd.DataFrame],
) -> list[str]:
    """DFS to find the longest chain from node, preferring largest child."""
    kids = children_of.get(node, [])
    if not kids:
        return [node]

    best_sub: list[str] = []
    for kid in kids:
        sub = _dfs_longest(kid, children_of, data)
        if len(sub) > len(best_sub):
            best_sub = sub

    return [node] + best_sub


def _pick_target(df: pd.DataFrame, schema: RelationalSchema) -> str | None:
    """Heuristically pick a categorical target column for ML utility."""
    candidates: list[tuple[str, int]] = []
    for col in df.columns:
        nunique = df[col].nunique()
        if 2 <= nunique <= 20:
            candidates.append((col, nunique))

    if not candidates:
        for col in df.columns:
            if df[col].dtype == object:
                nunique = df[col].nunique()
                if nunique >= 2:
                    candidates.append((col, nunique))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def _encode_features(df: pd.DataFrame) -> np.ndarray:
    """Simple feature encoding: label-encode categoricals, fill NaN with 0."""
    encoded = pd.DataFrame(index=df.index)
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name == "category":
            le = LabelEncoder()
            encoded[col] = le.fit_transform(df[col].astype(str).fillna("__NA__"))
        else:
            encoded[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return encoded.values.astype(float)
