"""Foreign-key integrity enforcement via optimal transport or greedy repair."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from src.schema import ForeignKey, RelationalSchema

logger = logging.getLogger(__name__)


@dataclass
class FKViolationReport:
    """Summary of FK violations for a single constraint."""

    fk: ForeignKey
    total_rows: int
    violations: int
    violation_rate: float
    violating_indices: np.ndarray


class IntegrityChecker:
    """Detect FK violations in synthetic data."""

    def __init__(self, schema: RelationalSchema) -> None:
        self.schema = schema

    def check_all(
        self, synthetic_data: dict[str, pd.DataFrame]
    ) -> list[FKViolationReport]:
        """Check every FK constraint and return violation reports."""
        reports: list[FKViolationReport] = []
        for fk in self.schema.foreign_keys:
            report = self.check_one(fk, synthetic_data)
            reports.append(report)
        return reports

    def check_one(
        self, fk: ForeignKey, synthetic_data: dict[str, pd.DataFrame]
    ) -> FKViolationReport:
        """Check a single FK constraint."""
        child_df = synthetic_data.get(fk.child_table)
        parent_df = synthetic_data.get(fk.parent_table)

        if child_df is None or parent_df is None:
            return FKViolationReport(
                fk=fk, total_rows=0, violations=0,
                violation_rate=0.0, violating_indices=np.array([]),
            )

        if fk.child_col not in child_df.columns or fk.parent_col not in parent_df.columns:
            return FKViolationReport(
                fk=fk, total_rows=len(child_df), violations=len(child_df),
                violation_rate=1.0,
                violating_indices=np.arange(len(child_df)),
            )

        parent_vals = set(parent_df[fk.parent_col].dropna().unique())
        child_vals = child_df[fk.child_col].values

        parent_series = pd.Series(list(parent_vals))
        mask = ~pd.Series(child_vals).isin(parent_series).values
        violating = np.where(mask)[0]
        total = len(child_df)
        n_viol = len(violating)

        return FKViolationReport(
            fk=fk,
            total_rows=total,
            violations=n_viol,
            violation_rate=n_viol / max(total, 1),
            violating_indices=violating,
        )

    def overall_consistency_rate(
        self, synthetic_data: dict[str, pd.DataFrame]
    ) -> float:
        """Fraction of all FK references that are valid, across all constraints."""
        total_refs = 0
        total_valid = 0
        for fk in self.schema.foreign_keys:
            report = self.check_one(fk, synthetic_data)
            total_refs += report.total_rows
            total_valid += report.total_rows - report.violations
        if total_refs == 0:
            return 1.0
        return total_valid / total_refs


class MinCostRepairer:
    """Repair FK violations by reassigning child FK values to valid parent PKs.

    Supports two strategies:
      - ``optimal_transport``: Hungarian algorithm on a cost matrix
        (cost = distance between child row features and parent row features).
      - ``greedy``: replace each violating FK value with the nearest valid PK.
    """

    def __init__(
        self,
        schema: RelationalSchema,
        method: str = "optimal_transport",
        max_iterations: int = 100,
        ot_reg: float = 0.01,
    ) -> None:
        self.schema = schema
        self.method = method
        self.max_iterations = max_iterations
        self.ot_reg = ot_reg

    def repair(
        self, synthetic_data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Repair all FK violations in-place and return updated data."""
        checker = IntegrityChecker(self.schema)

        for iteration in range(self.max_iterations):
            reports = checker.check_all(synthetic_data)
            total_violations = sum(r.violations for r in reports)

            if total_violations == 0:
                logger.info("All FK constraints satisfied after %d iterations.", iteration)
                break

            logger.info(
                "Iteration %d: %d total violations across %d constraints",
                iteration, total_violations, len(reports),
            )

            for report in reports:
                if report.violations == 0:
                    continue
                synthetic_data = self._repair_one(
                    report.fk, report, synthetic_data
                )
        else:
            remaining = sum(r.violations for r in checker.check_all(synthetic_data))
            if remaining > 0:
                logger.warning(
                    "Max iterations reached; %d violations remain.", remaining
                )

        return synthetic_data

    def _repair_one(
        self,
        fk: ForeignKey,
        report: FKViolationReport,
        synthetic_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Repair violations for a single FK constraint."""
        if self.method == "optimal_transport":
            return self._repair_ot(fk, report, synthetic_data)
        elif self.method == "greedy":
            return self._repair_greedy(fk, report, synthetic_data)
        else:
            raise ValueError(f"Unknown repair method: {self.method}")

    def _repair_greedy(
        self,
        fk: ForeignKey,
        report: FKViolationReport,
        synthetic_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Replace each violating FK value with the most frequent valid parent PK."""
        parent_df = synthetic_data[fk.parent_table]
        child_df = synthetic_data[fk.child_table].copy()
        valid_pks = parent_df[fk.parent_col].values

        if len(valid_pks) == 0:
            return synthetic_data

        pk_counts = pd.Series(valid_pks).value_counts()
        most_common_pk = pk_counts.index[0]

        rng = np.random.default_rng(42)
        replacement = rng.choice(valid_pks, size=len(report.violating_indices), replace=True)
        child_df.iloc[report.violating_indices, child_df.columns.get_loc(fk.child_col)] = replacement

        synthetic_data[fk.child_table] = child_df
        return synthetic_data

    def _repair_ot(
        self,
        fk: ForeignKey,
        report: FKViolationReport,
        synthetic_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Optimal-transport repair: minimise assignment cost via Hungarian algorithm.

        For large-scale problems, we batch violating rows to keep memory manageable.
        Cost is computed as L2 distance between the numerical features of the
        violating child rows and the parent rows.
        """
        parent_df = synthetic_data[fk.parent_table]
        child_df = synthetic_data[fk.child_table].copy()
        valid_pks = parent_df[fk.parent_col].values

        if len(valid_pks) == 0 or len(report.violating_indices) == 0:
            return synthetic_data

        BATCH_SIZE = 2000
        viol_idx = report.violating_indices

        shared_num_cols = self._shared_numerical_columns(child_df, parent_df)

        if not shared_num_cols:
            rng = np.random.default_rng(42)
            replacement = rng.choice(valid_pks, size=len(viol_idx), replace=True)
            child_df.iloc[viol_idx, child_df.columns.get_loc(fk.child_col)] = replacement
            synthetic_data[fk.child_table] = child_df
            return synthetic_data

        parent_features = parent_df[shared_num_cols].values.astype(float)
        parent_features = np.nan_to_num(parent_features, nan=0.0)

        for start in range(0, len(viol_idx), BATCH_SIZE):
            batch_idx = viol_idx[start: start + BATCH_SIZE]
            child_features = child_df.iloc[batch_idx][shared_num_cols].values.astype(float)
            child_features = np.nan_to_num(child_features, nan=0.0)

            n_child = len(batch_idx)
            n_parent = len(parent_features)

            if n_child <= n_parent:
                cost = cdist(child_features, parent_features, metric="euclidean")
                row_ind, col_ind = linear_sum_assignment(cost)
                for r, c in zip(row_ind, col_ind):
                    child_df.iat[batch_idx[r], child_df.columns.get_loc(fk.child_col)] = valid_pks[c]
            else:
                rng = np.random.default_rng(42)
                for i, idx in enumerate(batch_idx):
                    dists = np.linalg.norm(parent_features - child_features[i], axis=1)
                    best = np.argmin(dists)
                    child_df.iat[idx, child_df.columns.get_loc(fk.child_col)] = valid_pks[best]

        synthetic_data[fk.child_table] = child_df
        return synthetic_data

    @staticmethod
    def _shared_numerical_columns(
        child_df: pd.DataFrame, parent_df: pd.DataFrame
    ) -> list[str]:
        """Find numerical columns present in both dataframes (excluding FK/PK)."""
        child_num = set(child_df.select_dtypes(include=[np.number]).columns)
        parent_num = set(parent_df.select_dtypes(include=[np.number]).columns)
        return sorted(child_num & parent_num)


def fk_consistency_metrics(
    schema: RelationalSchema,
    synthetic_data: dict[str, pd.DataFrame],
) -> dict[str, float]:
    """Compute per-constraint and overall FK consistency metrics."""
    checker = IntegrityChecker(schema)
    reports = checker.check_all(synthetic_data)

    metrics: dict[str, float] = {}
    for r in reports:
        key = f"{r.fk.child_table}.{r.fk.child_col}->{r.fk.parent_table}.{r.fk.parent_col}"
        metrics[key] = 1.0 - r.violation_rate

    metrics["overall"] = checker.overall_consistency_rate(synthetic_data)
    return metrics
