"""Parent-first sequential table synthesiser using noisy marginals (PrivBayes-style)."""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from src.schema import ForeignKey, RelationalSchema

logger = logging.getLogger(__name__)


# ======================================================================
# Single-table PrivBayes-style synthesiser
# ======================================================================

class SingleTableSynthesizer:
    """DP synthesiser for one table via noisy 2-way marginals + Bayesian network.

    Pipeline:
      1. Discretise continuous columns into ``num_bins`` equal-width bins.
      2. Estimate all 2-way marginals, add calibrated Laplace noise (epsilon).
      3. Greedily build a Bayesian network (chain) by selecting highest-MI pairs.
      4. Sample synthetic rows from the noisy conditional distributions.
    """

    def __init__(
        self,
        epsilon: float,
        num_bins: int = 32,
        max_parents: int = 3,
        seed: int = 42,
    ) -> None:
        self.epsilon = epsilon
        self.num_bins = num_bins
        self.max_parents = max_parents
        self.rng = np.random.default_rng(seed)

        self._bin_edges: dict[str, np.ndarray] = {}
        self._domain: dict[str, np.ndarray] = {}
        self._bn_order: list[str] = []
        self._bn_parents: dict[str, list[str]] = {}
        self._noisy_cpts: dict[str, np.ndarray] = {}
        self._columns: list[str] = []
        self._dtypes: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, dtypes: dict[str, str]) -> SingleTableSynthesizer:
        """Learn noisy marginals and build a Bayesian network."""
        self._columns = list(df.columns)
        self._dtypes = dtypes

        encoded = self._encode(df)
        n_rows = len(encoded)

        num_marginals = len(self._columns) * (len(self._columns) - 1) // 2
        num_marginals = max(num_marginals, 1)
        eps_per_marginal = self.epsilon / (num_marginals + len(self._columns))

        noisy_marginals = self._estimate_noisy_marginals(encoded, eps_per_marginal, n_rows)

        self._build_bayesian_network(noisy_marginals)

        self._build_cpts(encoded, eps_per_marginal, n_rows)

        return self

    def sample(self, n: int) -> pd.DataFrame:
        """Generate n synthetic rows from the learned Bayesian network (vectorised)."""
        sampled: dict[str, np.ndarray] = {}

        for col in self._bn_order:
            parents = self._bn_parents[col]
            cpt = self._noisy_cpts[col]
            d_col = len(self._domain[col])

            if not parents:
                probs = np.clip(cpt, 0, None)
                total = probs.sum()
                probs = probs / total if total > 0 else np.ones(d_col) / d_col
                sampled[col] = self.rng.choice(d_col, size=n, p=probs)
            else:
                result = np.empty(n, dtype=int)
                parent_dims = tuple(len(self._domain[p]) for p in parents)
                parent_data = np.column_stack([sampled[p] for p in parents])

                unique_combos, inverse = np.unique(parent_data, axis=0, return_inverse=True)
                for ui, combo in enumerate(unique_combos):
                    mask = inverse == ui
                    count = mask.sum()
                    idx = tuple(int(combo[j]) for j in range(len(parents)))
                    probs = np.clip(cpt[idx], 0, None)
                    total = probs.sum()
                    probs = probs / total if total > 0 else np.ones(d_col) / d_col
                    result[mask] = self.rng.choice(d_col, size=count, p=probs)

                sampled[col] = result

        encoded_df = pd.DataFrame(sampled, columns=self._bn_order)
        return self._decode(encoded_df)

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map every column to integer codes, capping cardinality at num_bins."""
        encoded = pd.DataFrame(index=df.index)
        for col in self._columns:
            if self._dtypes.get(col) == "numerical":
                vals = df[col].values.astype(float)
                edges = np.linspace(
                    np.nanmin(vals) - 1e-9,
                    np.nanmax(vals) + 1e-9,
                    self.num_bins + 1,
                )
                self._bin_edges[col] = edges
                codes = np.digitize(vals, edges[1:-1])
                self._domain[col] = np.arange(self.num_bins)
                encoded[col] = codes
            else:
                cats = df[col].astype("category")
                categories = cats.cat.categories.tolist()
                if len(categories) > self.num_bins:
                    top = df[col].value_counts().head(self.num_bins - 1).index.tolist()
                    mapping = {v: i for i, v in enumerate(top)}
                    other_idx = len(top)
                    codes = df[col].map(lambda x, m=mapping, o=other_idx: m.get(x, o)).values
                    self._domain[col] = np.arange(self.num_bins)
                    self._bin_edges[col] = top + ["__OTHER__"]
                    encoded[col] = codes
                else:
                    self._domain[col] = np.arange(len(categories))
                    self._bin_edges[col] = categories
                    encoded[col] = cats.cat.codes.values
        return encoded

    def _decode(self, encoded: pd.DataFrame) -> pd.DataFrame:
        """Map integer codes back to original domain."""
        decoded = pd.DataFrame(index=encoded.index)
        for col in self._columns:
            if col not in encoded.columns:
                continue
            codes = encoded[col].values.astype(int)
            if self._dtypes.get(col) == "numerical":
                edges = self._bin_edges[col]
                mid = (edges[:-1] + edges[1:]) / 2.0
                codes_clipped = np.clip(codes, 0, len(mid) - 1)
                decoded[col] = mid[codes_clipped]
            else:
                cats = self._bin_edges[col]
                codes_clipped = np.clip(codes, 0, len(cats) - 1)
                decoded[col] = [cats[c] for c in codes_clipped]
        return decoded

    # ------------------------------------------------------------------
    # Noisy marginal estimation
    # ------------------------------------------------------------------

    def _estimate_noisy_marginals(
        self,
        encoded: pd.DataFrame,
        eps_per_marginal: float,
        n_rows: int,
    ) -> dict[tuple[str, str], np.ndarray]:
        """Compute noisy 2-way marginals with Laplace noise (vectorised)."""
        marginals: dict[tuple[str, str], np.ndarray] = {}
        cols = list(encoded.columns)
        sensitivity = 1.0 / n_rows
        scale = sensitivity / max(eps_per_marginal, 1e-12)

        for i, c1 in enumerate(cols):
            v1 = np.clip(encoded[c1].values.astype(int), 0, len(self._domain[c1]) - 1)
            for c2 in cols[i + 1:]:
                d1 = len(self._domain[c1])
                d2 = len(self._domain[c2])
                v2 = np.clip(encoded[c2].values.astype(int), 0, d2 - 1)
                flat = v1 * d2 + v2
                hist = np.bincount(flat, minlength=d1 * d2).reshape(d1, d2).astype(float)
                hist /= n_rows
                noise = self.rng.laplace(0, scale, size=hist.shape)
                hist_noisy = np.clip(hist + noise, 0, None)
                total = hist_noisy.sum()
                if total > 0:
                    hist_noisy /= total
                marginals[(c1, c2)] = hist_noisy

        return marginals

    def _mutual_information(self, marginal: np.ndarray) -> float:
        """MI from a 2-way marginal table."""
        p_xy = marginal.copy()
        p_xy = np.clip(p_xy, 1e-15, None)
        p_x = p_xy.sum(axis=1, keepdims=True)
        p_y = p_xy.sum(axis=0, keepdims=True)
        log_term = np.log(p_xy / (p_x * p_y + 1e-15) + 1e-15)
        return float(np.sum(p_xy * log_term))

    # ------------------------------------------------------------------
    # Bayesian network construction
    # ------------------------------------------------------------------

    def _build_bayesian_network(
        self, marginals: dict[tuple[str, str], np.ndarray]
    ) -> None:
        """Greedy BN construction: iteratively pick highest-MI attribute."""
        cols = list(self._domain.keys())
        if not cols:
            return

        mi_scores: dict[tuple[str, str], float] = {}
        for pair, marg in marginals.items():
            mi_scores[pair] = self._mutual_information(marg)
            mi_scores[(pair[1], pair[0])] = mi_scores[pair]

        selected: list[str] = []
        remaining = set(cols)

        best_first = max(cols, key=lambda c: sum(
            mi_scores.get((c, other), 0) for other in cols if other != c
        ))
        selected.append(best_first)
        remaining.remove(best_first)
        self._bn_parents[best_first] = []

        while remaining:
            best_col = None
            best_score = -float("inf")
            best_parents: list[str] = []

            for c in remaining:
                parent_candidates = selected[-self.max_parents:]
                score = sum(mi_scores.get((c, p), 0) for p in parent_candidates)
                if score > best_score:
                    best_score = score
                    best_col = c
                    best_parents = parent_candidates[:]

            assert best_col is not None
            selected.append(best_col)
            remaining.remove(best_col)
            self._bn_parents[best_col] = best_parents

        self._bn_order = selected

    def _build_cpts(
        self,
        encoded: pd.DataFrame,
        eps_per_marginal: float,
        n_rows: int,
    ) -> None:
        """Build noisy conditional probability tables for each node (vectorised)."""
        sensitivity = 1.0 / n_rows
        scale = sensitivity / max(eps_per_marginal, 1e-12)

        for col in self._bn_order:
            parents = self._bn_parents[col]
            d_col = len(self._domain[col])

            if not parents:
                vals = np.clip(encoded[col].values.astype(int), 0, d_col - 1)
                hist = np.bincount(vals, minlength=d_col).astype(float) / n_rows
                noise = self.rng.laplace(0, scale, size=hist.shape)
                cpt = np.clip(hist + noise, 0, None)
                total = cpt.sum()
                if total > 0:
                    cpt /= total
                self._noisy_cpts[col] = cpt
            else:
                parent_dims = tuple(len(self._domain[p]) for p in parents)
                shape = parent_dims + (d_col,)
                total_parent_cells = int(np.prod(parent_dims))
                flat_size = total_parent_cells * d_col

                parent_vals = encoded[parents].values.astype(int)
                col_vals = np.clip(encoded[col].values.astype(int), 0, d_col - 1)

                for j in range(len(parents)):
                    parent_vals[:, j] = np.clip(parent_vals[:, j], 0, parent_dims[j] - 1)

                strides = np.ones(len(parents), dtype=int)
                for j in range(len(parents) - 2, -1, -1):
                    strides[j] = strides[j + 1] * parent_dims[j + 1]
                flat_parent = (parent_vals * strides).sum(axis=1)
                flat_idx = flat_parent * d_col + col_vals

                hist_flat = np.bincount(flat_idx, minlength=flat_size).astype(float)
                hist = hist_flat.reshape(shape)

                for idx in np.ndindex(parent_dims):
                    row = hist[idx]
                    row_sum = row.sum()
                    if row_sum > 0:
                        row /= row_sum
                    noise = self.rng.laplace(0, scale, size=row.shape)
                    row_noisy = np.clip(row + noise, 0, None)
                    total = row_noisy.sum()
                    if total > 0:
                        row_noisy /= total
                    else:
                        row_noisy = np.ones(d_col) / d_col
                    hist[idx] = row_noisy

                self._noisy_cpts[col] = hist


# ======================================================================
# Sequential multi-table synthesiser
# ======================================================================

class SequentialSynthesizer:
    """Generate synthetic multi-table data in topological (parent-first) order.

    For each child table, the FK column values are sampled from the synthetic
    parent's PK distribution, ensuring referential plausibility before the
    integrity enforcer's repair step.
    """

    def __init__(
        self,
        schema: RelationalSchema,
        epsilon_alloc: dict[str, float],
        num_bins: int = 32,
        max_parents_bn: int = 3,
        seed: int = 42,
    ) -> None:
        self.schema = schema
        self.epsilon_alloc = epsilon_alloc
        self.num_bins = num_bins
        self.max_parents_bn = max_parents_bn
        self.seed = seed
        self.synthesizers: dict[str, SingleTableSynthesizer] = {}

    def fit(self, real_data: dict[str, pd.DataFrame]) -> SequentialSynthesizer:
        """Fit per-table synthesisers in topological order."""
        order = self.schema.topological_order()
        logger.info("Topological order: %s", order)

        for tname in order:
            df = real_data[tname]
            table_meta = self.schema.tables[tname]
            eps = self.epsilon_alloc.get(tname, 0.1)
            logger.info("Fitting %s (eps=%.4f, rows=%d)", tname, eps, len(df))

            synth = SingleTableSynthesizer(
                epsilon=eps,
                num_bins=self.num_bins,
                max_parents=self.max_parents_bn,
                seed=self.seed,
            )
            synth.fit(df, table_meta.dtypes)
            self.synthesizers[tname] = synth

        return self

    def sample(
        self,
        real_data: dict[str, pd.DataFrame],
        synthetic_data: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Sample synthetic tables in topological order with FK conditioning."""
        order = self.schema.topological_order()
        if synthetic_data is None:
            synthetic_data = {}

        for tname in order:
            n_rows = len(real_data[tname])
            synth = self.synthesizers[tname]
            syn_df = synth.sample(n_rows)

            syn_df = self._condition_fk_columns(tname, syn_df, synthetic_data)
            synthetic_data[tname] = syn_df
            logger.info("Sampled %s: %d rows", tname, len(syn_df))

        return synthetic_data

    def _condition_fk_columns(
        self,
        table_name: str,
        syn_df: pd.DataFrame,
        synthetic_data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Replace FK column values with values drawn from synthetic parent PK."""
        for fk in self.schema.foreign_keys:
            if fk.child_table != table_name:
                continue
            if fk.parent_table not in synthetic_data:
                logger.warning(
                    "Parent %s not yet synthesised for FK %s->%s",
                    fk.parent_table, fk.child_table, fk.parent_table,
                )
                continue

            parent_syn = synthetic_data[fk.parent_table]
            if fk.parent_col not in parent_syn.columns:
                continue

            parent_pk_vals = parent_syn[fk.parent_col].values
            if len(parent_pk_vals) == 0:
                continue

            rng = np.random.default_rng(self.seed)
            sampled_fks = rng.choice(parent_pk_vals, size=len(syn_df), replace=True)

            if fk.child_col in syn_df.columns:
                syn_df[fk.child_col] = sampled_fks
            else:
                syn_df.insert(0, fk.child_col, sampled_fks)

        return syn_df
