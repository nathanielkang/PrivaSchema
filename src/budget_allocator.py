"""Privacy budget allocation across tables under a single global epsilon."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from src.schema import RelationalSchema

logger = logging.getLogger(__name__)


@dataclass
class WorkloadQuery:
    """Represents a workload query over one or more tables."""

    tables: list[str]
    order: int  # 1-way, 2-way, etc.
    weight: float = 1.0


class BudgetAllocator:
    """Allocate a global privacy budget across tables.

    Strategies:
      - ``lp_rdp``: manuscript Eq. (5) — min-max ``Σ w/ε_i`` under ``Σ ε_i² ≤ ε²``
      - ``equal_rdp``: ``ε_i = ε / √K`` (paper equal-split)
      - ``analytical``: closed-form under linear ``Σ ε_i = ε`` (legacy)
      - ``lp``: linear LP + sum-normalize (legacy)
      - ``equal``: ``ε_i = ε / K`` (legacy linear)
      - ``query_proportional``: involvement counts, then sum-normalize
    """

    def __init__(
        self,
        schema: RelationalSchema,
        epsilon_total: float,
        delta: float,
        epsilon_min: float = 0.01,
        method: str = "lp_rdp",
        query_weight_decay: float = 0.9,
        table_sizes: dict[str, int] | None = None,
        workload_mode: str = "paper",
    ) -> None:
        self.schema = schema
        self.epsilon_total = epsilon_total
        self.delta = delta
        self.epsilon_min = epsilon_min
        self.method = method
        self.query_weight_decay = query_weight_decay
        self.table_names = list(schema.tables.keys())
        self.n_tables = len(self.table_names)
        self.table_sizes = table_sizes or {}
        self.workload_mode = workload_mode

    def _effective_epsilon_min(self) -> float:
        """Floor: yaml ε_min, but at least 35% of equal-split √K (no starved tables)."""
        eq = self.epsilon_total / np.sqrt(max(self.n_tables, 1))
        yaml_cap = min(self.epsilon_min, 0.9 * eq)
        return float(max(yaml_cap, 0.35 * eq))

    def allocate(
        self, workload: list[WorkloadQuery] | None = None
    ) -> dict[str, float]:
        """Return per-table ε. ``lp_rdp`` / ``equal_rdp`` obey ``Σ ε_i² ≤ ε²``."""
        if self.method == "equal":
            return self._equal_split()
        elif self.method == "equal_rdp":
            return self._equal_rdp()
        elif self.method == "query_proportional":
            return self._query_proportional(workload or [])
        elif self.method == "lp":
            return self._lp_allocation(workload or [])
        elif self.method == "lp_rdp":
            return self._lp_rdp_allocation(workload or [])
        elif self.method == "analytical":
            return self._analytical_allocation(workload or [])
        else:
            raise ValueError(f"Unknown allocation method: {self.method}")

    # ------------------------------------------------------------------
    # Allocation strategies
    # ------------------------------------------------------------------

    def _equal_split(self) -> dict[str, float]:
        eps = self.epsilon_total / max(self.n_tables, 1)
        return {t: eps for t in self.table_names}

    def _equal_rdp(self) -> dict[str, float]:
        """Paper equal-split under quadratic composition: ε_i = ε / √K."""
        eps = self.epsilon_total / np.sqrt(max(self.n_tables, 1))
        return {t: float(eps) for t in self.table_names}

    def _lp_rdp_allocation(
        self, workload: list[WorkloadQuery]
    ) -> dict[str, float]:
        """Manuscript Eq. (5): min z s.t. Σ_{i∈S_j} w_j/ε_i ≤ z, Σ ε_i² ≤ ε², ε_i ≥ ε_min.

        Does **not** rescale to Σ ε_i = ε (that would break RDP composition).
        """
        from scipy.optimize import minimize

        K = self.n_tables
        if K == 0:
            return {}
        if not workload:
            return self._equal_rdp()

        eps_min = self._effective_epsilon_min()
        eps_tot = float(self.epsilon_total)
        names = self.table_names

        query_idx: list[tuple[list[int], float]] = []
        for q in workload:
            idxs = [names.index(t) for t in q.tables if t in names]
            if idxs:
                query_idx.append((idxs, float(q.weight)))
        if not query_idx:
            return self._equal_rdp()

        # Decision vector: [ε_1, …, ε_K, z]
        def objective(x: np.ndarray) -> float:
            return float(x[-1])

        def cons_quad(x: np.ndarray) -> float:
            return eps_tot ** 2 - float(np.dot(x[:-1], x[:-1]))

        constraints: list[dict] = [{"type": "ineq", "fun": cons_quad}]
        for idxs, w in query_idx:
            def _query_cons(x: np.ndarray, idxs=idxs, w=w) -> float:
                proxy = w * sum(1.0 / max(float(x[i]), 1e-12) for i in idxs)
                return float(x[-1]) - proxy

            constraints.append({"type": "ineq", "fun": _query_cons})

        warm = self._analytical_rdp(workload)
        x_eps = np.array([warm[n] for n in names], dtype=float)
        z0 = max(
            w * sum(1.0 / max(float(x_eps[i]), 1e-12) for i in idxs)
            for idxs, w in query_idx
        )
        x0 = np.concatenate([x_eps, np.array([z0], dtype=float)])
        bounds = [(eps_min, eps_tot) for _ in range(K)] + [(0.0, None)]
        try:
            res = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 600, "ftol": 1e-10},
            )
            if not res.success:
                logger.warning("lp_rdp SLSQP: %s; quadratic-weight fallback.", res.message)
                return self._analytical_rdp(workload)
            alloc = {names[i]: float(res.x[i]) for i in range(K)}
        except Exception as e:  # noqa: BLE001
            logger.warning("lp_rdp failed (%s); quadratic-weight fallback.", e)
            return self._analytical_rdp(workload)

        alloc = self._project_rdp_ball(alloc)
        ss = sum(v * v for v in alloc.values())
        logger.info("lp_rdp alloc (sumsq=%.4f, cap=%.4f): %s", ss, eps_tot ** 2, alloc)
        return alloc

    def _analytical_rdp(self, workload: list[WorkloadQuery]) -> dict[str, float]:
        """Same sensitivity weights as ``analytical``, projected onto ``||ε||_2 = ε``."""
        weights = self._table_sensitivity_weights(workload)
        raw = {t: w ** (1.0 / 3.0) for t, w in weights.items()}
        return self._project_rdp_ball(raw)

    def _table_sensitivity_weights(
        self, workload: list[WorkloadQuery]
    ) -> dict[str, float]:
        query_counts = {t: 0.0 for t in self.table_names}
        for q in workload:
            for t in q.tables:
                if t in query_counts:
                    query_counts[t] += q.weight
        weights: dict[str, float] = {}
        for t in self.table_names:
            n_attrs = max(len(self.schema.tables[t].attributes), 1)
            n_rows = max(self.table_sizes.get(t, 1), 1)
            qw = max(query_counts.get(t, 0.0), 0.01)
            weights[t] = qw * n_attrs * np.log1p(n_rows)
        return weights

    def _project_rdp_ball(self, raw: dict[str, float]) -> dict[str, float]:
        """Scale so Σ ε_i² ≤ ε² and ε_i ≥ ε_min. Never sum-normalizes to Σ ε_i = ε."""
        eps_min = self._effective_epsilon_min()
        eps_tot = float(self.epsilon_total)
        vec = np.array([max(float(raw.get(t, 0.0)), 1e-12) for t in self.table_names])
        nrm = float(np.linalg.norm(vec))
        if nrm < 1e-15:
            return self._equal_rdp()
        scaled = vec * (eps_tot / nrm)
        scaled = np.maximum(scaled, eps_min)
        nrm2 = float(np.linalg.norm(scaled))
        if nrm2 > eps_tot + 1e-12:
            scaled = scaled * (eps_tot / nrm2)
            scaled = np.maximum(scaled, 1e-12)
        return {t: float(scaled[i]) for i, t in enumerate(self.table_names)}

    def _query_proportional(
        self, workload: list[WorkloadQuery]
    ) -> dict[str, float]:
        """Allocate proportional to how often a table appears in workload queries."""
        counts = {t: 0.0 for t in self.table_names}
        for q in workload:
            for t in q.tables:
                if t in counts:
                    counts[t] += q.weight

        total_count = sum(counts.values())
        if total_count == 0:
            return self._equal_split()

        alloc: dict[str, float] = {}
        eps_min_adj = self._effective_epsilon_min()
        for t in self.table_names:
            raw = (counts[t] / total_count) * self.epsilon_total
            alloc[t] = max(raw, eps_min_adj)

        return self._normalize(alloc)

    def _analytical_allocation(
        self, workload: list[WorkloadQuery]
    ) -> dict[str, float]:
        """Closed-form workload-aware allocation (variance-optimal).

        Under basic composition (sum eps_t = eps_total), minimizing
        total weighted error sum_t c_t / eps_t^2 yields:

            eps_t = eps_total * c_t^{1/3} / sum_j c_j^{1/3}

        where c_t encodes the sensitivity weight of table t,
        combining query involvement, schema complexity, and data volume.
        """
        eps_min_adj = self._effective_epsilon_min()
        weights = self._table_sensitivity_weights(workload)

        alloc_raw = {t: w ** (1.0 / 3.0) for t, w in weights.items()}
        total_raw = sum(alloc_raw.values())
        if total_raw < 1e-15:
            return self._equal_split()

        alloc: dict[str, float] = {}
        for t in self.table_names:
            eps = self.epsilon_total * alloc_raw[t] / total_raw
            alloc[t] = max(eps, eps_min_adj)

        logger.info(
            "Analytical weights: %s",
            {t: round(w, 3) for t, w in weights.items()},
        )
        return self._normalize(alloc)

    def _lp_allocation(
        self, workload: list[WorkloadQuery]
    ) -> dict[str, float]:
        """LP formulation: maximise minimum per-query accuracy.

        Decision variables: eps_1, ..., eps_K, z  (K = #tables, z = auxiliary)

        Maximise z  subject to:
          For each query q over tables T_q:
              sum_{t in T_q} eps_t  >=  z * w_q        (accuracy proxy)
          sum eps_t <= epsilon_total                   (budget constraint)
          eps_t >= epsilon_min_adj  for all t          (adaptive minimum)
        """
        K = self.n_tables
        if K == 0:
            return {}

        n_vars = K + 1  # eps_1..eps_K, z
        eps_min_adj = self._effective_epsilon_min()

        c = np.zeros(n_vars)
        c[-1] = -1.0

        A_ub_rows: list[list[float]] = []
        b_ub_list: list[float] = []

        for q in workload:
            row = [0.0] * n_vars
            for t in q.tables:
                if t in self.table_names:
                    idx = self.table_names.index(t)
                    row[idx] = -1.0
            row[-1] = q.weight
            A_ub_rows.append(row)
            b_ub_list.append(0.0)

        budget_row = [0.0] * n_vars
        for i in range(K):
            budget_row[i] = 1.0
        A_ub_rows.append(budget_row)
        b_ub_list.append(self.epsilon_total)

        A_ub = np.array(A_ub_rows) if A_ub_rows else np.zeros((0, n_vars))
        b_ub = np.array(b_ub_list) if b_ub_list else np.zeros(0)

        bounds = [(eps_min_adj, self.epsilon_total) for _ in range(K)]
        bounds.append((0.0, self.epsilon_total))

        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        except Exception as e:
            logger.warning("LP solver error: %s; falling back to analytical.", e)
            return self._analytical_allocation(workload)

        if not result.success:
            logger.warning(
                "LP status %d (%s); falling back to analytical allocation.",
                result.status,
                getattr(result, "message", "unknown"),
            )
            return self._analytical_allocation(workload)

        alloc = {self.table_names[i]: float(result.x[i]) for i in range(K)}
        return self._normalize(alloc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize(self, alloc: dict[str, float]) -> dict[str, float]:
        """Rescale so allocations sum exactly to epsilon_total."""
        total = sum(alloc.values())
        if total < 1e-15:
            return self._equal_split()
        eps_min_adj = self._effective_epsilon_min()
        scale = self.epsilon_total / total
        normed = {t: max(e * scale, eps_min_adj) for t, e in alloc.items()}
        total2 = sum(normed.values())
        if abs(total2 - self.epsilon_total) > 1e-10:
            scale2 = self.epsilon_total / total2
            normed = {t: e * scale2 for t, e in normed.items()}
        return normed

    def generate_workload(
        self, query_types: list[str] | None = None, num_queries: int = 100
    ) -> list[WorkloadQuery]:
        """Build the allocator workload.

        ``workload_mode="paper"`` (default): 1-way per table, every FK pair,
        and 3-table join *chains*. Weights are **volume-aware**
        (``log(1+n)``, joins up-weighted) so fact tables are not starved.

        ``workload_mode="random"``: legacy Monte-Carlo queries (evaluation only).
        """
        if self.workload_mode == "random":
            return self._generate_random_workload(query_types, num_queries)
        return self._generate_paper_workload()

    def _generate_paper_workload(self) -> list[WorkloadQuery]:
        queries: list[WorkloadQuery] = []
        seen: set[tuple[str, ...]] = set()

        def _add(tables: list[str], order: int) -> None:
            key = tuple(sorted(tables))
            if not key or key in seen:
                return
            seen.add(key)
            queries.append(WorkloadQuery(tables=list(tables), order=order, weight=1.0))

        for t in self.table_names:
            _add([t], 1)

        for fk in self.schema.foreign_keys:
            _add([fk.child_table, fk.parent_table], 2)

        # 3-way *paths* only (grandparent→parent→child). Two-parent forks
        # become a query over every table on a star and force equal-split.
        for fk1 in self.schema.foreign_keys:
            for fk2 in self.schema.foreign_keys:
                if fk1.child_table == fk2.parent_table:
                    _add([fk1.parent_table, fk1.child_table, fk2.child_table], 3)

        if not queries:
            return queries
        for q in queries:
            # log-volume: fact tables get more weight without zeroing small tables.
            vol = sum(np.log1p(max(self.table_sizes.get(t, 1), 1)) for t in q.tables)
            if q.order >= 2:
                vol *= 2.0
            q.weight = float(max(vol, 1e-9))
        z = sum(q.weight for q in queries)
        for q in queries:
            q.weight /= z
        return queries

    def _generate_random_workload(
        self, query_types: list[str] | None, num_queries: int
    ) -> list[WorkloadQuery]:
        rng = np.random.default_rng(42)
        queries: list[WorkloadQuery] = []
        tables = self.table_names
        if not tables:
            return queries

        if query_types is None:
            query_types = ["1way", "2way", "cross_table"]

        table_weights = np.array([
            max(len(self.schema.tables[t].attributes), 1)
            + len(self.schema.child_tables(t))
            + len(self.schema.parent_tables(t))
            for t in tables
        ], dtype=float)
        for i, t in enumerate(tables):
            n_rows = self.table_sizes.get(t, 1)
            table_weights[i] *= np.log1p(max(n_rows, 1))
        tw_sum = table_weights.sum()
        if tw_sum > 0:
            table_weights /= tw_sum
        else:
            table_weights = np.ones(len(tables)) / len(tables)

        for _ in range(num_queries):
            qtype = rng.choice(query_types)
            if qtype == "1way":
                t = rng.choice(tables, p=table_weights)
                queries.append(WorkloadQuery(tables=[t], order=1, weight=1.0))
            elif qtype == "2way":
                t = rng.choice(tables, p=table_weights)
                queries.append(WorkloadQuery(tables=[t], order=2, weight=self.query_weight_decay))
            elif qtype == "3way":
                t = rng.choice(tables, p=table_weights)
                queries.append(
                    WorkloadQuery(tables=[t], order=3, weight=self.query_weight_decay ** 2)
                )
            elif qtype == "cross_table":
                n = min(int(rng.integers(2, 4)), len(tables))
                if n >= len(tables):
                    chosen = list(tables)
                else:
                    chosen = list(rng.choice(tables, size=n, replace=False, p=table_weights))
                queries.append(
                    WorkloadQuery(tables=chosen, order=2, weight=self.query_weight_decay)
                )

        return queries


def rdp_compose_check(
    epsilons: dict[str, float], delta: float, orders: list[float] | None = None
) -> float:
    """Verify RDP composition of per-table Laplace mechanisms.

    For the Laplace mechanism with scale b = sensitivity / epsilon,
    the RDP at order alpha is:  alpha / (2 * b^2 * (alpha - 1))
    which simplifies (sensitivity=1) to:  alpha * eps^2 / (2*(alpha-1)).

    Returns the composed (epsilon, delta)-DP epsilon.
    """
    if orders is None:
        orders = [1.5, 2, 3, 5, 10, 25, 50, 100]

    best_eps = float("inf")
    for alpha in orders:
        if alpha <= 1:
            continue
        rdp_sum = sum(
            alpha * (eps ** 2) / (2.0 * (alpha - 1.0))
            for eps in epsilons.values()
        )
        composed_eps = rdp_sum + np.log(1.0 / delta) / (alpha - 1.0)
        best_eps = min(best_eps, composed_eps)

    return float(best_eps)
