"""Sanity-check manuscript Eq. (5) allocator before any campaign."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.budget_allocator import BudgetAllocator
from src.data.datasets import load_dataset


def main() -> int:
    real, schema = load_dataset("synthetic_tiny")
    sizes = {t: len(df) for t, df in real.items()}
    alloc = BudgetAllocator(
        schema=schema,
        epsilon_total=1.0,
        delta=1e-5,
        epsilon_min=0.01,
        method="lp_rdp",
        table_sizes=sizes,
        workload_mode="paper",
    )
    wl = alloc.generate_workload()
    eps = alloc.allocate(wl)
    ss = sum(v * v for v in eps.values())
    print("tables:", list(schema.tables.keys()))
    print("M workload:", len(wl))
    print("alloc:", {k: round(v, 4) for k, v in eps.items()})
    print("sumsq:", round(ss, 6), "cap:", 1.0)
    print("sum:", round(sum(eps.values()), 4), "(need not equal ε)")
    if ss > 1.0 + 1e-6:
        print("FAIL: quadratic composition violated")
        return 1
    if min(eps.values()) < 1e-8:
        print("FAIL: non-positive ε")
        return 1
    eq = alloc._equal_rdp()
    print("equal_rdp:", {k: round(v, 4) for k, v in eq.items()}, "|| ||", round(np.linalg.norm(list(eq.values())), 4))
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
