"""Emit Phase C summary tables from a campaign CSV. No TeX writes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CT = "query_accuracy.query_error_cross_table"
AVG = "query_accuracy.query_error_avg"
FK = "fk_consistency.overall"
ML = "ml_utility.accuracy_syn"
DISPLAY = {
    "independent": "Independent",
    "equal_split": "Equal-split",
    "privaschema": "PrivaSchema",
    "aim_repair": "AIM+Repair",
    "privaschema_aim": "PrivaSchema-AIM",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--eps", type=float, default=1.0)
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    sub = df[(df["epsilon"] - args.eps).abs() < 1e-9]
    if sub.empty:
        print("no rows")
        return 2
    g = sub.groupby(["dataset", "method"], as_index=False)[[CT, AVG, FK, ML]].mean()
    print(f"rows={len(df)} eps={args.eps} cells={len(g)}")
    print("\n=== per-dataset means ===")
    print(g.round(4).to_string(index=False))
    print("\n=== seven-dataset method means ===")
    m = g.groupby("method")[[CT, AVG, FK, ML]].mean().sort_values(CT)
    print(m.round(4).to_string())
    print("\n=== CT pivot ===")
    print(g.pivot_table(index="dataset", columns="method", values=CT).round(4).to_string())
    print("\n=== FK pivot ===")
    print(g.pivot_table(index="dataset", columns="method", values=FK).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
