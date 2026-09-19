"""Score Phase B CSV against the Nate win bar. No TeX writes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CT = "query_accuracy.query_error_cross_table"
AVG = "query_accuracy.query_error_avg"
FK = "fk_consistency.overall"
# Integrity set = methods that coordinate or repair keys. Independent is excluded.
# Rank only rows whose seed-mean FK is at least FK_MIN.
INTEGRITY = ("equal_split", "privaschema", "aim_repair", "privaschema_aim")
FK_MIN = 0.99


def _mean_at_eps(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    sub = df[np_isclose(df["epsilon"], eps)].copy()
    if sub.empty:
        return sub
    return sub.groupby(["dataset", "method"], as_index=False)[[CT, AVG, FK]].mean()


def np_isclose(series: pd.Series, value: float) -> pd.Series:
    return (series - value).abs() < 1e-9


def score(path: Path) -> int:
    df = pd.read_csv(path)
    print(f"rows={len(df)} methods={sorted(df['method'].unique())} datasets={sorted(df['dataset'].unique())}")
    for eps in (0.5, 1.0, 5.0):
        g = _mean_at_eps(df, eps)
        if g.empty:
            print(f"\nε={eps}: no rows yet")
            continue
        print(f"\n=== ε={eps} seven-dataset method means ===")
        m = g.groupby("method")[[CT, AVG, FK]].mean().sort_values(CT)
        print(m.round(4).to_string())
        integ = g[g["method"].isin(INTEGRITY)]
        if integ.empty:
            continue
        fk_ok = integ.groupby("method")[FK].mean()
        keep = [name for name, fk in fk_ok.items() if fk >= FK_MIN]
        print(f"\nFK>={FK_MIN} methods: {keep}")
        ranked = m.loc[m.index.intersection(keep)].sort_values(CT)
        print("Integrity CT ranking (FK-qualified):")
        print(ranked.round(4).to_string())
        pivot = integ.pivot_table(index="dataset", columns="method", values=CT)
        print("\nCT by dataset:")
        print(pivot.round(4).to_string())
        if "privaschema" in m.index and "equal_split" in m.index:
            gap = m.loc["equal_split", CT] - m.loc["privaschema", CT]
            print(f"\nallocator gap (equal CT - S CT) = {gap:.4f}  (>0 means S wins CT)")
        ours = [n for n in ("privaschema", "privaschema_aim") if n in ranked.index]
        if len(ranked) and ours:
            winner = ranked.index[0]
            ours_best = ranked.loc[ours, CT].idxmin()
            print(f"lowest integrity CT: {winner}")
            print(f"best ours: {ours_best}  win_bar={winner in ours}")
    print("\nWin bar: privaschema or privaschema_aim lowest mean CT among FK>=0.99 methods.")
    print("Independent is not in that set. Do not patch TeX from a loss.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    args = p.parse_args()
    if not args.csv.is_file():
        print(f"missing {args.csv}", file=sys.stderr)
        return 2
    return score(args.csv)


if __name__ == "__main__":
    raise SystemExit(main())
