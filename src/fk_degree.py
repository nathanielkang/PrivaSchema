"""DP degree-preserving FK assignment (parent-first).

Uses a slice of the child table budget to release a noisy FK histogram,
then samples child FKs only from synthetic parent primary keys.
This is the mechanism the join lower bound asks for: keys are coordinated,
not drawn independently and repaired after the fact.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.schema import RelationalSchema

logger = logging.getLogger(__name__)


def split_table_and_fk_eps(
    eps_table: float,
    fk_hist_frac: float | None = None,
    fk_rdp_share: float = 0.25,
) -> tuple[float, float]:
    """Quadratic RDP split: ε_fk² = share · ε_i², ε_synth² = (1-share) · ε_i².

    Default share is 0.25 / 0.75. ``fk_hist_frac`` is accepted as an alias for
    the quadratic share (not a linear slice of ε).
    """
    share = float(fk_rdp_share if fk_hist_frac is None else fk_hist_frac)
    share = min(max(share, 0.0), 0.49)
    if share <= 0.0 or eps_table <= 0.0:
        return float(eps_table), 0.0
    eps_fk = float(eps_table) * np.sqrt(share)
    eps_synth = float(eps_table) * np.sqrt(1.0 - share)
    return eps_synth, eps_fk


def noisy_fk_histogram(
    series: pd.Series,
    epsilon: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Laplace-noisy non-negative weights over observed FK values."""
    if epsilon <= 0 or series is None or len(series) == 0:
        return np.array([]), np.array([])
    counts = series.astype(str).value_counts()
    keys = counts.index.to_numpy()
    vals = counts.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    noisy = vals + rng.laplace(0.0, 1.0 / max(epsilon, 1e-8), size=len(vals))
    noisy = np.maximum(noisy, 0.0)
    if noisy.sum() <= 0:
        noisy = np.ones_like(noisy)
    return keys, noisy / noisy.sum()


def sample_uniform_fks(parent_pk_vals: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Uniform draw from live parent PKs (histogram ablation U)."""
    rng = np.random.default_rng(seed)
    parent_pk = np.asarray(parent_pk_vals)
    if len(parent_pk) == 0:
        return np.array([], dtype=object)
    return rng.choice(parent_pk, size=n, replace=True)


def sample_valid_fks(
    keys: np.ndarray,
    probs: np.ndarray,
    parent_pk_vals: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    """Sample n FKs. Prefer noisy degree mass that lands on live parent PKs."""
    rng = np.random.default_rng(seed)
    parent_pk = np.asarray(parent_pk_vals)
    if len(parent_pk) == 0:
        return np.array([], dtype=object)
    if len(keys) == 0:
        return rng.choice(parent_pk, size=n, replace=True)

    parent_str = parent_pk.astype(str)
    live = {str(v) for v in parent_str}
    raw = pd.DataFrame({"k": keys.astype(str), "p": probs})
    raw = raw[raw["k"].isin(live)]
    if raw.empty or raw["p"].sum() <= 0:
        return rng.choice(parent_pk, size=n, replace=True)

    raw["p"] = raw["p"] / raw["p"].sum()
    # Map string keys back to a live parent value of the original dtype
    lookup: dict[str, object] = {}
    for v in parent_pk:
        lookup.setdefault(str(v), v)
    mapped = np.array([lookup[k] for k in raw["k"]], dtype=object)
    return rng.choice(mapped, size=n, replace=True, p=raw["p"].to_numpy())


def fit_child_fk_hists(
    schema: RelationalSchema,
    table_name: str,
    real_df: pd.DataFrame,
    eps_fk: float,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """One noisy histogram per outgoing FK, splitting ε_fk under RDP."""
    fks = [fk for fk in schema.foreign_keys if fk.child_table == table_name]
    if not fks or eps_fk <= 0:
        return {}
    eps_each = eps_fk / np.sqrt(max(len(fks), 1))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, fk in enumerate(fks):
        if fk.child_col not in real_df.columns:
            continue
        keys, probs = noisy_fk_histogram(real_df[fk.child_col], float(eps_each), seed + i)
        out[fk.child_col] = (keys, probs)
        logger.info(
            "FK hist %s.%s eps=%.4f keys=%d",
            table_name, fk.child_col, eps_each, len(keys),
        )
    if len(fks) >= 2:
        joint = fit_junction_hist(real_df, fks, float(eps_fk), seed)
        if joint is not None:
            out["__junction__"] = joint  # type: ignore[assignment]
    return out


def fit_junction_hist(
    real_df: pd.DataFrame,
    fks: list,
    eps_fk: float,
    seed: int,
    max_pairs: int = 4000,
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    """Noisy joint histogram over a multi-parent child's FK columns (IMDB/TPC-H)."""
    cols = [fk.child_col for fk in fks if fk.child_col in real_df.columns]
    if len(cols) < 2 or eps_fk <= 0 or len(real_df) == 0:
        return None
    keys_s = real_df[cols].astype(str).agg("||".join, axis=1)
    counts = keys_s.value_counts()
    if len(counts) > max_pairs:
        counts = counts.head(max_pairs)
    keys = counts.index.to_numpy()
    vals = counts.to_numpy(dtype=float)
    rng = np.random.default_rng(seed + 17)
    noisy = np.maximum(vals + rng.laplace(0.0, 1.0 / max(eps_fk, 1e-8), size=len(vals)), 0.0)
    if noisy.sum() <= 0:
        noisy = np.ones_like(noisy)
    logger.info("Junction hist cols=%s pairs=%d eps=%.4f", cols, len(keys), eps_fk)
    return cols, keys, noisy / noisy.sum()


def sample_junction_fks(
    cols: list[str],
    keys: np.ndarray,
    probs: np.ndarray,
    parent_pk_by_col: dict[str, np.ndarray],
    n: int,
    seed: int,
) -> dict[str, np.ndarray] | None:
    """Sample n joint FK tuples, keeping only pairs whose keys are live."""
    if n <= 0 or not cols:
        return None
    live: dict[str, dict[str, object]] = {}
    for col in cols:
        lookup: dict[str, object] = {}
        for v in np.asarray(parent_pk_by_col.get(col, [])):
            lookup.setdefault(str(v), v)
        live[col] = lookup
        if not lookup:
            return None

    kept_keys: list[str] = []
    kept_p: list[float] = []
    for k, p in zip(keys.astype(str), probs):
        parts = str(k).split("||")
        if len(parts) != len(cols):
            continue
        if all(parts[i] in live[cols[i]] for i in range(len(cols))):
            kept_keys.append(str(k))
            kept_p.append(float(p))
    rng = np.random.default_rng(seed)
    if not kept_keys:
        return {
            col: rng.choice(np.array(list(live[col].values()), dtype=object), size=n, replace=True)
            for col in cols
        }
    p = np.asarray(kept_p, dtype=float)
    p = p / p.sum()
    chosen = rng.choice(np.asarray(kept_keys), size=n, replace=True, p=p)
    out: dict[str, np.ndarray] = {}
    for i, col in enumerate(cols):
        out[col] = np.array([live[col][str(k).split("||")[i]] for k in chosen], dtype=object)
    return out
