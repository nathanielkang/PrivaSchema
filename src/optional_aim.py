"""Optional AIM table engine (smartnoise-synth + private-pgm).

Used by method F (parent-first AIM + equal ε/√K). If the stack is missing,
raises OptionalExtraError with install text.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.errors import OptionalExtraError
from src.schema import RelationalSchema

logger = logging.getLogger(__name__)


def _import_aim():
    try:
        from snsynth import Synthesizer
    except ImportError as e:
        raise OptionalExtraError(
            "AIM extra requires smartnoise-synth and private-pgm. "
            "pip install smartnoise-synth && "
            "pip install git+https://github.com/ryan112358/private-pgm.git. "
            f"Import failed: {e}"
        ) from e
    return Synthesizer


def _noisy_oneway_sample(df: pd.DataFrame, epsilon: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df)
    k = max(df.shape[1], 1)
    eps_col = float(epsilon) / np.sqrt(k)
    out: dict[str, np.ndarray] = {}
    for col in df.columns:
        vals, counts = np.unique(df[col].astype(str).to_numpy(), return_counts=True)
        noisy = counts.astype(float) + rng.laplace(0.0, 1.0 / max(eps_col, 1e-8), size=len(counts))
        noisy = np.maximum(noisy, 0.0)
        if noisy.sum() <= 0:
            noisy = np.ones_like(noisy)
        out[col] = rng.choice(vals, size=n, replace=True, p=noisy / noisy.sum())
    return pd.DataFrame(out)


def synthesize_one_table_aim(
    df: pd.DataFrame,
    epsilon_table: float,
    delta: float,
    seed: int,
) -> pd.DataFrame:
    """AIM on a single table, or Laplace 1-way if AIM's candidate set is empty."""
    Synthesizer = _import_aim()
    work = df.copy()
    if work.empty or work.shape[1] == 0:
        return work
    cats = list(work.columns)
    try:
        synth = Synthesizer.create(
            "aim",
            epsilon=float(epsilon_table),
            delta=float(delta),
            verbose=False,
        )
        try:
            synth.fit(work, preprocessor_eps=0.0, categorical_columns=cats)
        except TypeError:
            synth.fit(work, categorical_columns=cats)
        syn_df = synth.sample(len(work))
        if not isinstance(syn_df, pd.DataFrame):
            syn_df = pd.DataFrame(syn_df, columns=work.columns)
        return syn_df.reset_index(drop=True)
    except OptionalExtraError:
        raise
    except Exception as e:  # noqa: BLE001
        logger.warning("AIM fit/sample failed (%s); Laplace 1-way fallback.", e)
        return _noisy_oneway_sample(work, float(epsilon_table), seed)


def require_aim_stack() -> None:
    """Fail fast so method F does not start a run without AIM."""
    _import_aim()
    _ = RelationalSchema
