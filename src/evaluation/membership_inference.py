"""Stub membership-inference eval on the joined table.

Protocol (train on real / synth), stated so a TIFS reader can rerun it:

1. Inner-join every table along the longest FK chain (same helper as ML utility).
2. Clip each join to ``join_clip_mult`` times the real join cardinality.
3. Encode the joined columns to a numeric feature matrix (label-encode
   categoricals; median-fill numerics). No raw identifiers are kept as labels.
4. **Member class (1):** rows sampled from the *real* joined table.
5. **Non-member class (0):** rows sampled from the *synthetic* joined table.
6. Stratified 70/30 train/test split. Fit a random forest on the train split
   only. Report test accuracy and ROC-AUC.

This is a presence / distinguishability stub on the release, not a full
shadow-model attack and not a proof of differential privacy. If either join
is too small, the function returns NaNs instead of inventing a number.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import _encode_features, _join_all
from src.schema import RelationalSchema

logger = logging.getLogger(__name__)


def membership_inference_joined(
    real_data: dict[str, pd.DataFrame],
    synthetic_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    seed: int = 42,
    join_clip_mult: float = 10.0,
    max_joined_rows: int = 20000,
    test_size: float = 0.3,
) -> dict[str, Any]:
    """Binary RF: real-joined (member) vs synth-joined (non-member)."""
    real_joined = _join_all(real_data, schema, max_joined_rows=max_joined_rows)
    syn_joined = _join_all(synthetic_data, schema, max_joined_rows=max_joined_rows)
    cap = int(join_clip_mult * max(len(real_joined), 1))
    if len(syn_joined) > cap:
        syn_joined = syn_joined.sample(n=cap, random_state=seed).reset_index(drop=True)
    if len(real_joined) > cap:
        real_joined = real_joined.sample(n=cap, random_state=seed).reset_index(drop=True)

    empty = {
        "mi_accuracy": float("nan"),
        "mi_auc": float("nan"),
        "mi_n_real": int(len(real_joined)),
        "mi_n_synth": int(len(syn_joined)),
        "mi_protocol": "real_joined=member, synth_joined=nonmember, RF holdout",
    }
    if real_joined.empty or syn_joined.empty:
        logger.warning("MI stub: empty join; returning NaNs (not a measured attack).")
        return empty

    cols = [c for c in real_joined.columns if c in syn_joined.columns]
    if not cols:
        return empty

    n = min(len(real_joined), len(syn_joined), max_joined_rows)
    if n < 40:
        logger.warning("MI stub: only %d paired rows; returning NaNs.", n)
        return empty

    rng = np.random.default_rng(seed)
    real_idx = rng.choice(len(real_joined), size=n, replace=False)
    syn_idx = rng.choice(len(syn_joined), size=n, replace=False)
    real_x = _encode_features(real_joined.iloc[real_idx][cols])
    syn_x = _encode_features(syn_joined.iloc[syn_idx][cols])

    x = np.vstack([real_x, syn_x])
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    try:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=test_size, random_state=seed, stratify=y
        )
    except ValueError:
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=test_size, random_state=seed
        )

    clf = RandomForestClassifier(
        n_estimators=80, max_depth=8, random_state=seed, n_jobs=-1
    )
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    proba = clf.predict_proba(x_te)[:, 1]
    acc = float(accuracy_score(y_te, pred))
    try:
        auc = float(roc_auc_score(y_te, proba))
    except ValueError:
        auc = float("nan")

    return {
        "mi_accuracy": acc,
        "mi_auc": auc,
        "mi_n_real": int(n),
        "mi_n_synth": int(n),
        "mi_protocol": "real_joined=member, synth_joined=nonmember, RF holdout",
    }
