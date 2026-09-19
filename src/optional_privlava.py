"""Optional PrivLava extra (method P).

This module never writes a measured table row from a CPU stand-in.

Official stack (all required):
  - git clone https://github.com/caicre/PrivLava extras/PrivLava
    (or set PRIVASCHEMA_PRIVLAVA_ROOT)
  - that clone must import CRF and PrivMRF
  - optional adapter: extras/PrivLava/privaschema_adapter.py::synthesize

If any piece is missing, ``run_privlava`` raises OptionalExtraError with
the install path. Do not treat a skipped P cell as a measured number.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from src.errors import OptionalExtraError
from src.schema import RelationalSchema

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/caicre/PrivLava"


def _extra_root(cfg: dict) -> Path:
    env = os.environ.get("PRIVASCHEMA_PRIVLAVA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    named = (cfg.get("baselines") or {}).get("privlava_path")
    if named:
        return Path(named).expanduser().resolve()
    here = Path(__file__).resolve().parents[1]
    return (here / "extras" / "PrivLava").resolve()


def _try_adapter(root: Path):
    adapter = root / "privaschema_adapter.py"
    if not adapter.is_file():
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("privlava_adapter", adapter)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "synthesize", None)


def run_privlava(
    real_data: dict[str, pd.DataFrame],
    schema: RelationalSchema,
    cfg: dict,
    epsilon: float,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Call the official PrivLava stack, or fail with a useful error."""
    root = _extra_root(cfg)
    if not root.is_dir() or not any(root.iterdir()):
        raise OptionalExtraError(
            f"PrivLava optional extra is not installed at {root}. "
            f"Clone the official repo: git clone {REPO_URL} {root} "
            "and install its CRF/PrivMRF stack. "
            "This extra will not emit a CPU stand-in or fake metrics."
        )

    import sys

    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)

    try:
        import CRF  # noqa: F401
        import PrivMRF  # noqa: F401
    except ImportError as e:
        raise OptionalExtraError(
            f"PrivLava clone exists at {root} but CRF/PrivMRF did not import ({e}). "
            "Install the clone's GPU/CuPy requirements. "
            "Refusing to write a stand-in row."
        ) from e

    synth_fn = _try_adapter(root)
    if synth_fn is None:
        raise OptionalExtraError(
            f"PrivLava CRF/PrivMRF imported from {root}, but "
            "privaschema_adapter.py::synthesize is missing. "
            "Add that adapter (official entry is dataset-specific scripts) "
            "before enabling method P. No fake numeric results will be written."
        )
    return synth_fn(real_data, schema, cfg, epsilon, seed)
