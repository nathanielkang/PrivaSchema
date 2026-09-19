"""Convenience entry: TIFS missing-cell config (F / U / C / P)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if "--config" not in sys.argv:
    sys.argv[1:1] = ["--config", str(ROOT / "configs" / "tifs_missing.yaml")]
runpy.run_path(str(ROOT / "scripts" / "run_experiments.py"), run_name="__main__")
