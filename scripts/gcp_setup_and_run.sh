#!/usr/bin/env bash
# PrivaSchema GCP setup + smoke + optional full campaign on fog-cpu.
# Run from the repository root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

echo "[1/5] Python venv"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt
# AIM wrapper
python -m pip install "smartnoise-synth>=1.0.0" || python -m pip install snsynth || true

echo "[2/5] Clone external baselines (CPU-friendly best-effort)"
python scripts/clone_external_baselines.py || true

echo "[3/5] Import check"
python - <<'PY'
from src.baselines import PAPER_METHODS, build_registry
print("paper", PAPER_METHODS)
print("registry", sorted(build_registry()))
PY

echo "[4/5] Smoke (synthetic_star, eps=1, 1 run, all methods; missing skipped)"
mkdir -p results/gcp_smoke
python scripts/run_experiments.py \
  --config configs/gcp_smoke.yaml \
  --output results/gcp_smoke \
  2>&1 | tee results/gcp_smoke/smoke.log

echo "[5/5] Smoke done. To launch FULL campaign:"
echo "  nohup python scripts/run_experiments.py --config configs/default.yaml --output results/gcp_full \\"
echo "    > results/gcp_full/full.log 2>&1 &"
echo "  mkdir -p results/gcp_full first."
