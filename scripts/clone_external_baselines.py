"""Clone official baseline repos into code/external/ (no experiment runs)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPOS = {
    "PrivLava": "https://github.com/caicre/PrivLava.git",
    "PrivPetal": "https://github.com/caicre/PrivPetal.git",
    "rdb-diffusion": "https://github.com/ketatam/rdb-diffusion.git",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone external PrivaSchema baselines")
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "external"),
        help="Destination directory (default: code/external)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of keys: PrivLava PrivPetal rdb-diffusion",
    )
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    selected = args.only or list(REPOS.keys())
    for name in selected:
        if name not in REPOS:
            print(f"Unknown repo key: {name}", file=sys.stderr)
            return 2
        dest = root / name
        if dest.is_dir() and any(dest.iterdir()):
            print(f"[skip] {name} already present at {dest}")
            continue
        url = REPOS[name]
        print(f"[clone] {url} -> {dest}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)
    print("Done. Install each clone's own requirements before enabling those methods.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
