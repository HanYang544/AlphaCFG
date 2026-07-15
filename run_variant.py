"""Compatibility wrapper for the unified MCTS experiment runner."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    runner = PROJECT_ROOT / "scripts" / "run_mcts_experiment.py"
    sys.argv = [str(runner), *sys.argv[1:]]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
