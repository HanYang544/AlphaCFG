"""Print the grammar/search metadata used by AlphaCFG variants."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphacfg.grammar import GRAMMAR_SPECS


def main() -> None:
    payload = {name: asdict(spec) for name, spec in GRAMMAR_SPECS.items()}
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
