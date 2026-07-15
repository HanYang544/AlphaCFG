"""Runtime helpers shared by AlphaCFG runners."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def build_output_dir(variant: str, run_name: str, output_dir: str = "") -> Path:
    """Return the canonical output directory for a run.

    Relative names are always placed below outputs/<variant>/ so experiment
    artifacts do not scatter under each legacy variant directory.
    """
    if output_dir:
        path = Path(output_dir)
        if path.is_absolute():
            return path
        return OUTPUT_ROOT / variant / path
    return OUTPUT_ROOT / variant / run_name


def namespace_to_dict(args: Namespace) -> dict[str, Any]:
    """Convert argparse arguments to a JSON-serializable dictionary."""
    return {key: _jsonable(value) for key, value in vars(args).items()}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a UTF-8 JSON file with stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_run_files(output_dir: Path, args: Namespace, extra: Mapping[str, Any] | None = None) -> None:
    """Persist run configuration and basic metadata in the run directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_config.json", namespace_to_dict(args))
    metadata = {
        "pid": os.getpid(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
    }
    if extra:
        metadata.update({key: _jsonable(value) for key, value in extra.items()})
    write_json(output_dir / "run_metadata.json", metadata)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)
