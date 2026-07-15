"""Context helpers for selecting MCTS variants and network backends."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Iterator

from alphacfg.variants import get_variant


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMON_ROOT = PROJECT_ROOT / "alphacfg"
NETWORK_BACKEND_ROOT = PROJECT_ROOT / "alphacfg" / "network_backends"


NETWORK_BACKENDS = {
    "treelstm": NETWORK_BACKEND_ROOT / "treelstm",
    "lstm": NETWORK_BACKEND_ROOT / "lstm",
    "cnn": NETWORK_BACKEND_ROOT / "cnn",
    "transformer": NETWORK_BACKEND_ROOT / "transformer",
}

LOCAL_NETWORK = "local"
METHOD_NETWORKS = {
    "rpn": ("lstm", "cnn", "transformer", LOCAL_NETWORK),
    "cfg-syn": ("lstm", "cnn", "transformer", "treelstm"),
    "cfg-sem": ("lstm", "cnn", "transformer", "treelstm"),
    "cfg-sem-k": ("lstm", "cnn", "transformer", "treelstm"),
}


def available_networks(variant: str) -> tuple[str, ...]:
    """Return network backends supported by one search method."""
    return METHOD_NETWORKS[get_variant(variant).name]


def default_network(variant: str) -> str:
    """Resolve the default network for a method."""
    spec = get_variant(variant)
    return LOCAL_NETWORK if spec.name == "rpn" else "treelstm"


def validate_network(variant: str, network: str | None) -> str:
    """Validate and normalize a method/network combination."""
    spec = get_variant(variant)
    network_key = (network or default_network(spec.name)).lower()
    allowed = available_networks(spec.name)
    if network_key not in allowed:
        raise ValueError(
            f"{spec.name} cannot use network '{network_key}'. "
            f"Available for this method: {', '.join(allowed)}."
        )
    if network_key != LOCAL_NETWORK and network_key not in NETWORK_BACKENDS:
        available = ", ".join(sorted(NETWORK_BACKENDS))
        raise ValueError(f"Unknown network backend '{network_key}'. Available: {available}")
    return network_key


@contextlib.contextmanager
def experiment_import_context(variant: str, network: str) -> Iterator[Path]:
    """Temporarily put selected variant and network backend first on sys.path."""
    spec = get_variant(variant)
    network_key = validate_network(spec.name, network)

    original_path = list(sys.path)
    original_cwd = Path.cwd()
    try:
        backend_root = None if network_key == LOCAL_NETWORK else NETWORK_BACKENDS[network_key]
        for path in (backend_root, spec.path, COMMON_ROOT, PROJECT_ROOT):
            if path is None:
                continue
            path_str = str(path)
            if path_str in sys.path:
                sys.path.remove(path_str)
        if backend_root is not None:
            sys.path.insert(0, str(backend_root))
        insert_at = 1 if backend_root is not None else 0
        sys.path.insert(insert_at, str(spec.path))
        sys.path.insert(insert_at + 1, str(COMMON_ROOT))
        sys.path.insert(insert_at + 2, str(PROJECT_ROOT))
        os.chdir(spec.path)
        yield spec.path
    finally:
        sys.path[:] = original_path
        os.chdir(original_cwd)
