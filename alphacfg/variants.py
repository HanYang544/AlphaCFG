"""Variant registry for the AlphaCFG experiment families."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ROOT = PROJECT_ROOT / "alphacfg" / "methods"


@dataclass(frozen=True)
class VariantSpec:
    """Static metadata for one experiment variant."""

    name: str
    directory: str
    description: str
    entries: Tuple[str, ...]

    @property
    def path(self) -> Path:
        return METHOD_ROOT / self.directory


VARIANTS: Dict[str, VariantSpec] = {
    "rpn": VariantSpec(
        name="rpn",
        directory="rpn",
        description="Reverse Polish Notation expression search.",
        entries=("run_pool.py", "run_single.py"),
    ),
    "cfg-syn": VariantSpec(
        name="cfg-syn",
        directory="cfg_syn",
        description="CFG search with syntax constraints.",
        entries=("run_pool.py", "run_single.py"),
    ),
    "cfg-sem": VariantSpec(
        name="cfg-sem",
        directory="cfg_sem",
        description="CFG search with syntax and semantic constraints.",
        entries=("run_pool.py", "run_single.py"),
    ),
    "cfg-sem-k": VariantSpec(
        name="cfg-sem-k",
        directory="cfg_sem_k",
        description="CFG semantic search with k-constraint optimizations.",
        entries=("run_pool.py", "run_single.py", "run_mask.py"),
    ),
}


def list_variants() -> Iterable[VariantSpec]:
    """Return all known experiment variants in a stable order."""
    for name in ("rpn", "cfg-syn", "cfg-sem", "cfg-sem-k"):
        yield VARIANTS[name]


def get_variant(name: str) -> VariantSpec:
    """Resolve a variant name into metadata and fail with a useful message."""
    key = name.lower()
    if key not in VARIANTS:
        available = ", ".join(VARIANTS)
        raise ValueError(f"Unknown AlphaCFG variant '{name}'. Available: {available}")
    return VARIANTS[key]
