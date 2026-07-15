"""Tiny compatibility entry helpers for method-local run scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap_project_path(method_file: str) -> None:
    project_root = Path(method_file).resolve().parents[3]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def force_reward_mode(mode: str) -> None:
    forwarded: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--reward-mode":
            skip_next = True
            continue
        if arg.startswith("--reward-mode="):
            continue
        forwarded.append(arg)
    sys.argv = [sys.argv[0], "--reward-mode", mode, *forwarded]


def run_method(method_file: str, variant: str, forced_reward_mode: str | None = None) -> None:
    bootstrap_project_path(method_file)
    if forced_reward_mode is not None:
        force_reward_mode(forced_reward_mode)
    from alphacfg.unified_runner import main

    main(default_variant=variant)
