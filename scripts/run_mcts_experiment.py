"""Unified entry point for AlphaCFG MCTS experiments."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphacfg.experiment_context import NETWORK_BACKENDS, default_network, experiment_import_context, validate_network
from alphacfg.seeding import seed_everything
from alphacfg.variants import list_variants


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[v.name for v in list_variants()], default="cfg-sem-k")
    parser.add_argument("--mode", choices=["pool", "single", "mask"], default="pool")
    parser.add_argument(
        "--network",
        choices=["auto", "local", *sorted(NETWORK_BACKENDS)],
        default="auto",
        help="Network backend. auto resolves to local for rpn and treelstm for CFG methods.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pool-capacity", type=int, default=None)
    parser.add_argument("--max-expression-length", type=int, default=None)
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    requested_network = None if args.network == "auto" else args.network
    try:
        network = validate_network(args.variant, requested_network or default_network(args.variant))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    seed_everything(args.seed, deterministic=args.deterministic)

    entry_name = {
        "pool": "run_pool.py",
        "single": "run_single.py",
        "mask": "run_mask.py",
    }[args.mode]

    with experiment_import_context(args.variant, network) as variant_root:
        entry = variant_root / entry_name
        if not entry.exists():
            raise FileNotFoundError(f"{args.variant} does not provide {entry_name}")

        runner_args = [
            str(entry),
            "--variant-name",
            args.variant,
            "--seed",
            str(args.seed),
            "--network-name",
            network,
        ]
        if args.pool_capacity is not None:
            runner_args.extend(["--pool-capacity", str(args.pool_capacity)])
        if args.max_expression_length is not None:
            runner_args.extend(["--max-expression-length", str(args.max_expression_length)])
        runner_args.extend(passthrough)
        sys.argv = runner_args
        runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
