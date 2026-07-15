"""Benchmark one policy/value evaluation for an AlphaCFG network backend."""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphacfg.experiment_context import experiment_import_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="cfg-sem-k")
    parser.add_argument("--network", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--expression",
        default="Corr(Cov(Log(Abs($close)),$vwap,20),$close,30)",
    )
    parser.add_argument("--continue-out-game", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    with experiment_import_context(args.variant, args.network):
        from env.token_v import create_token_map, get_first_token, parse_expression_to_tokens
        from networks.feature_ import FeatureExtractor
        from networks.policy_ import PolicyNetwork
        from networks.value_ import ValueNetwork

        embed_dim = 128
        feature_extractor = FeatureExtractor(embed_dim=embed_dim, h_size=embed_dim, dropout=0.0).to(device)
        policy_net = PolicyNetwork(feature_extractor, embed_dim).to(device).eval()
        value_net = ValueNetwork(feature_extractor, embed_dim).to(device).eval()

        token_map = create_token_map()
        tokens = parse_expression_to_tokens(args.expression, token_map)
        first_token = get_first_token(tokens, token_map)
        policy_src = args.expression
        if args.network == "lstm":
            policy_src = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        def sync():
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        def evaluate_once():
            with torch.no_grad():
                policy_net(policy_src, first_token=first_token, continue_out_game=args.continue_out_game)
                value_net(args.expression)

        for _ in range(args.warmup):
            evaluate_once()
        sync()

        elapsed_ms = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            evaluate_once()
            sync()
            elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    print({
        "variant": args.variant,
        "network": args.network,
        "device": str(device),
        "repeat": args.repeat,
        "mean_ms": statistics.mean(elapsed_ms),
        "median_ms": statistics.median(elapsed_ms),
        "min_ms": min(elapsed_ms),
        "max_ms": max(elapsed_ms),
    })


if __name__ == "__main__":
    main()
