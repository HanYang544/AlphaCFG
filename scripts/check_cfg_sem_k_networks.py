"""Check CFG-Sem-k network backends for forward/backward training sanity."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphacfg.experiment_context import NETWORK_BACKENDS, experiment_import_context
from alphacfg.runtime import OUTPUT_ROOT, write_json
from alphacfg.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="network_checks")
    parser.add_argument("--networks", nargs="*", default=sorted(NETWORK_BACKENDS))
    return parser.parse_args()


def clear_network_modules() -> None:
    for name in list(sys.modules):
        if name == "networks" or name.startswith("networks."):
            del sys.modules[name]


def check_one(network: str, seed: int, device: torch.device, steps: int) -> dict:
    seed_everything(seed, deterministic=True)
    with experiment_import_context("cfg-sem-k", network):
        clear_network_modules()
        from env.token_v import create_token_map, get_first_token, parse_expression_to_tokens

        feature_mod = importlib.import_module("networks.feature_")
        policy_mod = importlib.import_module("networks.policy_")
        value_mod = importlib.import_module("networks.value_")

        embed_dim = 64
        token_map = create_token_map()
        feature = feature_mod.FeatureExtractor(embed_dim=embed_dim, h_size=embed_dim, dropout=0.0).to(device)
        policy = policy_mod.PolicyNetwork(feature_extractor=feature, embed_dim=embed_dim).to(device)
        value = value_mod.ValueNetwork(feature_extractor=feature, embed_dim=embed_dim).to(device)

        expression = "J"
        tokens = parse_expression_to_tokens(expression, token_map)
        first_token = get_first_token(tokens, token_map)
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        target_value = torch.tensor([0.25], dtype=torch.float32, device=device)
        params = list({id(param): param for param in list(policy.parameters()) + list(value.parameters())}.values())
        optimizer = torch.optim.Adam(params, lr=1e-3)
        value_loss_fn = nn.MSELoss()

        losses = []
        grad_norms = []
        for _ in range(steps):
            optimizer.zero_grad()
            policy_outputs = policy(expression if network == "treelstm" else token_tensor, first_token=first_token)
            if not policy_outputs:
                raise RuntimeError(f"{network} produced empty policy outputs")
            probs = torch.stack([item["prob"] for item in policy_outputs])
            policy_loss = -torch.log(probs[0] + 1e-10)

            value_input = expression if network == "treelstm" else token_tensor
            value_pred = value(value_input).squeeze()
            value_loss = value_loss_fn(value_pred.unsqueeze(0), target_value)
            loss = policy_loss + value_loss
            loss.backward()

            grad_norm = 0.0
            for param in list(policy.parameters()) + list(value.parameters()):
                if param.grad is not None:
                    grad_norm += float(param.grad.detach().norm().cpu())
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(grad_norm)

        return {
            "network": network,
            "ok": losses[-1] <= losses[0] and max(grad_norms) > 0,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "grad_norm_max": max(grad_norms),
            "steps": steps,
        }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    results = []
    for network in args.networks:
        results.append(check_one(network, args.seed, device, args.steps))
    output_dir = OUTPUT_ROOT / "cfg-sem-k" / args.output_dir
    write_json(output_dir / "network_check_summary.json", {"results": results})
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    if not all(item["ok"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
