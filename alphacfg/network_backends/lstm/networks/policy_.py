"""Policy network for the LSTM backend."""

from __future__ import annotations

import traceback

import torch
import torch.nn as nn

from env.token_v import create_token_map, get_first_token, parse_expression_to_tokens


CATEGORY_MAP = {
    0: "Vari",
    1: "constant",
    2: "unaryop",
    3: "binaryop",
    4: "rollingop",
    5: "pairedrollingop",
    6: "num",
}

ACTION_MAP = {
    "unaryop": ["Log", "Abs", "Sign", "CSRank"],
    "binaryop": [
        "Add1",
        "Add2",
        "Sub1",
        "Sub2",
        "Sub3",
        "Mul1",
        "Mul2",
        "Div1",
        "Div2",
        "Div3",
        "Pow1",
        "Pow2",
        "Pow3",
        "Greater1",
        "Greater2",
        "Greater3",
        "Less1",
        "Less2",
        "Less3",
    ],
    "rollingop": [
        "Ref",
        "Skew",
        "Kurt",
        "Mean",
        "Sum",
        "Std",
        "Var",
        "Max",
        "Min",
        "Med",
        "Rank",
        "Mad",
        "Delta",
        "WMA",
        "EMA",
    ],
    "pairedrollingop": ["Cov", "Corr"],
    "Vari": ["high", "low", "volume", "open", "close", "vwap"],
    "constant": [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1],
    "num": [20, 30, 40],
}


class PolicyNetwork(nn.Module):
    """Policy head on top of an LSTM expression encoder."""

    def __init__(self, feature_extractor, embed_dim: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.hidden_layer1 = nn.Linear(embed_dim, 64)
        self.hidden_layer2 = nn.Linear(64, embed_dim)
        self.category_head = nn.Linear(embed_dim, len(CATEGORY_MAP))
        self.action_heads = nn.ModuleDict({
            category: nn.Linear(embed_dim, len(actions))
            for category, actions in ACTION_MAP.items()
        })
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, first_token=None, continue_out_game=3) -> list:
        features = self.feature_extractor(src)
        features = torch.relu(self.hidden_layer1(features))
        features = torch.relu(self.hidden_layer2(features))

        action_mask = {category: 1 for category in ACTION_MAP}
        if continue_out_game == 2:
            action_mask["pairedrollingop"] = 0
        elif continue_out_game == 1:
            for category in ("binaryop", "rollingop", "pairedrollingop"):
                action_mask[category] = 0
        elif continue_out_game == 0:
            for category in ("unaryop", "binaryop", "rollingop", "pairedrollingop"):
                action_mask[category] = 0

        if first_token == "J":
            for category in ("unaryop", "binaryop", "rollingop", "pairedrollingop", "Vari", "num"):
                action_mask[category] = 0
        elif first_token == "Q":
            for category in ("constant", "num"):
                action_mask[category] = 0
        elif first_token == "num":
            for category in ("unaryop", "binaryop", "rollingop", "pairedrollingop", "Vari", "constant"):
                action_mask[category] = 0

        category_logits = self.category_head(features)
        category_probs = self.softmax(category_logits).squeeze(0)
        category_mask = torch.tensor(
            [float(action_mask.get(CATEGORY_MAP[i], 1)) for i in range(len(CATEGORY_MAP))],
            dtype=category_probs.dtype,
            device=category_probs.device,
        )
        category_probs = category_probs * category_mask
        category_probs = category_probs / (category_probs.sum() + 1e-10)

        action_outputs = []
        for i, category_prob in enumerate(category_probs):
            category_name = CATEGORY_MAP.get(i)
            if category_name is None or float(category_prob.detach().cpu()) == 0.0:
                continue
            action_logits = self.action_heads[category_name](features)
            action_probs = self.softmax(action_logits).squeeze(0)
            for j, action_prob in enumerate(action_probs):
                action_outputs.append({
                    "category": category_name,
                    "action": ACTION_MAP[category_name][j],
                    "prob": category_prob * action_prob,
                })
        return action_outputs


def evaluate_policy_network(expression: str, policy_net, continue_out_game) -> list:
    if "J" not in expression and "num" not in expression and "Q" not in expression:
        return None

    try:
        device = next(policy_net.parameters()).device
        token_map = create_token_map()
        tokens = parse_expression_to_tokens(expression, token_map)
        src = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        first_token = get_first_token(tokens, token_map)
        if first_token is None:
            return None

        with torch.no_grad():
            action_outputs = policy_net(
                src,
                first_token=first_token,
                continue_out_game=continue_out_game,
            )

        clean_outputs = []
        for item in action_outputs:
            prob = item["prob"]
            if isinstance(prob, torch.Tensor):
                prob = prob.detach().cpu()
                if prob.numel() != 1:
                    continue
                prob = prob.item()
            clean_outputs.append({
                "category": item["category"],
                "action": item["action"],
                "prob": float(prob),
            })
        return clean_outputs
    except Exception:
        traceback.print_exc()
        return None
