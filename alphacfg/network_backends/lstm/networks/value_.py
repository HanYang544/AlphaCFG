"""Value network for the LSTM backend."""

from __future__ import annotations

import torch
import torch.nn as nn

from env.token_v import create_token_map, parse_expression_to_tokens


class ValueNetwork(nn.Module):
    """Scalar value head on top of an LSTM expression encoder."""

    def __init__(self, feature_extractor, embed_dim: int):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, src):
        features = self.feature_extractor(src)
        return self.value_head(features)


def evaluate_expression_value(expression, value_net):
    device = next(value_net.parameters()).device
    token_map = create_token_map()
    tokens = parse_expression_to_tokens(expression, token_map)
    src = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        value_output = value_net(src)
    return value_output.item()
