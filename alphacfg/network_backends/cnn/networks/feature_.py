"""CNN expression encoder compatible with CFG-Sem-k runners."""

from __future__ import annotations

import torch
import torch.nn as nn

import env.token_v as token_v


def _token_tools():
    if hasattr(token_v, "create_token_map"):
        return token_v.create_token_map(), getattr(token_v, "parse_expression_to_tokens", None)
    if hasattr(token_v, "ENV_TOKEN_MAP"):
        return token_v.ENV_TOKEN_MAP, None
    raise AttributeError("env.token_v must provide create_token_map() or ENV_TOKEN_MAP")


class FeatureExtractor(nn.Module):
    """Encode expression token sequences with 1D convolutions."""

    def __init__(
        self,
        vocab_size: int | None = None,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (1, 2, 3),
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        token_map, parser = _token_tools()
        self.token_map = token_map
        self.parser = parser
        self.vocab_size = vocab_size or (max(token_map.values()) + 1)
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), embed_dim)

    def _to_tensor(self, src: str | torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if isinstance(src, str):
            if self.parser is None:
                tokens = [self.token_map.get(token, 0) for token in src.split()]
            else:
                tokens = self.parser(src, self.token_map)
            return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        if src.dim() == 1:
            return src.to(device).unsqueeze(0)
        return src.to(device)

    def forward(self, src: str | torch.Tensor) -> torch.Tensor:
        x = self._to_tensor(src)
        embedded = self.embedding(x).permute(0, 2, 1)
        conv_outputs = []
        seq_len = embedded.shape[-1]
        for conv in self.convs:
            if conv.kernel_size[0] > seq_len:
                continue
            y = torch.relu(conv(embedded))
            conv_outputs.append(torch.max(y, dim=2)[0])
        if not conv_outputs:
            pooled = torch.zeros(x.shape[0], self.fc.in_features, device=x.device)
        else:
            pooled = torch.cat(conv_outputs, dim=1)
            if pooled.shape[1] < self.fc.in_features:
                pad = torch.zeros(x.shape[0], self.fc.in_features - pooled.shape[1], device=x.device)
                pooled = torch.cat([pooled, pad], dim=1)
        return self.fc(self.dropout(pooled))
