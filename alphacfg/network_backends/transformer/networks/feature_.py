"""Transformer expression encoder compatible with CFG-Sem-k runners."""

from __future__ import annotations

import math

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
    """Encode expression token sequences with a Transformer encoder."""

    def __init__(
        self,
        vocab_size: int | None = None,
        embed_dim: int = 128,
        nhead: int = 4,
        nhid: int = 256,
        nlayers: int = 2,
        dropout: float = 0.1,
        max_len: int = 128,
        **_: object,
    ):
        super().__init__()
        token_map, parser = _token_tools()
        self.token_map = token_map
        self.parser = parser
        self.vocab_size = vocab_size or (max(token_map.values()) + 1)
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        self.register_buffer("positional_encoding", self._make_positional_encoding(max_len, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=nhid,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.fc = nn.Linear(embed_dim, embed_dim)

    @staticmethod
    def _make_positional_encoding(max_len: int, embed_dim: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: pe[0, :, 1::2].shape[-1]])
        return pe

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
        seq_len = x.shape[1]
        embedded = self.embedding(x) + self.positional_encoding[:, :seq_len, :].to(x.device)
        memory = self.encoder(embedded)
        return self.fc(memory.mean(dim=1))
