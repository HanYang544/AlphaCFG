"""Sequence LSTM feature extractor compatible with CFG-Sem-k runners."""

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


class ExpressionEmbedding(nn.Module):
    """Token embedding layer for prefix expression token ids."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(x)


class FeatureExtractor(nn.Module):
    """Encode an expression string or token tensor with a standard LSTM."""

    def __init__(
        self,
        vocab_size: int | None = None,
        embed_dim: int = 128,
        hidden_dim: int | None = None,
        h_size: int | None = None,
        nlayers: int = 2,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        token_map, parser = _token_tools()
        self.vocab_size = vocab_size or (max(token_map.values()) + 1)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or h_size or embed_dim
        self.nlayers = nlayers
        self.token_map = token_map
        self.parser = parser

        self.embedding = ExpressionEmbedding(self.vocab_size, embed_dim)
        lstm_dropout = dropout if nlayers > 1 else 0.0
        self.lstm = nn.LSTM(
            embed_dim,
            self.hidden_dim,
            nlayers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(self.hidden_dim, embed_dim)

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
        src_tensor = self._to_tensor(src)
        src_embedded = self.embedding(src_tensor)
        _, (hn, _) = self.lstm(src_embedded)
        pooled_output = hn[-1]
        return self.fc(pooled_output)
