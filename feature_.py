import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Transformer embedding and feature extractor
class ExpressionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=5000):
        super(ExpressionEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        positional_encoding = torch.zeros(max_len, embed_dim)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        token_embeds = self.token_embedding(x)
        seq_len = x.size(1)
        x = token_embeds + self.positional_encoding[:, :seq_len, :].to(x.device)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, nhid, nlayers, dropout=0.1, max_len=5000):
        super(FeatureExtractor, self).__init__()
        self.embedding = ExpressionEmbedding(vocab_size, embed_dim, max_len)
        encoder_layers = TransformerEncoderLayer(embed_dim, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, src, src_key_padding_mask=None):
        src_embedded = self.embedding(src)
        memory = self.transformer_encoder(src_embedded, src_key_padding_mask=src_key_padding_mask)
        pooled_output = memory.mean(dim=1)
        features = self.fc(pooled_output)
        return features

# 测试代码
if __name__ == "__main__":
    vocab_size = 100
    embed_dim = 64
    nhead = 8
    nhid = 128
    nlayers = 2
    dropout = 0.1

    feature_extractor = FeatureExtractor(
        vocab_size, embed_dim, nhead, nhid, nlayers, dropout
    )

    src = torch.randint(0, vocab_size, (1, 10))  # 随机生成一个序列
    features = feature_extractor(src)
    # print("Feature Shape:", features.shape)















