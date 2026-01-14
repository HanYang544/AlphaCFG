import torch
import torch.nn as nn

# LSTM embedding and feature extractor
class ExpressionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(ExpressionEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        token_embeds = self.token_embedding(x)
        return token_embeds

class FeatureExtractor(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, nlayers, dropout=0.1):
        super(FeatureExtractor, self).__init__()
        self.embedding = ExpressionEmbedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, nlayers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, embed_dim)

    def forward(self, src):
        src_embedded = self.embedding(src)
        # LSTM expects input of shape (batch_size, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(src_embedded)
        # We use the last hidden state (hn) as the pooled output
        pooled_output = hn[-1]  # Taking the last layer's hidden state
        features = self.fc(pooled_output)
        return features
















