import torch
import torch.nn as nn


# Embedding layer for token ids (unchanged)
class ExpressionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Token embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the embedding vectors.
        """
        super(ExpressionEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        Forward pass for token embeddings.

        Args:
            x (torch.Tensor): Input tensor of token ids, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Embedded tokens, shape (batch_size, seq_len, embed_dim).
        """
        token_embeds = self.token_embedding(x)  # (batch_size, seq_len, embed_dim)
        return token_embeds


class FeatureExtractor(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters=128, kernel_sizes=[1], dropout=0.1):
        """
        1D CNN-based feature extractor for sequence data (e.g., text).

        Instead of using an LSTM, this model applies multiple 1D convolutional filters
        over the embedded tokens, followed by global max pooling and a fully-connected layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of the token embeddings.
            num_filters (int, optional): Number of output channels (filters) for each convolutional layer. Default is 128.
            kernel_sizes (list of int, optional): List of convolution kernel sizes (e.g., [1, 2, 3]). Default is [1].
            dropout (float, optional): Dropout probability. Default is 0.1.
        """
        super(FeatureExtractor, self).__init__()
        self.embedding = ExpressionEmbedding(vocab_size, embed_dim)

        # Create multiple 1D convolution layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer to map concatenated CNN outputs to desired feature dimension (e.g., embed_dim)
        # Input dim: num_filters * len(kernel_sizes)
        # Output dim: embed_dim (can be modified if needed)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), embed_dim)

    def forward(self, src):
        """
        Forward pass of the feature extractor.

        Args:
            src (torch.Tensor): Input tensor of token ids, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Extracted feature vector, shape (batch_size, embed_dim).
        """
        # Embed the input tokens
        src_embedded = self.embedding(src)  # (batch_size, seq_len, embed_dim)

        # Conv1d expects input in the format: (batch_size, in_channels, seq_len)
        # So we permute the dimensions from (B, L, C) -> (B, C, L)
        src_embedded = src_embedded.permute(0, 2, 1)  # (B, embed_dim, seq_len)

        # Apply each convolutional filter, followed by ReLU and global max pooling
        conv_outputs = []
        for conv in self.convs:
            # Convolution: (B, embed_dim, seq_len) -> (B, num_filters, seq_len - k + 1)
            conv_out = conv(src_embedded)
            conv_out = torch.relu(conv_out)  # Activation

            # Global max pooling over the sequence dimension -> (B, num_filters)
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)

        # Concatenate the outputs from all different convolutional kernels
        # Shape: (B, num_filters * len(kernel_sizes))
        cat_features = torch.cat(conv_outputs, dim=1)

        # Apply dropout for regularization
        features = self.dropout(cat_features)

        # Final fully-connected layer to map to desired feature dimension (e.g., embed_dim)
        features = self.fc(features)  # (B, embed_dim)

        return features