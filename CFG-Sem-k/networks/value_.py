import torch
import torch.nn as nn

from env.token_v import *
from networks.feature_ import *


class ValueNetwork(nn.Module):
    def __init__(self, feature_extractor, embed_dim):
        super(ValueNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),   # First hidden layer with size 64
            nn.ReLU(),                  # Activation function: ReLU
            nn.Linear(64, 64),          # Second hidden layer with size 64
            nn.ReLU(),                  # Activation function: ReLU
            nn.Linear(64, 1),           # Output layer, outputs a single value
            #nn.Sigmoid()               # Activation function: maps output to (-1, 1)
        )

    def forward(self, expression):
        feature = self.feature_extractor(expression)
        value = self.value_head(feature)
        return value


def evaluate_expression_value(expression, value_net):
    value_output = value_net(expression)  # Directly pass the expression to the network
    return value_output.item()


# Test code
if __name__ == "__main__":
    expression = "Corr $vwap E num"
    # Initialize feature extractor
    embed_dim = 256
    h_size = 256
    dropout = 0.5
    feature_extractor = FeatureExtractor(
        embed_dim=embed_dim,
        h_size=h_size,
        dropout=dropout,
        pretrained_emb=None
    )

    value_net = ValueNetwork(feature_extractor=feature_extractor, embed_dim=embed_dim)
    value_net.eval()

    value = evaluate_expression_value(expression, value_net)
    print(f"Value after reset: {value:.4f}")