import torch
import torch.nn as nn
from networks.feature_ import FeatureExtractor  # Ensure this is the LSTM-based FeatureExtractor

class ValueNetwork(nn.Module):
    def __init__(self, feature_extractor, embed_dim):
        super(ValueNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),   # First hidden layer with size 64
            nn.ReLU(),                   # Activation function: ReLU
            nn.Linear(64, 64),          # Second hidden layer with size 64
            nn.ReLU(),                   # Activation function: ReLU
            nn.Linear(64, 1)         # Output layer, producing a single value
        )

    def forward(self, src):
        features = self.feature_extractor(src)  # Extract features using the feature extractor
        value = self.value_head(features)       # Pass the features through the value head
        return value

def evaluate_expression_value(expression_tokens, value_net):
    """
    Evaluates the value of an expression given its tokenized representation.

    Parameters:
    expression_tokens: A sequence of tokens representing the expression.
    value_net: The value network model used for evaluation.

    Returns:
    The evaluated value of the expression as a scalar.
    """
    
    # Get the device where the model is located (CPU or GPU)
    device = next(value_net.parameters()).device

    # Convert the parsed token sequence into tensor format and move it to the appropriate device
    src = torch.tensor(expression_tokens, dtype=torch.long, device=device)  # (batch_size, sequence_length)

    # Use the provided value network to get the output
    value_output = value_net(src)

    return value_output.item()  # Return the scalar value as a Python float
