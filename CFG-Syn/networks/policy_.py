import torch
import torch.nn as nn
from env.token_v import create_token_map, parse_expression_to_tokens, get_first_token
from networks.feature_ import FeatureExtractor 

# Category name mapping
CATEGORY_MAP = {
    0: 'Vari',
    1: 'constant',
    2: 'unaryop',
    3: 'binaryop',
    4: 'rollingop',
    5: 'pairedrollingop',
    6: 'num'
}

# Action mapping
ACTION_MAP = {
    'unaryop': ['Log', 'Abs', 'Sign', 'CSRank'],
    'binaryop': ['Add', 'Sub' , 'Mul' , 'Div' , 'Pow' , 'Greater', 'Less'],
    'rollingop': ['Ref', 'Skew', 'Kurt', 'Mean', 'Sum', 'Std', 'Var', 'Max', 'Min', 'Med', 'Rank', 'Mad', 'Delta', 'WMA', 'EMA'],
    'pairedrollingop': ['Cov', 'Corr'],
    'Vari': ['high', 'low', 'volume', 'open', 'close', 'vwap'],
    'constant': [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1],
    'num': [20, 30, 40]
}

class PolicyNetwork(nn.Module):
    def __init__(self, feature_extractor, embed_dim):
        super(PolicyNetwork, self).__init__()
        self.feature_extractor = feature_extractor

        # Add two hidden layers
        self.hidden_layer1 = nn.Linear(embed_dim, 64)
        self.hidden_layer2 = nn.Linear(64, embed_dim)

        # Output dimensions for each category
        self.category_head = nn.Linear(embed_dim, len(CATEGORY_MAP))
        # Specific action mappings for each category, using nn.ModuleDict
        self.action_heads = nn.ModuleDict({
            'unaryop': nn.Linear(embed_dim, len(ACTION_MAP['unaryop'])),
            'binaryop': nn.Linear(embed_dim, len(ACTION_MAP['binaryop'])),
            'rollingop': nn.Linear(embed_dim, len(ACTION_MAP['rollingop'])),
            'pairedrollingop': nn.Linear(embed_dim, len(ACTION_MAP['pairedrollingop'])),
            'Vari': nn.Linear(embed_dim, len(ACTION_MAP['Vari'])),
            'constant': nn.Linear(embed_dim, len(ACTION_MAP['constant'])),
            'num': nn.Linear(embed_dim, len(ACTION_MAP['num'])),
        })
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, expression: str, first_token=None) -> list:
        features = self.feature_extractor(expression)
        features = torch.relu(self.hidden_layer1(features))
        features = torch.relu(self.hidden_layer2(features))

        # Get device
        device = features.device

        # Define category mask
        action_mask = {category: 1 for category in CATEGORY_MAP}  # Initialize to all 1, meaning all categories and actions are valid

        # Update mask based on first_token
        if first_token == 'Q':
            action_mask.update({category: 0 for category in ['num']})  # Disable 'constant' and 'num'
        elif first_token == 'num':
            action_mask.update({category: 0 for category in ['unaryop', 'binaryop', 'rollingop', 'pairedrollingop', 'Vari', 'constant']})  # Only allow 'num'

        # Category probability calculation
        category_logits = self.category_head(features)
        category_probs = self.softmax(category_logits).squeeze(0)

        # Apply category mask
        for i, category_name in CATEGORY_MAP.items():
            if category_name in action_mask and action_mask[category_name] == 0:
                category_probs[i] = 0  # Force disabled category probability to 0

        # Normalize probabilities
        category_probs = category_probs / category_probs.sum() if category_probs.sum() > 0 else category_probs

        # Calculate action probabilities for each category
        action_outputs = []
        for i, category_prob in enumerate(category_probs):
            category_name = CATEGORY_MAP.get(i)
            if category_name is None or category_prob == 0:
                continue  # Skip if category is masked

            # Get action logits
            action_head = self.action_heads[category_name]
            action_logits = action_head(features)
            action_probs = self.softmax(action_logits).squeeze(0)

            # Calculate final probability for each action
            for j, action_prob in enumerate(action_probs):
                action_value = ACTION_MAP[category_name][j]
                prob = category_prob * action_prob  # Product of category probability and action probability
                action_outputs.append({
                    'category': category_name,
                    'action': action_value,
                    'prob': prob
                })

        # Return the probability distribution of all actions
        return action_outputs


    
def evaluate_policy_network(expression:str, policy_net)->list:
    if 'J' not in expression and 'num' not in expression and 'Q' not in expression:
        return None  # Terminal state, return None

    try:
        token_map = create_token_map()
        tokens = parse_expression_to_tokens(expression, token_map)

        first_token = get_first_token(tokens, token_map)

        if first_token is None:
            print(f"Unrecognized first symbol, expression: {expression}")
            return None
        
        # Call policy network
        action_outputs = policy_net(expression, first_token=first_token)

        # Ensure valid action outputs are returned
        if not action_outputs:
            print(f"evaluate_policy_network returned empty action outputs, expression: {expression}")
            return []

        return action_outputs

    except Exception as e:
        print(f"evaluate_policy_network encountered an exception: {e}")
        import traceback
        traceback.print_exc()  # Print full exception stack trace
        return None


# Test code
if __name__ == "__main__":
    expression = "Corr $vwap 0.05 num"
    embed_dim = 256
    h_size = 256
    dropout = 0.5

    feature_extractor = FeatureExtractor(embed_dim=embed_dim, h_size=h_size, dropout=dropout)
    policy_net = PolicyNetwork(feature_extractor=feature_extractor, embed_dim=embed_dim)
    policy_net.eval()

    action_outputs = evaluate_policy_network(expression, policy_net)
    if action_outputs:
        for output in action_outputs:
            print(output)
