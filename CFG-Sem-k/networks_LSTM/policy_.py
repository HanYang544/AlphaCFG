import torch
import torch.nn as nn
from env.token_v import create_token_map, parse_expression_to_tokens, get_first_token
from networks.feature_ import FeatureExtractor  

# Category name mapping
CATEGORY_MAP = {
    0: 'Vari',           # Variable features (e.g., high, low, volume)
    1: 'constant',       # Constant values
    2: 'unaryop',        # Unary operators (e.g., Log, Abs)
    3: 'binaryop',       # Binary operators (e.g., Add, Sub)
    4: 'rollingop',      # Rolling operators (e.g., Mean, Sum)
    5: 'pairedrollingop', # Paired rolling operators (e.g., Cov, Corr)
    6: 'num'             # Numeric constants (e.g., 20, 30, 40)
}

# Action mapping
ACTION_MAP = {
    'unaryop': ['Log', 'Abs', 'Sign', 'CSRank'],  # Unary operator actions
    'binaryop': ['Add1', 'Add2', 'Sub1','Sub2','Sub3', 'Mul1','Mul2', 'Div1','Div2','Div3', 'Pow1','Pow2','Pow3', 'Greater1', 'Greater2', 'Less1', 'Less2'],  # Binary operator actions
    'rollingop': ['Ref', 'Skew', 'Kurt', 'Mean', 'Sum', 'Std', 'Var', 'Max', 'Min', 'Med', 'Rank', 'Mad', 'Delta', 'WMA', 'EMA'],  # Rolling operator actions
    'pairedrollingop': ['Cov', 'Corr'],  # Paired rolling operator actions
    'Vari': ['high', 'low', 'volume', 'open', 'close', 'vwap'],  # Variable features
    'constant': [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1],  # Constant values
    'num': [20, 30, 40]  # Numeric constants for window sizes
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

    def forward(self, src, first_token=None, continue_out_game=3) -> list:
        # Extract features from the expression
        features = self.feature_extractor(src)

        features = torch.relu(self.hidden_layer1(features))
        features = torch.relu(self.hidden_layer2(features))

        # Get the device
        device = features.device

        # Define category mask
        action_mask = {category: 1 for category in CATEGORY_MAP}  # Initialize to all 1s, meaning all categories and actions are valid

        # Set mask based on continue_out_game
        if continue_out_game == 3:
            pass  # No restrictions
        elif continue_out_game == 2:
            action_mask = {'pairedrollingop': 0, **{key: 1 for key in CATEGORY_MAP if key != 'pairedrollingop'}}  # Disable pairedrollingop
        elif continue_out_game == 1:
            action_mask = {category: 0 for category in ['binaryop', 'rollingop', 'pairedrollingop']}  # Disable these categories
        elif continue_out_game == 0:
            action_mask = {category: 0 for category in ['unaryop', 'binaryop', 'rollingop', 'pairedrollingop']}  # Disable these categories

        # Update mask based on first_token
        if first_token == 'J':
            action_mask.update({category: 0 for category in ['unaryop', 'binaryop', 'rollingop', 'pairedrollingop', 'Vari', 'num']})  # Only 'constant' allowed
        elif first_token == 'Q':
            action_mask.update({category: 0 for category in ['constant', 'num']})  # Disable 'constant' and 'num'
        elif first_token == 'num':
            action_mask.update({category: 0 for category in ['unaryop', 'binaryop', 'rollingop', 'pairedrollingop', 'Vari', 'constant']})  # Only allow 'num'

        # Calculate category probabilities
        category_logits = self.category_head(features)
        category_probs = self.softmax(category_logits).squeeze(0)

        # Apply category mask
        category_mask = torch.tensor(
            [
                float(action_mask.get(CATEGORY_MAP[i], 1))
                for i in range(len(CATEGORY_MAP))
            ],
            dtype=category_probs.dtype,
            device=category_probs.device
        )
        category_probs = category_probs * category_mask

        # Normalize probabilities
        category_probs = category_probs / (category_probs.sum() + 1e-10)

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


    
def evaluate_policy_network(expression: str, policy_net, continue_out_game) -> list:
    if 'J' not in expression and 'num' not in expression and 'Q' not in expression:
        return None  # Terminal state, return None

    try:
        # Get the device where the policy network is located
        device = next(policy_net.parameters()).device
        token_map = create_token_map()

        tokens = parse_expression_to_tokens(expression, token_map)

        src = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        first_token = get_first_token(tokens, token_map)

        if first_token is None:
            print(f"Unrecognized first symbol, expression: {expression}")
            return None
        
        # MCTS search does not need gradient.
        with torch.no_grad():
            action_outputs = policy_net(src, first_token=first_token, continue_out_game=continue_out_game)

        # Ensure valid action outputs are returned
        if not action_outputs:
            print(f"evaluate_policy_network returned empty action outputs, expression: {expression}")
            return []

        # Convert tensor probabilities to Python floats for MCTS.
        clean_outputs = []
        for item in action_outputs:
            prob = item['prob']

            if isinstance(prob, torch.Tensor):
                prob = prob.detach().cpu()

                if prob.numel() != 1:
                    print("Invalid prior_prob shape in evaluate_policy_network:", prob.shape)
                    print("item:", item)
                    continue

                prob = prob.item()

            clean_outputs.append({
                'category': item['category'],
                'action': item['action'],
                'prob': float(prob)
            })

        return clean_outputs

    except Exception as e:
        print(f"evaluate_policy_network encountered an exception: {e}")
        import traceback
        traceback.print_exc()  # Print the full exception stack trace
        return None


