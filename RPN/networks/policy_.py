import torch
import torch.nn as nn

################################################################################
# 1) Define environment categories and sub-action mappings
################################################################################

# Main categories in the environment:
#   - feature
#   - constant
#   - num
#   - unaryop
#   - binaryop
#   - rollingop
#   - pairedrollingop
#   - stop (indicates the user has chosen to stop)
ENV_CATEGORIES = [
    "feature",
    "constant",
    "num",
    "unaryop",
    "binaryop",
    "rollingop",
    "pairedrollingop",
    "stop",
]

# Possible sub-actions for each category
ENV_ACTIONS = {
    "feature": ["high", "low", "volume", "open", "close", "vwap"],
    "constant": [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1],
    "num": [20, 30, 40],
    "unaryop": ["Log", "Abs", "Sign", "CSRank"],
    "binaryop": ["Add", "Sub", "Mul", "Div", "Greater", "Less", "Pow"],
    "rollingop": [
        "Ref", "Skew", "Kurt", "Mean", "Sum", "Std", "Var",
        "Max", "Min", "Med", "Rank", "Mad", "Delta", "WMA", 'EMA'
    ],
    "pairedrollingop": ["Cov", "Corr"],
    "stop": [None],  # stop action has no sub-action values, represented by None
}

################################################################################
# 2) Policy Network Definition
################################################################################

class PolicyNetwork(nn.Module):
    def __init__(self, feature_extractor, embed_dim):
        """
        Parameters:
          - feature_extractor: A FeatureExtractor instance to encode token sequences into feature vectors
          - embed_dim: The dimension of the output from the feature_extractor (also the input dimension of the MLP)
        """
        super(PolicyNetwork, self).__init__()
        self.feature_extractor = feature_extractor

        # Two hidden layers
        self.hidden_layer1 = nn.Linear(embed_dim, 64)
        self.hidden_layer2 = nn.Linear(64, embed_dim)

        # Category head, output the probabilities of ENV_CATEGORIES (8 categories)
        self.category_head = nn.Linear(embed_dim, len(ENV_CATEGORIES))

        # Action heads, output the distribution for each sub-action of each category
        self.action_heads = nn.ModuleDict({
            cat: nn.Linear(embed_dim, len(ENV_ACTIONS[cat])) for cat in ENV_CATEGORIES
        })

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, possible_actions):
        """
        Parameters:
          - src: The token sequence tensor, shape (batch_size, seq_len)
          - possible_actions: A list of valid actions returned by the environment,
            e.g., [("feature", "low"), ("binaryop", "Add"), ("stop", None)]
        
        Returns:
          - A list where each element is { "category": str, "action": Any, "prob": float }
            It includes only valid actions, and the probabilities sum to 1.
        """
        # 1. Use the FeatureExtractor to get state features
        features = self.feature_extractor(src)

        # 2. Pass through two hidden layers
        h = self.relu(self.hidden_layer1(features))
        h = self.relu(self.hidden_layer2(h))

        # 3. Compute category logits -> probabilities
        cat_logits = self.category_head(h)     # shape=(batch_size, len(ENV_CATEGORIES))
        cat_probs = self.softmax(cat_logits)   # shape=(batch_size, len(ENV_CATEGORIES))

        # If batch_size=1, take the 0th element
        cat_probs = cat_probs[0]  # Remove batch dimension

        # If only 1 category remains, cat_probs.shape may be (1,) or even a scalar
        if cat_probs.dim() == 0:
            # Only 1 category, cat_probs.item() must be 1.0 (since softmax only has one value)
            only_cat_p = cat_probs

            # Get the corresponding action_head for this category
            cat_name = ENV_CATEGORIES[0]
            sub_logits = self.action_heads[cat_name](h)[0]  # (len(ENV_ACTIONS[cat_name]),)
            sub_probs = self.softmax(sub_logits)  # shape=(len(ENV_ACTIONS[cat_name]),)

            # Create a dictionary of valid actions for each category
            valid_dict = {cat: [] for cat in ENV_CATEGORIES}
            for cat, val in possible_actions:
                if cat in valid_dict:
                    valid_dict[cat].append(val)

            sub_prob_list = []
            cat_sub_total = torch.tensor(0.0, device=h.device)
            for sub_idx, sub_val in enumerate(ENV_ACTIONS[cat_name]):
                raw_sub_p = sub_probs[sub_idx]
                if sub_val in valid_dict[cat_name]:
                    sub_p = only_cat_p * raw_sub_p  # Only 1 category => cat_p = only_cat_p
                    sub_prob_list.append((sub_val, sub_p))
                    cat_sub_total = cat_sub_total + sub_p

            # Normalize the probabilities
            if cat_sub_total.detach().item() < 1e-12:
                return []
            masked_output = []
            for (v, p) in sub_prob_list:
                prob = p / cat_sub_total
                masked_output.append({
                    "category": cat_name,
                    "action": v,
                    "prob": prob,
                    "prob_value": prob.detach().item()
                })
            masked_output.sort(key=lambda x: x["prob_value"], reverse=True)
            return masked_output

        else:
            # Multiple categories, cat_probs.shape = (len(ENV_CATEGORIES),)
            subaction_probs = {}
            for cat in ENV_CATEGORIES:
                logits = self.action_heads[cat](h)  # (batch_size, len(ENV_ACTIONS[cat]))
                subaction_probs[cat] = self.softmax(logits)[0]  # shape=(len(ENV_ACTIONS[cat]),)

            # Create a dictionary of valid actions for each category
            valid_dict = {cat: [] for cat in ENV_CATEGORIES}
            for cat, val in possible_actions:
                if cat in valid_dict:
                    valid_dict[cat].append(val)

            masked_output = []
            total_prob = torch.tensor(0.0, device=h.device)

            # Iterate through all categories
            for cat_idx, cat_name in enumerate(ENV_CATEGORIES):
                cat_p = cat_probs[cat_idx]
                if len(valid_dict[cat_name]) == 0:
                    continue  # No valid actions in this category
                sub_probs = subaction_probs[cat_name]  # tensor (n_subactions,)
                cat_sub_total = torch.tensor(0.0, device=h.device)
                sub_prob_list = []
                for sub_idx, sub_val in enumerate(ENV_ACTIONS[cat_name]):
                    raw_sub_p = sub_probs[sub_idx]
                    if sub_val in valid_dict[cat_name]:
                        sub_p = cat_p * raw_sub_p
                        sub_prob_list.append((sub_val, sub_p))
                        cat_sub_total = cat_sub_total + sub_p
                if cat_sub_total.detach().item() > 1e-12:
                    for (v, p) in sub_prob_list:
                        masked_output.append({
                            "category": cat_name,
                            "action": v,
                            "prob": p
                        })
                    total_prob = total_prob + cat_sub_total

            # Normalize probabilities
            if total_prob.detach().item() < 1e-12:
                return []
            for item in masked_output:
                prob = item["prob"] / total_prob
                item["prob"] = prob
                item["prob_value"] = prob.detach().item()

            masked_output.sort(key=lambda x: x["prob_value"], reverse=True)
            return masked_output


def evaluate_policy_network(expression_tokens, policy_net, possible_actions=None):
    """
    Given an expression, use the policy network to predict the distribution of valid actions.
    
    Parameters:
      expression_token: List[int] or other integer sequence representing the token sequence of a state/expressed expression
      policy_net: The policy network (which includes FeatureExtractor, etc.)
      possible_actions: List of valid actions returned by the environment, e.g., [("feature", "low"), ("binaryop", "Add"), ("stop", None)]
                        
    Returns:
        The masked action distribution predicted by the policy network (in list form)
    """
    # Get the device where the policy network is located
    device = next(policy_net.parameters()).device
        
    # Construct the input tensor, shape (batch_size, sequence_length)
    src = torch.tensor(expression_tokens, dtype=torch.long, device=device).unsqueeze(0)

    # Call the policy network with possible actions
    action_outputs = policy_net(src, possible_actions=possible_actions)
        
    return action_outputs
