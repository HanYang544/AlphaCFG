def compute_expression_similarity(expr_str1, expr_str2, token_map=None):
    """
    Calculate the normalized similarity between two expression strings.
    
    Parameters:
        expr_str1: The first expression string, e.g., "Div Max $volume 40 Med $open 40"
        expr_str2: The second expression string, e.g., "Med $open 40"
        token_map: Optional parameter, if not provided, a new TokenMap object will be created internally
    
    Returns:
        A float value representing the normalized similarity, ranging from 0 to 1.
    """
    # If no token_map is provided, create a new TokenMap instance
    if token_map is None:
        token_map = TokenMap()
    
    # Parse the expression strings into expression trees
    tree1 = parse_expression(expr_str1, token_map)
    tree2 = parse_expression(expr_str2, token_map)
    
    # Calculate the normalized similarity
    similarity = normalize_similarity(tree1, tree2)
    return similarity

# The following is the internal code used

# Define which nodes have children with order insensitivity (can be matched in any order)
UNORDERED_NODES = {
    'Cov',   # arity=3, the order of the first two children is insensitive
    'Corr',  # arity=3
    'Add1',  # arity=2
    'Mul1',  # arity=2
    'Greater1',  # arity=2
    'Less1',     # arity=2
}

import re

from data.expression import *
from env.features import *

class TokenMap:
    def __init__(self):
        self.token_map = {
            # ========== 1) Unary operators ========== 
            'Log': 1,
            'Abs': 2,
            'Sign': 3,
            'CSRank': 4,

            # ========== 2) Binary operators ========== 
            'Add1': 5,
            'Add2': 6,
            'Sub1': 7,
            'Sub2': 8,
            'Sub3': 9,
            'Mul1': 10,
            'Mul2': 11,
            'Div1': 12,
            'Div2': 13,
            'Div3': 14,
            'Pow1': 15,
            'Pow2': 16,
            'Pow3': 17,
            'Greater1': 18,
            'Greater2': 19,
            'Less1': 20,
            'Less2': 21,

            # ========== 3) Rolling operators ========== 
            'Ref': 22,
            'Skew': 23,
            'Kurt': 24,
            'Mean': 25,
            'Sum': 26,
            'Std': 27,
            'Var': 28,
            'Max': 29,
            'Min': 30,
            'Med': 31,
            'Rank': 32,
            'Mad': 33,
            'Delta': 34,
            'WMA': 35,
            'EMA': 36,  # Added EMA

            # ========== 4) Paired rolling operators ========== 
            'Cov': 37,
            'Corr': 38,

            # ========== 5) Features ========== 
            '$high': 39,
            '$low': 40,
            '$volume': 41,
            '$open': 42,
            '$close': 43,
            '$vwap': 44,

            # ========== 6) Special tokens ========== 
            'J': 45,         # Non-terminal E
            'Q': 46,         # Non-terminal Q
            'num': 47,       # Used for placeholder numbers (num_range)

            # ========== 7) Constants ========== 
            '-0.1': 48,
            '-0.05': 49,
            '-0.01': 50,
            '0.01': 51,
            '0.05': 52,
            '0.1': 53,

            # ========== 8) Numbers (num_range) ========== 
            '40': 54,
            '30': 55,
            '20': 56
        }

        # ========== Operator arity (number of arguments) ========== 
        self.operator_arities = {
            # Unary operators
            'Log': 1,
            'Abs': 1,
            'Sign': 1,
            'CSRank': 1,

            # Binary operators
            'Add1': 2,
            'Add2': 2,
            'Sub1': 2,
            'Sub2': 2,
            'Sub3': 2,
            'Mul1': 2,
            'Mul2': 2,
            'Div1': 2,
            'Div2': 2,
            'Div3': 2,
            'Pow1': 2,
            'Pow2': 2,
            'Pow3': 2,
            'Greater1': 2,
            'Greater2': 2,
            'Less1': 2,
            'Less2': 2,

            # Rolling operators
            'Ref': 2,
            'Skew': 2,
            'Kurt': 2,
            'Mean': 2,
            'Sum': 2,
            'Std': 2,
            'Var': 2,
            'Max': 2,
            'Min': 2,
            'Med': 2,
            'Rank': 2,
            'Mad': 2,
            'Delta': 2,
            'WMA': 2,
            'EMA': 2,

            # Paired rolling operators
            'Cov': 3,
            'Corr': 3
        }

    def get_word_id(self, token):
        """
        Return the ID of the token, default is 0.
        """
        return self.token_map.get(token, 0)

    def get_arity(self, operator):
        """
        Return the number of arguments for the operator, default is 0 (arity not defined).
        """
        return self.operator_arities.get(operator, 0)

class ExpressionTree:
    def __init__(self, value=None, children=None, is_operator=False):
        """
        Initialize an expression tree node
        :param value: The value of the node, e.g., 'Add', '$high', '0.05', etc.
        :param children: List of child nodes
        :param is_operator: Whether it is an operator node
        """
        self.value = value
        self.children = children if children is not None else []
        self.is_operator = is_operator

    def is_leaf(self):
        """Check if it is a leaf node"""
        return len(self.children) == 0

    def __repr__(self):
        """Return the string representation of the node"""
        if self.is_leaf():
            return f"Leaf({self.value})"
        else:
            children_repr = ", ".join([str(child) for child in self.children])
            return f"({self.value}({children_repr}))"

def parse_expression(expression_str, token_map):
    """
    Parse an expression string without parentheses and build a syntax tree
    :param expression_str: The expression string, e.g., 'Rank Corr $high 0.05 40 40'
    :param token_map: TokenMap object
    :return: ExpressionTree object
    """
    tokens = expression_str.strip().split()
    tokens = tokens[::-1]  # Reverse for efficient popping from the end

    def helper():
        if not tokens:
            return None
        token = tokens.pop()
        if token in token_map.operator_arities:
            arity = token_map.get_arity(token)
            children = []
            for _ in range(arity):
                child = helper()
                if child is not None:
                    children.append(child)
            return ExpressionTree(value=token, children=children, is_operator=True)
        else:
            # It's either a variable or a constant/number
            return ExpressionTree(value=token, is_operator=False)

    tree = helper()
    return tree

from itertools import permutations

def common_subtree_size(t1, t2):
    if t1.value != t2.value or len(t1.children) != len(t2.children):
        return 0

    value = t1.value

    # ===== Special handling: binary operators with completely order-insensitive children =====
    if value in {'Add1', 'Mul1', 'Greater1', 'Less1'} and len(t1.children) == 2:
        return _common_subtree_size_binary_unordered(t1, t2)

    # ===== Special handling: Cov / Corr, first two children order-insensitive, third sensitive =====
    elif value in {'Cov', 'Corr'} and len(t1.children) == 3:
        return _common_subtree_size_cov_corr(t1, t2)

    # ===== Default case: all children are order-sensitive =====
    else:
        return _common_subtree_size_ordered(t1, t2)

def _common_subtree_size_ordered(t1, t2):
    size = 1  # Current node matches
    for child1, child2 in zip(t1.children, t2.children):
        child_size = common_subtree_size(child1, child2)
        if child_size == 0:
            return 0
        size += child_size
    return size


def _common_subtree_size_binary_unordered(t1, t2):
    size = 1  # Current node matches

    c1_0, c1_1 = t1.children
    c2_0, c2_1 = t2.children

    # Try both possible order combinations
    # Case 1: Original order (c1_0 ↔ c2_0, c1_1 ↔ c2_1)
    size1 = common_subtree_size(c1_0, c2_0)
    size2 = common_subtree_size(c1_1, c2_1)
    if size1 > 0 and size2 > 0:
        size += size1 + size2
        return size

    # Case 2: Swapped order (c1_0 ↔ c2_1, c1_1 ↔ c2_0)
    size1 = common_subtree_size(c1_0, c2_1)
    size2 = common_subtree_size(c1_1, c2_0)
    if size1 > 0 and size2 > 0:
        size += size1 + size2
        return size

    # Neither permutation works
    return 0

def _common_subtree_size_cov_corr(t1, t2):
    size = 1  # Current node matches

    c1_0, c1_1, c1_2 = t1.children
    c2_0, c2_1, c2_2 = t2.children

    # Third child (index 2) is order-sensitive
    size3 = common_subtree_size(c1_2, c2_2)
    if size3 == 0:
        return 0
    size += size3

    # First two children (index 0,1) are order-insensitive, try both permutations
    # Case 1: Original order
    size1 = common_subtree_size(c1_0, c2_0)
    size2 = common_subtree_size(c1_1, c2_1)
    if size1 > 0 and size2 > 0:
        size += size1 + size2
        return size

    # Case 2: Swapped order
    size1 = common_subtree_size(c1_0, c2_1)
    size2 = common_subtree_size(c1_1, c2_0)
    if size1 > 0 and size2 > 0:
        size += size1 + size2
        return size

    # No matching permutation
    return 0

def get_all_subtrees(tree):
    """
    Recursively get all subtrees of an expression tree (including itself).
    """
    subtrees = [tree]
    for child in tree.children:
        subtrees.extend(get_all_subtrees(child))
    return subtrees

def max_common_subtree_size(tree1, tree2):
    """
    Traverse all subtrees of both trees and calculate the maximum common subtree size.
    """
    subtrees1 = get_all_subtrees(tree1)
    subtrees2 = get_all_subtrees(tree2)
    max_size = 0
    for t1 in subtrees1:
        for t2 in subtrees2:
            size = common_subtree_size(t1, t2)
            if size > max_size:
                max_size = size
    return max_size

def normalize_similarity(tree1, tree2):
    """
    Normalize the maximum common subtree size, using the formula:
    similarity = max_common_subtree_size / max(total_nodes(tree1), total_nodes(tree2))
    """
    def total_nodes(tree):
        return 1 + sum(total_nodes(child) for child in tree.children)
    
    max_common = max_common_subtree_size(tree1, tree2)
    total1 = total_nodes(tree1)
    total2 = total_nodes(tree2)
    norm_sim = max_common / max(total1, total2)
    return norm_sim

def expression_to_prefix_str(expr: Expression) -> str:
    """
    Convert an Expression instance to a prefix expression string.
    For example, for the expression Div(Max($volume,40), Med($open,40))
    it returns "Div Max $volume 40 Med $open 40"
    """
    # If it's a constant, directly return the number string
    if isinstance(expr, Constant):
        return str(expr._value)
    # If it's a feature (Feature), directly call its __str__ method (usually returns something like "$xxx")
    elif isinstance(expr, Feature):
        return str(expr)
    # If it's a unary operator
    elif isinstance(expr, UnaryOperator):
        return f"{type(expr).__name__} " + expression_to_prefix_str(expr._operand)
    # If it's a binary operator
    elif isinstance(expr, BinaryOperator):
        return f"{type(expr).__name__} " + expression_to_prefix_str(expr._lhs) + " " + expression_to_prefix_str(expr._rhs)
    # If it's a rolling operator (RollingOperator), assuming the second parameter is the window length
    elif isinstance(expr, RollingOperator):
        return f"{type(expr).__name__} " + expression_to_prefix_str(expr._operand) + " " + str(expr._delta_time)
    # If it's a paired rolling operator (PairRollingOperator)
    elif isinstance(expr, PairRollingOperator):
        return f"{type(expr).__name__} " + expression_to_prefix_str(expr._lhs) + " " + expression_to_prefix_str(expr._rhs) + " " + str(expr._delta_time)
    else:
        # Other cases, directly use str(expr)
        return str(expr)
    
# Test the wrapped function
if __name__ == "__main__":
    expr1 = Mad(Sign(Mul(Constant(0.05),vwap)),20)
    expr2 = vwap
    #expr_str1 = "$open"
    #expr_str2 = "Med $open num"
    expr_str1 = expression_to_prefix_str(expr1)
    expr_str2 = expression_to_prefix_str(expr2)
    sim = compute_expression_similarity(expr_str1, expr_str2)
    print("Normalized similarity:", sim)