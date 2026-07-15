"""Prefix-expression tree similarity helpers used by alpha pools."""

from __future__ import annotations

from alphacfg.data.expression import (
    BinaryOperator,
    Constant,
    Expression,
    Feature,
    PairRollingOperator,
    RollingOperator,
    UnaryOperator,
)


class TokenMap:
    """Token arity table for CFG-style prefix expressions."""

    def __init__(self):
        self.operator_arities = {
            "Log": 1,
            "Abs": 1,
            "Sign": 1,
            "CSRank": 1,
            "Add1": 2,
            "Add2": 2,
            "Sub1": 2,
            "Sub2": 2,
            "Sub3": 2,
            "Mul1": 2,
            "Mul2": 2,
            "Div1": 2,
            "Div2": 2,
            "Div3": 2,
            "Pow1": 2,
            "Pow2": 2,
            "Pow3": 2,
            "Greater1": 2,
            "Greater2": 2,
            "Greater3": 2,
            "Less1": 2,
            "Less2": 2,
            "Less3": 2,
            "Ref": 2,
            "Skew": 2,
            "Kurt": 2,
            "Mean": 2,
            "Sum": 2,
            "Std": 2,
            "Var": 2,
            "Max": 2,
            "Min": 2,
            "Med": 2,
            "Rank": 2,
            "Mad": 2,
            "Delta": 2,
            "WMA": 2,
            "EMA": 2,
            "Cov": 3,
            "Corr": 3,
        }

    def get_arity(self, operator: str) -> int:
        return self.operator_arities.get(operator, 0)


class ExpressionTree:
    """Small immutable-ish tree node for subtree matching."""

    def __init__(self, value=None, children=None, is_operator=False):
        self.value = value
        self.children = children if children is not None else []
        self.is_operator = is_operator


def parse_expression(expression_str: str, token_map: TokenMap) -> ExpressionTree:
    tokens = expression_str.strip().split()[::-1]

    def helper():
        if not tokens:
            return None
        token = tokens.pop()
        if token in token_map.operator_arities:
            children = []
            for _ in range(token_map.get_arity(token)):
                child = helper()
                if child is not None:
                    children.append(child)
            return ExpressionTree(value=token, children=children, is_operator=True)
        return ExpressionTree(value=token, is_operator=False)

    return helper()


def compute_expression_similarity(expr_str1: str, expr_str2: str, token_map: TokenMap | None = None) -> float:
    token_map = token_map or TokenMap()
    tree1 = parse_expression(expr_str1, token_map)
    tree2 = parse_expression(expr_str2, token_map)
    return normalize_similarity(tree1, tree2)


def common_subtree_size(t1: ExpressionTree, t2: ExpressionTree) -> int:
    if t1.value != t2.value or len(t1.children) != len(t2.children):
        return 0
    if t1.value in {"Add1", "Mul1"} and len(t1.children) == 2:
        return _common_subtree_size_binary_unordered(t1, t2)
    if t1.value in {"Cov", "Corr"} and len(t1.children) == 3:
        return _common_subtree_size_cov_corr(t1, t2)
    return _common_subtree_size_ordered(t1, t2)


def _common_subtree_size_ordered(t1: ExpressionTree, t2: ExpressionTree) -> int:
    size = 1
    for child1, child2 in zip(t1.children, t2.children):
        child_size = common_subtree_size(child1, child2)
        if child_size == 0:
            return 0
        size += child_size
    return size


def _common_subtree_size_binary_unordered(t1: ExpressionTree, t2: ExpressionTree) -> int:
    size1 = common_subtree_size(t1.children[0], t2.children[0])
    size2 = common_subtree_size(t1.children[1], t2.children[1])
    if size1 > 0 and size2 > 0:
        return 1 + size1 + size2
    size1 = common_subtree_size(t1.children[0], t2.children[1])
    size2 = common_subtree_size(t1.children[1], t2.children[0])
    if size1 > 0 and size2 > 0:
        return 1 + size1 + size2
    return 0


def _common_subtree_size_cov_corr(t1: ExpressionTree, t2: ExpressionTree) -> int:
    size3 = common_subtree_size(t1.children[2], t2.children[2])
    if size3 == 0:
        return 0
    size1 = common_subtree_size(t1.children[0], t2.children[0])
    size2 = common_subtree_size(t1.children[1], t2.children[1])
    if size1 > 0 and size2 > 0:
        return 1 + size1 + size2 + size3
    size1 = common_subtree_size(t1.children[0], t2.children[1])
    size2 = common_subtree_size(t1.children[1], t2.children[0])
    if size1 > 0 and size2 > 0:
        return 1 + size1 + size2 + size3
    return 0


def get_all_subtrees(tree: ExpressionTree) -> list[ExpressionTree]:
    subtrees = [tree]
    for child in tree.children:
        subtrees.extend(get_all_subtrees(child))
    return subtrees


def max_common_subtree_size(tree1: ExpressionTree, tree2: ExpressionTree) -> int:
    max_size = 0
    for subtree1 in get_all_subtrees(tree1):
        for subtree2 in get_all_subtrees(tree2):
            max_size = max(max_size, common_subtree_size(subtree1, subtree2))
    return max_size


def total_nodes(tree: ExpressionTree) -> int:
    return 1 + sum(total_nodes(child) for child in tree.children)


def normalize_similarity(tree1: ExpressionTree, tree2: ExpressionTree) -> float:
    return max_common_subtree_size(tree1, tree2) / max(total_nodes(tree1), total_nodes(tree2))


def expression_to_prefix_str(expr: Expression) -> str:
    if isinstance(expr, Constant):
        return str(expr._value)
    if isinstance(expr, Feature):
        return str(expr)
    if isinstance(expr, UnaryOperator):
        return f"{type(expr).__name__} {expression_to_prefix_str(expr._operand)}"
    if isinstance(expr, BinaryOperator):
        return f"{type(expr).__name__} {expression_to_prefix_str(expr._lhs)} {expression_to_prefix_str(expr._rhs)}"
    if isinstance(expr, RollingOperator):
        return f"{type(expr).__name__} {expression_to_prefix_str(expr._operand)} {expr._delta_time}"
    if isinstance(expr, PairRollingOperator):
        return f"{type(expr).__name__} {expression_to_prefix_str(expr._lhs)} {expression_to_prefix_str(expr._rhs)} {expr._delta_time}"
    return str(expr)
