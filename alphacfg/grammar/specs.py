"""Explicit grammar metadata for the AlphaCFG method families.

This module does not replace the method-specific environments yet.  It gives
the framework one canonical place to describe the paper-level language layer
implemented by each variant, so runners, docs, and checks do not need to infer
that information from ad-hoc environment strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class GrammarSpec:
    """Static description of one alpha generation language."""

    name: str
    language_layer: str
    state_form: str
    start_symbol: str
    supports_tree_lstm: bool
    enforces_syntax: bool
    enforces_semantics: bool
    enforces_length_bound: bool
    terminals: Mapping[str, Sequence[str]]
    nonterminals: Sequence[str]
    notes: Sequence[str]


COMMON_FEATURES = ("open", "high", "low", "close", "volume", "vwap")
COMMON_CONSTANTS = ("-0.1", "-0.05", "-0.01", "0.01", "0.05", "0.1")
COMMON_WINDOWS = ("20", "30", "40")


CFG_TERMINALS = {
    "features": COMMON_FEATURES,
    "constants": COMMON_CONSTANTS,
    "windows": COMMON_WINDOWS,
    "unary": ("Log", "Abs", "Sign", "CSRank"),
    "binary": (
        "Add1",
        "Add2",
        "Sub1",
        "Sub2",
        "Sub3",
        "Mul1",
        "Mul2",
        "Div1",
        "Div2",
        "Div3",
        "Pow1",
        "Pow2",
        "Pow3",
        "Greater1",
        "Greater2",
        "Greater3",
        "Less1",
        "Less2",
        "Less3",
    ),
    "rolling": (
        "Ref",
        "Skew",
        "Kurt",
        "Mean",
        "Sum",
        "Std",
        "Var",
        "Max",
        "Min",
        "Med",
        "Rank",
        "Mad",
        "Delta",
        "WMA",
        "EMA",
    ),
    "paired_rolling": ("Cov", "Corr"),
}


GRAMMAR_SPECS: dict[str, GrammarSpec] = {
    "rpn": GrammarSpec(
        name="rpn",
        language_layer="RPN baseline",
        state_form="token stack / postfix sequence",
        start_symbol="empty expression stack",
        supports_tree_lstm=False,
        enforces_syntax=True,
        enforces_semantics=False,
        enforces_length_bound=True,
        terminals={
            "features": COMMON_FEATURES,
            "constants": COMMON_CONSTANTS,
            "operators": (
                "Add",
                "Sub",
                "Mul",
                "Div",
                "Greater",
                "Less",
                "Pow",
                "Ref",
                "Mean",
                "Std",
                "Corr",
            ),
        },
        nonterminals=(),
        notes=(
            "RPN is kept as a sequence baseline and cannot use TreeLSTM.",
            "Its validity is checked through stack/expression-builder state.",
        ),
    ),
    "cfg-syn": GrammarSpec(
        name="cfg-syn",
        language_layer="alpha-Syn",
        state_form="partial prefix expression tree",
        start_symbol="Q",
        supports_tree_lstm=True,
        enforces_syntax=True,
        enforces_semantics=False,
        enforces_length_bound=False,
        terminals=CFG_TERMINALS,
        nonterminals=("Q", "J", "num"),
        notes=(
            "Syntax constraints keep generated formulas parseable.",
            "Financial semantic constraints are intentionally weaker than cfg-sem.",
        ),
    ),
    "cfg-sem": GrammarSpec(
        name="cfg-sem",
        language_layer="alpha-Sem",
        state_form="partial prefix expression tree",
        start_symbol="Q",
        supports_tree_lstm=True,
        enforces_syntax=True,
        enforces_semantics=True,
        enforces_length_bound=False,
        terminals=CFG_TERMINALS,
        nonterminals=("Q", "J", "num"),
        notes=(
            "Operator variants encode simple financial semantic roles.",
            "No global k-bound is enforced by the grammar layer.",
        ),
    ),
    "cfg-sem-k": GrammarSpec(
        name="cfg-sem-k",
        language_layer="alpha-Sem-k",
        state_form="partial prefix expression tree with remaining length budget",
        start_symbol="Q",
        supports_tree_lstm=True,
        enforces_syntax=True,
        enforces_semantics=True,
        enforces_length_bound=True,
        terminals=CFG_TERMINALS,
        nonterminals=("Q", "J", "num"),
        notes=(
            "Length is tracked by production-rule increments.",
            "This is the closest implementation to the paper's main AlphaCFG setting.",
        ),
    ),
}


def get_grammar_spec(name: str) -> GrammarSpec:
    """Return grammar metadata by variant name."""

    key = name.lower()
    if key not in GRAMMAR_SPECS:
        available = ", ".join(sorted(GRAMMAR_SPECS))
        raise ValueError(f"Unknown grammar spec '{name}'. Available: {available}")
    return GRAMMAR_SPECS[key]
