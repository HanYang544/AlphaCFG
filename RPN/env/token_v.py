import torch
from typing import List
from env.alpha_env import *

def convert_state_tokens_to_ids(state_tokens: List[Token]) -> List[int]:
    """
    Converts the list of environment tokens (state_tokens) into a list of integers.
    Each token is first converted to a string using str(token), and then mapped to an integer using token_map.
    If the token is not found in the map, -1 is returned to indicate an unknown token.
    """
    int_list = []
    for t in state_tokens:
        token_str = str(t)  # e.g., "BEG", "$low", "Add", "-0.1", "20", "SEP", ...
        if token_str in ENV_TOKEN_MAP:
            int_list.append(ENV_TOKEN_MAP[token_str])
        else:
            int_list.append(-1)  # Unknown token
    return int_list


# Token mapping to integers
ENV_TOKEN_MAP = {
    # --------- Special Tokens ---------
    "BEG": 0,   # Beginning of the expression
    "SEP": 1,   # Separator between tokens

    # --------- Feature Tokens ---------
    "$high": 2,    # Feature: high
    "$low": 3,     # Feature: low
    "$volume": 4,  # Feature: volume
    "$open": 5,    # Feature: open
    "$close": 6,   # Feature: close
    "$vwap": 7,    # Feature: vwap

    # --------- Constant Tokens ---------
    "-0.1": 8,  # Constant value: -0.1
    "-0.05": 9, # Constant value: -0.05
    "-0.01": 10, # Constant value: -0.01
    "0.01": 11,  # Constant value: 0.01
    "0.05": 12,  # Constant value: 0.05
    "0.1": 13,   # Constant value: 0.1

    # --------- DeltaTime Tokens ---------
    "20": 14,  # DeltaTime: 20
    "30": 15,  # DeltaTime: 30
    "40": 16,  # DeltaTime: 40

    # --------- Unary Operators ---------
    "Log": 17,    # Unary operator: Log
    "Abs": 18,    # Unary operator: Abs
    "Sign": 19,   # Unary operator: Sign
    "CSRank": 20, # Unary operator: CSRank

    # --------- Binary Operators ---------
    "Add": 21,    # Binary operator: Add
    "Sub": 22,    # Binary operator: Sub
    "Mul": 23,    # Binary operator: Mul
    "Div": 24,    # Binary operator: Div
    "Greater": 25,# Binary operator: Greater
    "Less": 26,   # Binary operator: Less
    "Pow": 27,    # Binary operator: Pow

    # --------- Rolling Operators ---------
    "Ref": 28,   # Rolling operator: Ref
    "Skew": 29,  # Rolling operator: Skew
    "Kurt": 30,  # Rolling operator: Kurt
    "Mean": 31,  # Rolling operator: Mean
    "Sum": 32,   # Rolling operator: Sum
    "Std": 33,   # Rolling operator: Std
    "Var": 34,   # Rolling operator: Var
    "Max": 35,   # Rolling operator: Max
    "Min": 36,   # Rolling operator: Min
    "Med": 37,   # Rolling operator: Med
    "Rank": 38,  # Rolling operator: Rank
    "Mad": 39,   # Rolling operator: Mad
    "Delta": 40, # Rolling operator: Delta
    "WMA": 41,   # Rolling operator: WMA
    "EMA": 42,   # Rolling operator: EMA

    # --------- Paired Rolling Operators ---------
    "Cov": 43,   # Paired rolling operator: Cov
    "Corr": 44,  # Paired rolling operator: Corr
}
