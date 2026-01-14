import torch
from data.stock_data import *
from data.calculator__ import *
from data.expression import *
import multiprocessing
from joblib import Parallel, delayed


# Set the multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)


def parse_expression_from_string(expression_str: str) -> Expression:
    """
    Parse a prefix notation expression string and return an Expression object.

    Parameters:
    - expression_str: str, a prefix notation expression, e.g., "Add $high $low"

    Returns:
    - Expression object
    """
    tokens = expression_str.split()
    tokens = tokens[::-1]  # Reverse the list for convenient pop() from the end

    def parse() -> Expression:
        if not tokens:
            raise ValueError("Invalid expression: missing tokens.")
        token = tokens.pop()

        # Binary operators (convert operators with suffixes to base forms)
        if token in [
            'Add', 
            'Sub',
            'Mul',  
            'Div',
            'Pow',
            'Greater', 
            'Less'
        ]:
            lhs = parse()  # Left operand
            rhs = parse()  # Right operand
            operator_map = {
                'Add': Add,
                'Sub': Sub,
                'Mul': Mul,
                'Div': Div,
                'Greater': Greater,
                'Less': Less,
                'Pow': Pow
            }
            return operator_map[token](lhs, rhs)  # Pass both left and right operands

        # Unary operators
        elif token in ['Abs', 'Sign', 'Log', 'CSRank']:
            operand = parse()
            operator_map = {
                'Abs': Abs,
                'Sign': Sign,
                'Log': Log,
                'CSRank': CSRank
            }
            return operator_map[token](operand)

        # Rolling operators
        elif token in ['Ref', 'Mean', 'Sum', 'Std', 'Var', 'Skew', 'Kurt', 'Max', 'Min', 
                       'Med', 'Mad', 'Rank', 'Delta', 'WMA', 'EMA']:
            operand = parse()
            delta_time_token = tokens.pop()
            try:
                delta_time = int(delta_time_token)
            except ValueError:
                raise ValueError(f"Invalid delta_time value: {delta_time_token}")
            operator_map = {
                'Ref': Ref,
                'Mean': Mean,
                'Sum': Sum,
                'Std': Std,
                'Var': Var,
                'Skew': Skew,
                'Kurt': Kurt,
                'Max': Max,
                'Min': Min,
                'Med': Med,
                'Mad': Mad,
                'Rank': Rank,
                'Delta': Delta,
                'WMA': WMA,
                'EMA': EMA
            }
            return operator_map[token](operand, delta_time)

        # Paired rolling operators
        elif token in ['Cov', 'Corr']:
            lhs = parse()
            rhs = parse()
            delta_time_token = tokens.pop()
            try:
                delta_time = int(delta_time_token)
            except ValueError:
                raise ValueError(f"Invalid delta_time value: {delta_time_token}")
            operator_map = {
                'Cov': Cov,
                'Corr': Corr
            }
            return operator_map[token](lhs, rhs, delta_time)

        # Feature variables
        elif token.startswith('$'):
            feature_name = token[1:].upper()  # Assume FeatureType names are uppercase
            try:
                feature = FeatureType[feature_name]
            except KeyError:
                raise ValueError(f"Unknown feature variable: {feature_name}")
            return Feature(feature)

        # Constants
        else:
            try:
                value = float(token)
                return Constant(value)
            except ValueError:
                raise ValueError(f"Unknown token or invalid constant: {token}")

    expression = parse()
    if tokens:
        raise ValueError("Invalid expression: extra tokens remaining.")
    return expression