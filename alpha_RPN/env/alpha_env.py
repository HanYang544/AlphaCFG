import random
from data.expression import *
from data.tokens import *
from typing import Any, Dict, List, Optional, Tuple
from env.tree import ExpressionBuilder, InvalidExpressionException

# ----------------------------------------------------------------------
# Map operator strings to actual operator classes
# ----------------------------------------------------------------------
OPERATOR_MAP: Dict[str, Any] = {
    'Log': Log,
    'Abs': Abs,
    'Sign': Sign,
    'CSRank': CSRank,
    'Add': Add,
    'Sub': Sub,
    'Mul': Mul,
    'Div': Div,
    'Greater': Greater,
    'Less': Less,
    'Pow': Pow,
    'Ref': Ref,
    'Skew': Skew,
    'Kurt': Kurt,
    'Mean': Mean,
    'Sum': Sum,
    'Std': Std,
    'Var': Var,
    'Max': Max,
    'Min': Min,
    'Med': Med,
    'Rank': Rank,
    'Mad': Mad,
    'Delta': Delta,
    'WMA': WMA,
    'EMA': EMA,
    'Cov': Cov,
    'Corr': Corr
}

# ----------------------------------------------------------------------
# AlphaEnv environment
# ----------------------------------------------------------------------
class AlphaEnv:
    def __init__(self, seed=None):
        """
        Maintains:
          1) ExpressionBuilder instance (self._builder) for managing the RPN stack.
          2) RPN token sequence (self.state).
          3) Definitions for various operators, features, constants, and DeltaTime numbers.
        """
        # Definitions for various actions
        self.unary_operators = ['Log', 'Abs', 'Sign', 'CSRank'] 
        self.binary_operators = ['Add', 'Sub', 'Mul', 'Div', 'Greater', 'Less', 'Pow'] 
        self.rolling_operators = ['Ref', 'Skew', 'Kurt', 'Mean', 'Sum', 'Std', 'Var',
                                  'Max', 'Min', 'Med', 'Rank', 'Mad', 'Delta', 'WMA','EMA']
        self.paired_rolling_operators = ['Cov', 'Corr']
        self.features = ['high', 'low', 'volume', 'open', 'close', 'vwap']
        self.constants = [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]
        self.num_range = [20, 30, 40]

        # ExpressionBuilder instance
        self._builder = ExpressionBuilder()
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # Initial state: add a special BEG_TOKEN to mark the start
        self.state: List[Token] = [BEG_TOKEN]

        # When user stops, the final constructed Expression will be stored here
        self.expression: Optional[Expression] = None

        # _done indicates whether the process has stopped
        self._done = False

    def reset(self):
        """Reset the environment state"""
        self.state = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        self._done = False
        self.expression = None
        return self.state_description()

    def state_description(self):
        """Return the current RPN token sequence as a string"""
        return " ".join(str(tok) for tok in self.state)

    def is_terminal_state(self):
        """Consider it a terminal state when the user executes the stop action"""
        return self._done

    def get_possible_actions(self):
        """
        Return a list of valid actions (category, value) for the current ExpressionBuilder state.
        When a valid expression is constructed, add ("stop", None) as an option.
        """
        if self._done:
            return []

        actions = []

        # 1) Feature tokens
        for f_str in self.features:
            ft = FeatureToken(FeatureType[f_str.upper()])
            if self._builder.validate(ft):
                actions.append(("feature", f_str))

        # 2) Constant tokens
        for c in self.constants:
            ct = ConstantToken(c)
            if self._builder.validate(ct):
                actions.append(("constant", c))

        # 3) DeltaTime tokens (using num as the category here)
        for n in self.num_range:
            dt = DeltaTimeToken(n)
            if self._builder.validate(dt):
                actions.append(("num", n))

        # 4) Unary operators
        for op_str in self.unary_operators:
            op_cls = OPERATOR_MAP[op_str]
            op_token = OperatorToken(op_cls)
            if self._builder.validate(op_token):
                actions.append(("unaryop", op_str))

        # 5) Binary operators
        for op_str in self.binary_operators:
            op_cls = OPERATOR_MAP[op_str]
            op_token = OperatorToken(op_cls)
            if self._builder.validate(op_token):
                actions.append(("binaryop", op_str))

        # 6) Rolling operators
        for op_str in self.rolling_operators:
            op_cls = OPERATOR_MAP[op_str]
            op_token = OperatorToken(op_cls)
            if self._builder.validate(op_token):
                actions.append(("rollingop", op_str))

        # 7) Paired rolling operators
        for op_str in self.paired_rolling_operators:
            op_cls = OPERATOR_MAP[op_str]
            op_token = OperatorToken(op_cls)
            if self._builder.validate(op_token):
                actions.append(("pairedrollingop", op_str))

        # 8) If ExpressionBuilder constructs a valid expression, we can stop
        if self._builder.is_valid():
            actions.append(("stop", None))

        return actions

    def step(self, action: Tuple[str, Any]):
        """
        Execute one step (category, value):
         - Construct the corresponding Token
         - Call builder.validate(...) and add_token(...)
         - Update the state
         - If stop, set self.expression = builder.get_tree() and add SEP_TOKEN to the sequence
        """
        if self._done:
            return self.state_description(), 0.0, True, {}

        category, value = action
        token: Optional[Token] = None

        if category == 'feature':
            feature_type = FeatureType[value.upper()]
            token = FeatureToken(feature_type)
        elif category == 'constant':
            token = ConstantToken(value)
        elif category == 'num':
            token = DeltaTimeToken(value)
        elif category in ('unaryop', 'binaryop', 'rollingop', 'pairedrollingop'):
            op_cls = OPERATOR_MAP[value]
            token = OperatorToken(op_cls)
        elif category == 'stop':
            # End the expression: automatically add SEP_TOKEN and get the final Expression
            self.state.append(SEP_TOKEN)
            self._done = True
            try:
                self.expression = self._builder.get_tree()
            except InvalidExpressionException as ex:
                # If an exception is thrown here, it means the expression is incomplete or has an error
                raise ValueError(f"Cannot generate final Expression on stop: {ex}")
            return self.state_description(), 0.0, True, {}
        else:
            raise ValueError(f"Unknown action: {action}")

        # Validate & Add token to builder
        try:
            if not self._builder.validate(token):
                raise InvalidExpressionException("This token is not allowed in the current state.")
            self._builder.add_token(token)
        except InvalidExpressionException as ex:
            raise ValueError(f"Invalid token for action {action}, error message: {ex}")

        # Add to RPN token sequence
        self.state.append(token)

        reward = 0.0  # Could be adjusted as needed
        info = {}
        return self.state_description(), reward, False, info

    def apply_action(self, action_category, action_value):
        """Convenience method: env.apply_action('binaryop', 'Add')"""
        return self.step((action_category, action_value))

if __name__ == '__main__':
    env = AlphaEnv(seed=42)

    print("Initial State:", env.state_description())
    print("Initial Valid Actions:", env.get_possible_actions())
    # At this point, env.expression should still be None
    print("env.expression =", env.expression)

    # 1. Add Feature token "low"
    s, r, d, i = env.step(('feature', 'low'))
    print("\n[Step 1] State:", s)
    print("Valid Actions:", env.get_possible_actions())
    print("env.expression =", env.expression)

    # 2. Add Feature token "high"
    s, r, d, i = env.step(('feature', 'high'))
    print("\n[Step 2] State:", s)
    print("Valid Actions:", env.get_possible_actions())
    print("env.expression =", env.expression)

    # 3. Add Binary operator "Add" -> Add($low, $high)
    s, r, d, i = env.step(('binaryop', 'Add'))
    print("\n[Step 3] State:", s)
    print("Valid Actions:", env.get_possible_actions())
    print("env.expression =", env.expression)

    # 4. Stop
    s, r, d, i = env.step(('stop', None))
    print("\n[Step 4] State:", s)
    print("Final Expression Construction Complete, done =", d)
    # At this point, env.expression should be an Expression like Add($low, $high)
    print("env.expression =", env.expression)
    # You can also view the expression string like this
    print("str(env.expression) =", str(env.expression))
