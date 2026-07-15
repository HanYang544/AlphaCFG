import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_VARIANT_ROOT = Path(__file__).resolve().parent
if str(_VARIANT_ROOT) not in sys.path:
    sys.path.insert(0, str(_VARIANT_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.alpha_env import AlphaEnv
from env.token_v import convert_state_tokens_to_ids
from networks.policy_ import evaluate_policy_network
from networks.value_ import evaluate_expression_value

from alphacfg.mcts_core import EvaluationItem, MCTSMethods, build_mcts_classes


class RPNMethods(MCTSMethods):
    """RPN 的 MCTS 方法：栈式 ExpressionBuilder 状态转移。"""

    @classmethod
    def initial_state(cls):
        return AlphaEnv()

    @classmethod
    def apply_action(cls, state, category, action):
        state.apply_action(category, action)

    @classmethod
    def evaluation_item(cls, node):
        return EvaluationItem(state=node.state)

    @classmethod
    def evaluate_batch(cls, items, policy_net, value_net):
        values = [
            evaluate_expression_value(
                expression_tokens=convert_state_tokens_to_ids(item.state.state),
                value_net=value_net,
            )
            for item in items
        ]
        probas = [
            evaluate_policy_network(
                expression_tokens=convert_state_tokens_to_ids(item.state.state),
                possible_actions=item.state.get_possible_actions(),
                policy_net=policy_net,
            )
            for item in items
        ]
        return probas, values


Node, MCTS = build_mcts_classes(RPNMethods)
