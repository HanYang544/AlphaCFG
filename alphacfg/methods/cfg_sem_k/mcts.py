import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_VARIANT_ROOT = Path(__file__).resolve().parent
if str(_VARIANT_ROOT) not in sys.path:
    sys.path.insert(0, str(_VARIANT_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.alpha_env import AlphaEnv
from networks.policy_ import evaluate_policy_network
from networks.value_ import evaluate_expression_value

from alphacfg.mcts_core import EvaluationItem, MCTSMethods, build_mcts_classes


class CFGSemKMethods(MCTSMethods):
    """CFG-Sem-k 的 MCTS 方法：CFG 文法状态 + 剩余展开步数约束。"""

    max_expression_length = 10
    default_continue_out_game = 3

    @classmethod
    def initial_state(cls):
        return AlphaEnv()

    @classmethod
    def child_continue_out_game(cls, state, parent_continue_out_game):
        depth = getattr(state, "expansion_depth", 0)
        if depth <= cls.max_expression_length - 3:
            return 3
        return max(0, cls.max_expression_length - depth)

    @classmethod
    def evaluation_item(cls, node):
        return EvaluationItem(
            state=node.state.state_description(),
            continue_out_game=node.continue_out_game,
        )

    @classmethod
    def evaluate_batch(cls, items, policy_net, value_net):
        values = [
            evaluate_expression_value(item.state, value_net=value_net) for item in items
        ]
        probas = [
            evaluate_policy_network(
                item.state,
                policy_net=policy_net,
                continue_out_game=item.continue_out_game,
            )
            for item in items
        ]
        return probas, values


Node, MCTS = build_mcts_classes(CFGSemKMethods)
