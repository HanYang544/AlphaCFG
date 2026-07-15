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


class CFGSemMethods(MCTSMethods):
    """MCTS adapter for CFG-Sem semantic CFG string states."""

    @classmethod
    def initial_state(cls):
        return AlphaEnv()

    @classmethod
    def evaluation_item(cls, node):
        return EvaluationItem(state=node.state.state_description())

    @classmethod
    def evaluate_batch(cls, items, policy_net, value_net):
        values = [
            evaluate_expression_value(item.state, value_net=value_net) for item in items
        ]
        probas = [
            evaluate_policy_network(item.state, policy_net=policy_net) for item in items
        ]
        return probas, values


Node, MCTS = build_mcts_classes(CFGSemMethods)
