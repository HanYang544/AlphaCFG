import unittest
from types import SimpleNamespace

from alphacfg.data.expression import Feature
from alphacfg.data.stock_data import FeatureType
from alphacfg.unified_runner import ReplayBuffer, process_rewards


class FixedCalculator:
    def __init__(self, ic):
        self.ic = ic

    def calc_single_IC_ret(self, expression):
        return self.ic


def terminal_game(expression):
    return [{
        "state_input": [1, 2],
        "aux": ["open"],
        "raw_pi": {("feature", "open"): 1.0},
        "terminal": True,
        "final_state": SimpleNamespace(expression=expression),
    }]


class SingleRewardTests(unittest.TestCase):
    def test_negative_ic_uses_absolute_value_without_structure_penalty(self):
        expression = Feature(FeatureType.OPEN)
        replay = ReplayBuffer(10)

        summary = process_rewards(
            "rpn",
            [terminal_game(expression)],
            replay,
            "single",
            FixedCalculator(-0.07),
            generated_exprs=set(),
            pool=None,
        )

        self.assertAlmostEqual(replay.buffer[0][2], 0.07)
        self.assertAlmostEqual(summary["records"][0]["ic"], -0.07)
        self.assertAlmostEqual(summary["records"][0]["reward"], 0.07)
        self.assertEqual(summary["records"][0]["max_tree_similarity"], 0.0)

    def test_duplicate_expression_uses_zero_reward(self):
        expression = Feature(FeatureType.OPEN)
        replay = ReplayBuffer(10)

        process_rewards(
            "rpn",
            [terminal_game(expression)],
            replay,
            "single",
            FixedCalculator(-0.07),
            generated_exprs={str(expression)},
            pool=None,
        )

        self.assertEqual(replay.buffer[0][2], 0.0)


if __name__ == "__main__":
    unittest.main()
