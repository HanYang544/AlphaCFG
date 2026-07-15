import unittest
from types import SimpleNamespace

from alphacfg.data.expression import Feature
from alphacfg.data.stock_data import FeatureType
from alphacfg.unified_runner import (
    ReplayBuffer,
    _accepted_pool_reward,
    process_rewards,
)


class FakePool:
    def __init__(self, old_ic=0.04, new_ic=0.05, similarity=0.25):
        self.size = 1
        self.exprs = [Feature(FeatureType.CLOSE), None]
        self.single_ics = __import__("numpy").zeros(2)
        self._weights = __import__("numpy").zeros(2)
        self._mutual_ics = __import__("numpy").identity(2)
        self._extra_info = [None, None]
        self.best_obj = old_ic
        self.best_ic_ret = old_ic
        self.update_history = []
        self._failure_cache = set()
        self.eval_cnt = 0
        self._ic = old_ic
        self._new_ic = new_ic
        self._similarity = similarity
        self.inserted = False

    def evaluate_ensemble(self):
        return self._ic

    def compute_avg_similarity_with_pool(self, expression):
        if self.inserted:
            raise AssertionError("similarity must be measured before insertion")
        if isinstance(self._similarity, dict):
            return self._similarity[expression]
        return self._similarity

    def try_new_expr(self, expression):
        self.inserted = True
        self.eval_cnt += 1
        self._ic = self._new_ic


class PoolRewardTests(unittest.TestCase):
    def test_reward_formula(self):
        self.assertAlmostEqual(_accepted_pool_reward(0.08, 0.25), 0.06)
        self.assertEqual(_accepted_pool_reward(-0.08, 0.25), 0.0)
        self.assertEqual(_accepted_pool_reward(0.08, 1.0), 0.0)

    def test_process_rewards_measures_similarity_before_insert(self):
        expression = Feature(FeatureType.OPEN)
        final_state = SimpleNamespace(expression=expression)
        game = [{
            "state_input": [1, 2],
            "aux": ["open"],
            "raw_pi": {("feature", "open"): 1.0},
            "terminal": True,
            "final_state": final_state,
        }]
        replay = ReplayBuffer(10)
        pool = FakePool(old_ic=0.04, new_ic=0.08, similarity=0.25)

        summary = process_rewards(
            "rpn",
            [game],
            replay,
            "pool",
            calculator=None,
            generated_exprs=set(),
            pool=pool,
        )

        self.assertEqual(len(replay), 1)
        self.assertAlmostEqual(replay.buffer[0][2], 0.06)
        self.assertAlmostEqual(summary["average_reward"], 0.06)
        self.assertAlmostEqual(summary["average_tree_similarity"], 0.25)

    def test_each_trajectory_state_gets_its_own_similarity_penalty(self):
        final_state = SimpleNamespace(state_description=lambda: "Abs $open")
        game = [
            {
                "state_input": state,
                "aux": 3,
                "raw_pi": {("unaryop", "Abs"): 1.0},
                "terminal": True,
                "final_state": final_state,
            }
            for state in ("Q", "Abs Q", "Abs $open")
        ]
        replay = ReplayBuffer(10)
        pool = FakePool(
            old_ic=0.04,
            new_ic=0.08,
            similarity={"Q": 0.0, "Abs Q": 0.25, "Abs $open": 0.5},
        )

        summary = process_rewards(
            "cfg-sem-k",
            [game],
            replay,
            "pool",
            calculator=None,
            generated_exprs=set(),
            pool=pool,
        )

        rewards = [sample[2] for sample in replay.buffer]
        self.assertEqual(len(rewards), 3)
        self.assertAlmostEqual(rewards[0], 0.08)
        self.assertAlmostEqual(rewards[1], 0.06)
        self.assertAlmostEqual(rewards[2], 0.04)
        self.assertAlmostEqual(summary["average_reward"], 0.06)
        self.assertAlmostEqual(summary["average_tree_similarity"], 0.25)


if __name__ == "__main__":
    unittest.main()
