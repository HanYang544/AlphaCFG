import unittest

from alphacfg.methods.cfg_sem_k.env.alpha_env import AlphaEnv


class MaskStateTests(unittest.TestCase):
    def test_accepts_terminals_before_non_terminals(self):
        env = AlphaEnv(seed=0)

        state = env.set_state_from_expression("Mul1 Rank Q num Q")

        self.assertEqual(state, ["Mul1 Rank Q num Q"])

    def test_rejects_terminal_after_non_terminal(self):
        env = AlphaEnv(seed=0)

        with self.assertRaisesRegex(ValueError, "all symbols after"):
            env.set_state_from_expression("Mul1 Q Rank Q num")


if __name__ == "__main__":
    unittest.main()
