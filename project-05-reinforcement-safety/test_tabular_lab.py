import unittest
import numpy as np
from tabular_lab import ArmMDP, Config, position, rollout, td_target, train, value_iteration


class Tests(unittest.TestCase):
    def test_kinematics(self):
        np.testing.assert_allclose(position([0, 0]), [1, 0])
        np.testing.assert_allclose(position([np.pi/2, 0]), [0, 1], atol=1e-15)

    def test_projection(self):
        env = ArmMDP()
        env.reset(0)
        nxt, _, terminated, _, info = env.step(1)
        self.assertEqual(nxt, 0)
        self.assertTrue(info["intervention"])
        self.assertFalse(info["violation"] or terminated)

    def test_bypass(self):
        env = ArmMDP(filtered=False)
        env.reset(0)
        nxt, _, terminated, _, info = env.step(1)
        self.assertTrue(info["violation"] and terminated)
        self.assertEqual(nxt, env.failure_state)
        self.assertLess(info["q"][0], -1.5)
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_all_projection_states(self):
        env = ArmMDP()
        for start in env.starts:
            for action in range(5):
                env.reset(int(start))
                _, _, _, _, info = env.step(action)
                self.assertTrue(-1.5 <= info["q"][0] <= 1.5 and 0 <= info["q"][1] <= 2.4)
                self.assertFalse(info["violation"])

    def test_time_limit(self):
        env = ArmMDP(Config(max_steps=1))
        _, _, terminated, truncated, _ = env.step(4)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(td_target(1, [2, 3], terminated, 0.9), 3.7)
        self.assertEqual(td_target(1, [2, 3], True, 0.9), 1)

    def test_success_terminal(self):
        env = ArmMDP()
        pair = next((int(s), a) for s in env.starts for a in range(5) if env.terminal[env.dest[s, a]])
        env.reset(pair[0])
        _, _, terminated, _, info = env.step(pair[1])
        self.assertTrue(terminated and info["success"])

    def test_oracle(self):
        env = ArmMDP(Config(bins=9))
        q, residuals = value_iteration(env)
        self.assertLess(residuals[-1], 1e-10)
        target = env.reward+env.cfg.gamma*q.max(axis=1)[env.dest]*(~env.terminal[env.dest])
        np.testing.assert_allclose(q[~env.terminal], target[~env.terminal], atol=1e-9)
        self.assertTrue(all(rollout(env.cfg, q, s)[1]["success"] for s in env.starts))

    def test_repeatable_training(self):
        cfg = Config(bins=7, episodes=20, max_steps=12)
        a, av, al = train(cfg, 4)
        b, bv, bl = train(cfg, 4)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(av, bv)
        self.assertEqual(al, bl)
        self.assertEqual(av.sum(), al[-1]["environment_steps"])

    def test_input_validation(self):
        for kwargs in ({"gamma": 1}, {"alpha": 0}, {"episodes": 0}, {"goal_shift": float("nan")}):
            with self.assertRaises(ValueError):
                Config(**kwargs)
        env = ArmMDP()
        for action in (-1, 5, 0.5):
            with self.assertRaises(ValueError):
                env.step(action)
        with self.assertRaises(ValueError):
            env.reset(env.failure_state)


if __name__ == "__main__":
    unittest.main()
