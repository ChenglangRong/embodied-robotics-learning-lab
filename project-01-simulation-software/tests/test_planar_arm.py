import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robot_lab.planar_arm import Config, TwoLinkReachEnv


class TestPlanarArm(unittest.TestCase):
    def test_forward_kinematics(self):
        env = TwoLinkReachEnv()
        q = np.array([np.pi / 2, np.pi / 2])
        np.testing.assert_allclose(env.position(q), [-0.4, 0.6], atol=1e-12)

    def test_jacobian_matches_central_difference(self):
        env = TwoLinkReachEnv()
        q = np.array([0.35, 0.55])
        eps = 1e-6
        numeric = np.column_stack([
            (env.position(q + np.eye(2)[i] * eps) - env.position(q - np.eye(2)[i] * eps)) / (2 * eps)
            for i in range(2)
        ])
        np.testing.assert_allclose(env.jacobian(q), numeric, atol=1e-9)

    def test_reset_is_deterministic(self):
        a = TwoLinkReachEnv().reset(seed=11)
        b = TwoLinkReachEnv().reset(seed=11)
        np.testing.assert_array_equal(a["q"], b["q"])

    def test_action_limit_is_reported(self):
        env = TwoLinkReachEnv(Config(velocity_limit=0.1))
        env.reset(seed=0)
        _, _, _, info = env.step(np.array([1.0, -1.0]))
        self.assertTrue(info["action_clipped"])


if __name__ == "__main__":
    unittest.main()
