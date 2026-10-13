"""Native widget smoke tests, kept separate from headless CI math tests."""
import unittest
import numpy as np
from panel import Panel


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.app = Panel()
        self.app.withdraw()

    def tearDown(self):
        self.app.close()

    def test_frozen_step(self):
        initial = self.app.q.copy()
        self.app.step()
        self.assertEqual(self.app.env.steps, 1)
        np.testing.assert_array_equal(initial, self.app.q)
        self.assertIn("TD目标", self.app.detail.get("1.0", "end"))

    def test_learning_step(self):
        self.app.mode.set("未训练 Q")
        self.app.load_policy()
        before = self.app.q.copy()
        self.app.step(learn=True)
        self.assertEqual(np.count_nonzero(self.app.q-before), 1)

    def test_reset_and_filter(self):
        self.app.filtered.set(False)
        self.app.start.set("0")
        self.app.reset()
        self.assertFalse(self.app.env.filtered)
        self.assertEqual(self.app.env.state, 0)


if __name__ == "__main__":
    unittest.main()
