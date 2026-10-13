import unittest
from continuous_panel import ContinuousPanel


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.app = ContinuousPanel()
        self.app.withdraw()

    def tearDown(self):
        self.app.close()

    def test_load_baseline_gif_and_metrics(self):
        self.assertGreater(len(self.app.frames), 0)
        self.assertIn("successes", self.app.metrics.get("1.0", "end"))

    def test_missing_replay_explicit(self):
        self.app.policy.set("zero")
        self.app.load_replay()
        self.assertEqual(len(self.app.frames), 0)
        self.assertIn("实时3D", self.app.picture.cget("text"))

    def test_replay_step_and_pause(self):
        before = self.app.frame
        self.app.toggle()
        self.app.tick()
        self.assertEqual(self.app.frame, (before+1)%len(self.app.frames))
        self.app.stop()
        self.assertFalse(self.app.playing)


if __name__ == "__main__":
    unittest.main()
