import unittest
from panda_panel import PandaPanel


class PandaPanelTests(unittest.TestCase):
    def setUp(self):
        self.app=PandaPanel();self.app.withdraw()

    def tearDown(self):self.app.close()

    def test_metrics_and_fixed_replay(self):
        self.assertGreater(len(self.app.frames),0)
        self.assertIn("ID8案例",self.app.metrics.get("1.0","end"))
        self.assertIn("mm",self.app.metrics.get("1.0","end"))

    def test_prefix_and_play_pause(self):
        self.app.prefix_replay();self.assertGreater(len(self.app.frames),0)
        self.app.toggle();self.app.tick();self.assertEqual(self.app.frame,1)
        self.app.stop();self.assertFalse(self.app.playing)

    def test_missing_replay_keeps_live_route(self):
        self.app.policy.set("random");self.app.replay()
        self.assertEqual(len(self.app.frames),0)
        self.assertIn("实时3D",self.app.picture.cget("text"))


if __name__=="__main__":unittest.main()
