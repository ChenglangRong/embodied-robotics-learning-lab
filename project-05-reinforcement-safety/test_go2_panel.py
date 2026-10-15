import unittest
from go2_panel import Go2Panel


class Go2PanelTests(unittest.TestCase):
    def setUp(self):self.app=Go2Panel();self.app.withdraw()
    def tearDown(self):self.app.close()

    def test_metrics_and_plot(self):
        self.assertGreater(len(self.app.frames),0)
        self.assertIn("原始RMSE",self.app.metrics.get("1.0","end"))
        self.assertTrue(hasattr(self.app,"plot_image"))

    def test_play_pause(self):
        self.app.toggle();self.app.tick();self.assertEqual(self.app.frame,1)
        self.app.stop();self.assertFalse(self.app.playing)

    def test_missing_replay_and_zero_gains(self):
        self.app.mode.set("wrong_sign");self.app.replay()
        self.assertEqual(len(self.app.frames),0)
        self.app.kv.set("0");self.app.kw.set("0");self.assertIsNotNone(self.app.parameters())


if __name__=="__main__":unittest.main()
