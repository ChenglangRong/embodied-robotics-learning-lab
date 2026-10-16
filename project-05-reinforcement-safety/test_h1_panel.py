import unittest
try:
    import tkinter as tk
    from h1_recovery import CONTROLLERS,SCENARIOS
    from h1_panel import BASELINE,H1Panel,LESSONS,ROOT
    READY=True
except ImportError as error:
    READY=False;IMPORT_ERROR=error

BUTTONS=("实时3D · 当前选择","载入保存回放与指标","回放播放 / 暂停","运行新实验 · 五种子全对照",
         "打开知识与编程操作课（07-H1扰动恢复实操课.md）","打开结果与失败轨迹","停止当前运行 · 保留日志")


class H1PanelTests(unittest.TestCase):
    def setUp(self):
        if not READY:self.skipTest(f"panel dependencies unavailable: {IMPORT_ERROR}")
        try:self.app=H1Panel()
        except tk.TclError as error:self.skipTest(f"no display: {error}")
        self.app.withdraw()
    def tearDown(self):
        if getattr(self,"app",None) is not None:self.app.close()

    def test_window_and_controls_exist(self):
        self.app.update_idletasks()
        self.assertTrue(self.app.winfo_exists())
        self.assertEqual(self.app.mode.get(),"com")
        self.assertEqual(self.app.scenario.get(),"strong_front")
        for label in BUTTONS:
            self.assertIn(label,self.app.buttons)
            self.assertEqual(self.app.buttons[label].winfo_class(),"TButton")
        self.assertEqual(self.app.selectors["控制器"]["values"],CONTROLLERS)
        self.assertEqual(self.app.selectors["场景"]["values"],SCENARIOS)
        self.assertEqual(len(self.app.selectors["实时重置种子"]["values"]),5)
        self.assertEqual(self.app.logdir,ROOT/"outputs/h1-student")

    def test_live_and_experiment_arguments(self):
        self.assertEqual(self.app.live_args(),["--demo","--controller","com","--scenario","strong_front","--seed","3000000"])
        self.app.mode.set("wrong_sign");self.app.scenario.set("push_left");self.app.seed.set("3000004")
        self.assertEqual(self.app.live_args(),["--demo","--controller","wrong_sign","--scenario","push_left","--seed","3000004"])
        folder=self.app.logdir/"unit-test"
        self.assertEqual(self.app.experiment_args(folder),["--output",str(folder)])

    def test_play_pause_and_stop(self):
        self.app.toggle();self.assertTrue(self.app.playing)
        if self.app.frames:
            self.app.tick();self.assertEqual(self.app.frame,1)
            self.app.tick();self.assertEqual(self.app.frame,2%len(self.app.frames))
        self.app.stop();self.assertFalse(self.app.playing)

    def test_replay_frames_and_frozen_metrics(self):
        if not self.app.frames:self.skipTest(f"frozen demos not generated yet under {BASELINE}")
        self.assertIn("冻结结果",self.app.metrics.get("1.0","end"))
        self.assertIn(self.app.mode.get(),self.app.metrics.get("1.0","end"))

    def test_missing_replay_folder_is_reported(self):
        self.app.folder=ROOT/"outputs"/"h1-panel-test-missing-folder"
        self.app.replay()
        self.assertEqual(len(self.app.frames),0)
        self.assertIn("未生成",self.app.picture.cget("text"))
        self.assertIn("尚未生成",self.app.metrics.get("1.0","end"))
        self.assertIsNone(self.app.results_path())

    def test_lesson_and_results_paths(self):
        path=self.app.lesson_path()
        if path is not None:
            self.assertTrue(path.exists())
            self.assertEqual(path,next(ROOT/name for name in LESSONS if (ROOT/name).exists()))
        self.assertEqual(self.app.results_path(),BASELINE if BASELINE.exists() else None)


if __name__=="__main__":unittest.main()
