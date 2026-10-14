import unittest
import importlib
from pathlib import Path
import numpy as np
from panda_residual_env import Config, PandaResidual
from panda_residual import episode


class PandaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=PandaResidual()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.reset(1000000)

    def test_frozen_prefix_and_observation(self):
        self.assertTrue(self.env.prefix_contacts.all())
        self.assertGreater(self.env.prefix_object[2],.38)
        self.assertEqual(self.env.observe().shape,(27,))
        self.assertTrue(np.isfinite(self.env.observe()).all())

    def test_full_state_reset(self):
        before=self.env.d.qpos.copy()
        expected=self.env.observe().copy()
        self.env.step([1,1]);actual,_=self.env.reset(1000000)
        np.testing.assert_array_equal(before,self.env.d.qpos)
        np.testing.assert_array_equal(expected,actual)

    def test_action_units_filter_and_physics(self):
        tick=self.env.d.time
        _,_,_,_,info=self.env.step([1,-1])
        np.testing.assert_allclose(info["requested_offset_m"],[.08,-.08])
        np.testing.assert_allclose(info["applied_offset_m"],[.008,-.008])
        self.assertTrue(info["filter_intervention"])
        self.assertAlmostEqual(self.env.d.time-tick,.04)
        self.assertEqual(info["gripper_command"],0)

    def test_invalid_action_and_config(self):
        for action in ([0], [np.nan,0], [0,np.inf]):
            with self.assertRaises(ValueError):self.env.step(action)
        with self.assertRaises(ValueError):PandaResidual(Config(residual_m=.2))

    def test_analytical_and_zero_matched_task(self):
        analytical,_,_=episode(self.env,"analytical",1000000)
        zero,_,_=episode(self.env,"zero",1000000)
        self.assertTrue(analytical["success"])
        self.assertFalse(zero["success"])
        self.assertLess(analytical["final_error_m"],.01)
        self.assertGreater(zero["final_error_m"],.035)
        self.assertEqual(analytical["steps"],125)
        self.assertTrue(analytical["criteria"]["released"])

    def test_end_is_task_terminal_and_needs_reset(self):
        for _ in range(125):o,r,term,trunc,info=self.env.step([0,0])
        self.assertTrue(term)
        self.assertFalse(trunc)
        with self.assertRaises(RuntimeError):self.env.step([0,0])
        self.env.reset(3)
        self.assertFalse(self.env.ended)

    def test_ood_changes_goal_only(self):
        q=self.env.d.qpos.copy()
        self.env.reset(2000000,"ood")
        radius=np.linalg.norm(self.env.goal[:2]-[.5,-.19])
        self.assertGreaterEqual(radius,.075)
        self.assertLessEqual(radius,.095)
        np.testing.assert_array_equal(q,self.env.d.qpos)

    def test_bypass_has_explicit_requested_execution(self):
        other=PandaResidual(Config(filter_enabled=False))
        try:
            _,_,_,_,info=other.step([1,-1])
            np.testing.assert_allclose(info["applied_offset_m"],[.08,-.08])
            self.assertFalse(info["filter_intervention"])
        finally:other.close()

    def test_release_rejects_single_finger_contact(self):
        result,_,_=episode(self.env,"analytical",1000000)
        self.assertTrue(result["success"])
        self.env.rows[-1]["finger_contacts"]=[True,False]
        self.env.rows[-1]["bilateral_contact"]=False
        self.assertFalse(self.env.assess()["criteria"]["released"])

    def test_import_preserves_other_project_panel(self):
        module=importlib.import_module("panel")
        self.assertEqual(Path(module.__file__).resolve().parent,Path(__file__).resolve().parent)


if __name__=="__main__":unittest.main()
