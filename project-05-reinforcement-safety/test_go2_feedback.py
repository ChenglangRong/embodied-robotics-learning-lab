import importlib
from pathlib import Path
import unittest
import numpy as np
from go2_feedback import Go2,Config,command,stride_reference,world_to_body,episode


class FeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.env=Go2()
    @classmethod
    def tearDownClass(cls):cls.env.close()

    def test_world_body_velocity_rotation(self):
        r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_allclose(world_to_body(r,[0,1,0]),[1,0,0])

    def test_command_schedule_and_stop(self):
        np.testing.assert_array_equal(command("step_stop",2),[.08,0])
        np.testing.assert_array_equal(command("step_stop",4),[.16,0])
        np.testing.assert_array_equal(command("step_stop",7),[0,0])
        with self.assertRaises(ValueError):command("unknown",0)

    def test_negative_feedback_math_and_yaw_sign(self):
        cfg=Config();cmd=np.array([.12,.25]);measured=np.array([.1,.1])
        s=stride_reference("feedback",cmd,measured,cfg)
        effective=cmd+2*(cmd-measured)
        np.testing.assert_allclose(s,.45*(effective[0]-effective[1]*np.array([.095,-.095,.095,-.095])))
        self.assertGreater(s[1],s[0])

    def test_zero_gain_matches_feedforward(self):
        cfg=Config(velocity_gain=0,yaw_gain=0);cmd=np.array([.12,.25]);observed=np.array([.05,0])
        np.testing.assert_array_equal(stride_reference("feedback",cmd,observed,cfg),stride_reference("feedforward",cmd,observed,cfg))

    def test_zero_command_stand_mode(self):
        for m in ("feedforward","feedback","wrong_sign"):
            np.testing.assert_array_equal(stride_reference(m,np.zeros(2),np.array([.5,.5]),Config()),np.zeros(4))

    def test_reset_reproducibility(self):
        self.env.reset(1000000);q=self.env.d.qpos.copy()
        self.env.step("feedback","left");self.env.reset(1000000)
        np.testing.assert_array_equal(q,self.env.d.qpos)

    def test_control_period_torque_and_rate_bounds(self):
        self.env.reset(1000000);t=self.env.d.time;r=self.env.step("feedback","fast")
        self.assertAlmostEqual(self.env.d.time-t,.02)
        self.assertLessEqual(max(abs(np.array(r["executed_stride_m"]))),.002+1e-12)
        self.assertLessEqual(r["torque_peak_nm"],max(abs(self.env.m.actuator_ctrlrange.ravel()))+1e-12)
        self.assertEqual(len(r["joint_reference_rad"]),12)

    def test_cycle_average_is_recorded(self):
        self.env.reset(1000000)
        for _ in range(30):self.env.step("feedback","left")
        self.assertAlmostEqual(self.env.rows[-1]["cycle_v_m_s"],np.mean([r["v_body_m_s"] for r in self.env.rows[-25:]]))
        self.assertEqual(len(self.env.rows[-1]["foot_contacts"]),4)

    def test_feedback_straight_and_directional_turn(self):
        fast,_,_=episode(self.env,"feedback","fast",1000000)
        left,_,_=episode(self.env,"feedback","left",1000000)
        self.assertTrue(fast["criteria"]["upright"])
        self.assertLess(fast["speed_rmse_m_s"],.07)
        self.assertGreater(left["mean_yaw_rate_rad_s"],.05)
        with self.assertRaises(RuntimeError):self.env.step("feedback","left")

    def test_invalid_gains_and_no_import_pollution(self):
        with self.assertRaises(ValueError):Go2(Config(velocity_gain=np.nan))
        self.assertEqual(Path(importlib.import_module("panel").__file__).resolve().parent,Path(__file__).resolve().parent)


class ReplayToleranceTests(unittest.TestCase):
    def test_observed_cloud_roundoff_is_measured(self):
        from verify_go2 import equal_trajectory
        stats={"compared_float_values":0,"maximum_absolute_difference":0.,"largest_difference_path":None,"maximum_tolerance_fraction":0.}
        equal_trajectory({"applied_torque_nm":[0.0008412553537757717]},
                         {"applied_torque_nm":[0.0008412532144022045]},stats=stats)
        self.assertEqual(stats["compared_float_values"],1)
        self.assertAlmostEqual(stats["maximum_absolute_difference"],2.1393735671897396e-9,places=17)

    def test_meaningful_error_and_nonfinite_values_fail(self):
        from verify_go2 import equal_trajectory
        for value in (0.0009412532144022045,float("nan"),float("inf")):
            with self.assertRaises(AssertionError):equal_trajectory(value,0.0008412532144022045)

    def test_discrete_state_and_structure_stay_exact(self):
        from verify_go2 import equal_trajectory
        for actual,expected in ((True,False),(1,2),([1.],[]),({"v":0.},{"w":0.}),(1,True)):
            with self.assertRaises(AssertionError):equal_trajectory(actual,expected)


if __name__=="__main__":unittest.main()
