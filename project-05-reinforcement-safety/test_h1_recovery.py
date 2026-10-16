import importlib
from pathlib import Path
import unittest
import numpy as np
from h1_recovery import (CONTROLLERS,SCENARIOS,SCENARIO_PUSH,REFERENCE_ANKLE,REFERENCE_HIP,REFERENCE_KNEE,REFERENCE_PELVIS_Z,
                         Config,H1Stand,episode,push_force,reference_correction)

EXPECTED_CRITERIA={"complete","upright","min_height","max_tilt","recovered","recovery_time",
                   "final_stability","pushed_by","slip","double_support"}


class StandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.env=H1Stand()
    @classmethod
    def tearDownClass(cls):cls.env.close()
    def setUp(self):self.env.reset(3000000)

    def test_frozen_protocol_constants(self):
        cfg=Config()
        self.assertEqual((cfg.dt,cfg.substeps,cfg.horizon),(.02,10,300))
        self.assertEqual((cfg.push_onset_s,cfg.feedback_start_s),(2.,.3))
        self.assertEqual((cfg.com_gain,cfg.com_rate_gain,cfg.com_target_m,cfg.filter_alpha),(12.,2.4,.05,.15))
        self.assertEqual((cfg.reference_hip,cfg.reference_knee,cfg.reference_ankle,cfg.reference_pelvis_z),
                         (REFERENCE_HIP,REFERENCE_KNEE,REFERENCE_ANKLE,REFERENCE_PELVIS_Z))
        self.assertEqual(cfg.kp_leg,(160.,240.,300.,300.,100.))
        self.assertEqual(cfg.kd_scale,.6)
        self.assertEqual(CONTROLLERS,("fixed","com","com_no_damping","wrong_sign"))
        self.assertEqual(SCENARIOS,("none","push_front","strong_front","push_back","push_left"))
        self.assertEqual(SCENARIO_PUSH,{"none":(0.,0.,.2),"push_front":(40.,0.,.2),"strong_front":(60.,0.,.2),
                                         "push_back":(-60.,0.,.2),"push_left":(0.,120.,.2)})

    def test_reference_stance_and_reset_protocol(self):
        home=self.env.home
        self.assertEqual(len(home),self.env.m.nu)
        for side in (0,5):
            self.assertAlmostEqual(home[side+2],REFERENCE_HIP)
            self.assertAlmostEqual(home[side+3],REFERENCE_KNEE)
            self.assertAlmostEqual(home[side+4],REFERENCE_ANKLE)
        self.assertEqual(int(np.count_nonzero(home)),6)
        self.assertEqual(self.env.steps,0)
        self.assertEqual(self.env.rows,[])
        self.assertFalse(self.env.ended)
        self.assertIsNone(self.env.settled_from)
        self.assertLess(np.abs(self.env.d.qpos[:2]).max(),.01)
        self.assertLess(np.abs(self.env.d.qpos[self.env.qa]-home).max(),.01)
        self.assertLess(np.abs(self.env.d.qvel[:6]).max(),.02)

    def test_reset_reproducibility(self):
        first=self.env.reset(3000000)
        qpos,qvel=self.env.d.qpos.copy(),self.env.d.qvel.copy()
        second=self.env.reset(3000000)
        np.testing.assert_array_equal(qpos,self.env.d.qpos)
        np.testing.assert_array_equal(qvel,self.env.d.qvel)
        self.assertEqual(first["error_x"],second["error_x"])
        self.env.reset(3000001)
        self.assertFalse(np.array_equal(qpos,self.env.d.qpos))

    def test_control_period_and_substeps(self):
        cfg=self.env.cfg
        self.assertAlmostEqual(self.env.m.opt.timestep,cfg.dt/cfg.substeps)
        start=self.env.d.time
        row=self.env.step("com","none")
        self.assertAlmostEqual(self.env.d.time-start,cfg.dt)
        self.assertEqual(row["step"],1)
        self.assertAlmostEqual(row["time_s"],cfg.dt)
        self.assertEqual(len(row["joint_reference_rad"]),self.env.m.nu)
        self.assertEqual(len(row["joint_position_rad"]),self.env.m.nu)
        self.assertEqual(len(row["joint_velocity_rad_s"]),self.env.m.nu)
        self.assertEqual(len(row["applied_torque_nm"]),self.env.m.nu)

    def test_push_window_and_direction(self):
        cfg=Config()
        self.assertFalse(push_force("none",2.1,cfg).any())
        np.testing.assert_array_equal(push_force("push_front",1.999,cfg),[0.,0.,0.])
        np.testing.assert_array_equal(push_force("push_front",2.0,cfg),[40.,0.,0.])
        np.testing.assert_array_equal(push_force("push_front",2.199,cfg),[40.,0.,0.])
        np.testing.assert_array_equal(push_force("push_front",2.2,cfg),[0.,0.,0.])
        np.testing.assert_array_equal(push_force("strong_front",2.1,cfg),[60.,0.,0.])
        np.testing.assert_array_equal(push_force("push_back",2.1,cfg),[-60.,0.,0.])
        np.testing.assert_array_equal(push_force("push_left",2.1,cfg),[0.,120.,0.])
        with self.assertRaises(ValueError):push_force("unknown",2.1,cfg)
        self.env.reset(3000000)
        rows=[self.env.step("com","push_front") for _ in range(110)]
        pushed=[r for r in rows if any(r["push_force_n"])]
        self.assertTrue(all(not any(r["push_force_n"]) for r in rows[:100]))
        self.assertEqual(len(pushed),10)
        self.assertTrue(all(2.0<r["time_s"]<=2.2 for r in pushed))
        self.assertTrue(all(r["push_force_n"][0]==40. and r["push_force_n"][1:]==[0.,0.] for r in pushed))

    def test_reference_correction_sign_and_matched_ablation(self):
        cfg,error,rate=Config(),.04,-.3
        self.assertEqual(reference_correction("fixed",error,rate,cfg),0.)
        com=reference_correction("com",error,rate,cfg)
        self.assertAlmostEqual(com,cfg.com_gain*error+cfg.com_rate_gain*rate)
        np.testing.assert_allclose(reference_correction("wrong_sign",error,rate,cfg),-com)
        self.assertAlmostEqual(reference_correction("com_no_damping",error,rate,cfg),cfg.com_gain*error)
        self.assertAlmostEqual(reference_correction("com_no_damping",error,rate,cfg),
                               reference_correction("com_no_damping",error,-10.*rate,cfg))
        self.assertAlmostEqual(reference_correction("com",error,rate,cfg)-reference_correction("com",error,0.,cfg),
                               cfg.com_rate_gain*rate,places=15)
        for mode in ("com_feedback","pitch","unknown"):
            with self.assertRaises(ValueError):reference_correction(mode,error,rate,cfg)

    def test_matched_controllers_separate_by_outcome(self):
        good,good_rows,_=episode(self.env,"com","none",3000000)
        bad,bad_rows,_=episode(self.env,"wrong_sign","none",3000000)
        self.assertTrue(good["criteria"]["upright"] and good["criteria"]["recovered"])
        self.assertFalse(bad["criteria"]["upright"])
        self.assertLess(len(bad_rows),len(good_rows))

    def test_com_target_offset_and_metric_definitions(self):
        state=self.env.measure()
        self.assertAlmostEqual(state["error_x"],state["com"][0]-state["support"][0]-self.env.cfg.com_target_m,places=15)
        self.assertAlmostEqual(state["error_y"],state["com"][1]-state["support"][1],places=15)
        np.testing.assert_allclose(state["support"],
                                   .5*(self.env.d.xpos[self.env.left_ankle]+self.env.d.xpos[self.env.right_ankle]))
        row=self.env.step("com","none")
        self.assertAlmostEqual(row["com_error_x_m"],row["com_x_m"]-row["support_x_m"]-Config().com_target_m,places=12)
        self.assertAlmostEqual(row["com_error_y_m"],row["com_y_m"]-row["support_y_m"],places=12)
        record,rows,_=episode(self.env,"com","none",3000000)
        tail=rows[-25:]
        self.assertAlmostEqual(record["settled_com_offset_m"],rows[-1]["com_error_x_m"]+Config().com_target_m,places=12)
        self.assertAlmostEqual(record["tail_com_rate_m_s"],
                               float(np.hypot(np.mean([r["com_rate_x_m_s"] for r in tail]),
                                              np.mean([r["com_rate_y_m_s"] for r in tail]))),places=12)

    def test_push_peak_and_lateral_metrics_measure_the_push_window(self):
        record,rows,_=episode(self.env,"com","push_front",3000000)
        pre=[r for r in rows if 1.5<=r["time_s"]<2.0]
        after=[r for r in rows if r["time_s"]>=2.0]
        base=(float(np.mean([r["com_x_m"] for r in pre])),float(np.mean([r["com_y_m"] for r in pre])))
        self.assertAlmostEqual(record["pre_push_com_x_m"],base[0],places=12)
        self.assertAlmostEqual(record["push_peak_com_error_m"],
                               max(float(np.hypot(r["com_x_m"]-base[0],r["com_y_m"]-base[1])) for r in after),places=12)
        self.assertAlmostEqual(record["max_lateral_error_m"],max(abs(r["com_error_y_m"]) for r in after),places=12)
        self.assertGreater(record["push_peak_com_error_m"],record["peak_com_error_m"]*0)

    def test_fall_detection_and_termination_semantics(self):
        self.env.reset(3000000)
        while not self.env.ended:self.env.step("fixed","none")
        self.assertLess(self.env.steps,self.env.cfg.horizon)
        last=self.env.rows[-1]
        self.assertTrue(last["fallen"])
        self.assertTrue(last["pelvis_height_m"]<.62 or max(abs(last["pitch_rad"]),abs(last["roll_rad"]))>.55)
        with self.assertRaises(RuntimeError):self.env.step("fixed","none")
        record,rows,_=episode(self.env,"fixed","none",3000000)
        self.assertEqual(record["steps"],len(rows))
        self.assertFalse(record["criteria"]["complete"])
        self.assertFalse(record["criteria"]["upright"])
        self.assertFalse(record["success"])
        record,rows,_=episode(self.env,"com","none",3000000)
        self.assertEqual(record["steps"],self.env.cfg.horizon)
        self.assertTrue(self.env.ended and record["criteria"]["complete"])

    def test_contacts_and_slip_records(self):
        contacts,slips=self.env.foot_contacts()
        self.assertEqual(contacts.shape,(2,))
        self.assertEqual(contacts.dtype,bool)
        self.assertIsInstance(slips,list)
        row=self.env.step("com","none")
        self.assertEqual(len(row["foot_contacts"]),2)
        self.assertIsInstance(row["foot_contacts"][0],bool)
        self.assertIsInstance(row["contact_slip_m_s"],list)
        self.assertGreaterEqual(len(row["contact_slip_m_s"]),sum(row["foot_contacts"]))
        self.assertLessEqual(len(row["contact_slip_m_s"]),len(self.env.foot_geoms))
        self.assertTrue(all(s>=0. for s in row["contact_slip_m_s"]))

    def test_criteria_and_success_consistency(self):
        for mode,scenario in (("com","none"),("fixed","none"),("com","push_left")):
            record,rows,_=episode(self.env,mode,scenario,3000000)
            self.assertEqual(set(record["criteria"]),EXPECTED_CRITERIA)
            self.assertEqual(record["success"],all(record["criteria"].values()))
            self.assertEqual(record["steps"],len(rows))
            self.assertEqual(record["criteria"]["complete"],record["steps"]==self.env.cfg.horizon)
            self.assertEqual(record["criteria"]["upright"],not any(r["fallen"] for r in rows))
            window=[r for r in rows if r["time_s"]>=self.env.cfg.push_onset_s]
            self.assertEqual(record["criteria"]["double_support"],all(sum(r["foot_contacts"])==2 for r in window))
            self.assertEqual(record["lost_double_support_samples"],sum(1 for r in window if sum(r["foot_contacts"])!=2))
            self.assertEqual(record["criteria"]["pushed_by"],
                             scenario=="none" or any(any(r["push_force_n"]) for r in rows))
            self.assertAlmostEqual(record["saturation_fraction"],
                                   sum(r["saturated_substeps"] for r in rows)/(self.env.cfg.substeps*len(rows)),places=12)

    def test_invalid_config_and_reference_stance_guard(self):
        for cfg in (Config(dt=.01),Config(substeps=5),Config(horizon=100),Config(com_gain=np.nan),
                    Config(com_gain=-1.),Config(com_rate_gain=-1.),Config(filter_alpha=0.),Config(filter_alpha=1.5),
                    Config(kp_leg=(1.,2.,3.)),Config(reference_knee=0.)):
            with self.assertRaises(ValueError):H1Stand(cfg)
        for cfg in (Config(reference_hip=-.19),Config(reference_knee=.46),Config(reference_ankle=-.24),
                    Config(reference_pelvis_z=1.06)):
            with self.assertRaises(ValueError):H1Stand(cfg)

    def test_import_does_not_pollute_panel(self):
        self.assertEqual(Path(importlib.import_module("panel").__file__).resolve().parent,Path(__file__).resolve().parent)


class ReplayToleranceTests(unittest.TestCase):
    def test_observed_roundoff_difference_is_measured(self):
        from verify_h1 import equal_trajectory
        stats={"compared_float_values":0,"maximum_absolute_difference":0.,"largest_difference_path":None,"maximum_tolerance_fraction":0.}
        observed,expected=0.0008412553537757717,0.0008412532144022045
        equal_trajectory({"applied_torque_nm":[observed]},{"applied_torque_nm":[expected]},stats=stats)
        self.assertEqual(stats["compared_float_values"],1)
        self.assertAlmostEqual(stats["maximum_absolute_difference"],abs(observed-expected),places=20)
        self.assertLess(stats["maximum_tolerance_fraction"],1.)
        self.assertIn("applied_torque_nm",stats["largest_difference_path"])

    def test_meaningful_error_and_nonfinite_values_fail(self):
        from verify_h1 import equal_trajectory
        for value in (0.0009412532144022045,float("nan"),float("inf")):
            with self.assertRaises(AssertionError):equal_trajectory(value,0.0008412532144022045)

    def test_relative_tolerance_boundary(self):
        from verify_h1 import equal_trajectory
        equal_trajectory(1000.00000005,1000.,label="roundoff")
        with self.assertRaises(AssertionError):equal_trajectory(1000.001,1000.,label="physical difference")

    def test_discrete_state_and_structure_stay_exact(self):
        from verify_h1 import equal_trajectory
        for actual,expected in ((True,False),(1,2),([1.],[]),({"v":0.},{"w":0.}),(1,True)):
            with self.assertRaises(AssertionError):equal_trajectory(actual,expected)


if __name__=="__main__":unittest.main()
