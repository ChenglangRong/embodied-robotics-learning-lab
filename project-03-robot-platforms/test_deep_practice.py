import unittest
import types
import mujoco
import numpy as np
from deep_practice import contact_on_geom, tangent_speed, observe, stance_segments, SUITES
from lab import build, Controller


class DiagnosticTests(unittest.TestCase):
    def test_contact_action_reaction(self):
        con = types.SimpleNamespace(geom1=1, geom2=2, frame=np.eye(3).ravel())
        np.testing.assert_array_equal(contact_on_geom(con, [2, 3, 4], 2), [2, 3, 4])
        np.testing.assert_array_equal(contact_on_geom(con, [2, 3, 4], 1), [-2, -3, -4])
        with self.assertRaises(ValueError):
            contact_on_geom(con, [2, 3, 4], 3)

    def test_contact_rotation(self):
        con = types.SimpleNamespace(geom1=0, geom2=1, frame=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).ravel())
        np.testing.assert_array_equal(contact_on_geom(con, [2, 0, 0], 1), [0, 0, 2])

    def test_static_object_force_balances_weight(self):
        for floor_first in (True, False):
            floor = '<geom name="floor" type="plane" size="1 1 .1"/>'
            obj = '<body name="object" pos="0 0 .1"><freejoint/><geom name="ball" type="sphere" size=".1" mass=".1"/></body>'
            model = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".002"/><worldbody>' + (floor + obj if floor_first else obj + floor) + '</worldbody></mujoco>')
            data = mujoco.MjData(model)
            for _ in range(1000):
                mujoco.mj_step(model, data)
            gid = model.geom("ball").id
            fz = 0.
            for i, con in enumerate(data.contact):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                fz += contact_on_geom(con, force, gid)[2]
            self.assertAlmostEqual(fz, .981, places=3)

    def test_observer_preserves_state(self):
        for kind in SUITES:
            model, data, qa, va, _ = build(kind)
            ctrl = Controller(kind, model, data, qa, va)
            ctrl.step()
            mujoco.mj_step(model, data)
            fields = ("qpos", "qvel", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat", "qacc", "efc_force")
            before = [getattr(data, name).copy() for name in fields]
            before_time = data.time
            row = observe(kind, model, data, ctrl)
            self.assertTrue(np.isfinite(row["tracking_rmse_rad"]))
            self.assertEqual(before_time, data.time)
            for old, current in zip(before, (getattr(data, name) for name in fields)):
                np.testing.assert_array_equal(old, current)

    def test_inclined_box_friction_vector_and_generalized_force(self):
        angle = .3
        quat = f"{np.cos(angle/2)} 0 {np.sin(angle/2)} 0"
        normal = np.array([np.sin(angle), 0, np.cos(angle)])
        center = " ".join(str(x) for x in normal * .06)
        xml = f'<mujoco><option timestep=".002" cone="elliptic"/><worldbody><geom name="floor" type="plane" size="2 2 .1" quat="{quat}" friction="1 .01 .001"/><body name="object" pos="{center}" quat="{quat}"><freejoint/><geom name="block" type="box" size=".1 .1 .05" mass=".2" friction="1 .01 .001"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        for _ in range(3000):
            mujoco.mj_step(model, data)
        force_world, tangent_local = np.zeros(3), 0.
        for i, con in enumerate(data.contact):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            force_world += contact_on_geom(con, force, model.geom("block").id)
            tangent_local += np.linalg.norm(force[1:3])
        self.assertGreater(tangent_local, .1)
        np.testing.assert_allclose(force_world, [0, 0, 1.962], atol=1e-4)
        np.testing.assert_allclose(force_world, data.qfrc_constraint[:3], atol=1e-6)

    def test_suite_single_field_change(self):
        for suite in SUITES.values():
            self.assertEqual(suite[0], ("baseline", {}))
            for _, options in suite[1:]:
                self.assertEqual(len(options), 1)

    def test_stance_net_and_path_differ(self):
        rows = [{"time_s": 3 + i * .02, "FL_contact": 1, "FL_x_m": x, "FL_y_m": 0.} for i, x in enumerate([0, .01, 0, .01, 0])]
        segment = stance_segments(rows, "FL")[0]
        self.assertEqual(segment["net_travel_m"], 0.)
        self.assertAlmostEqual(segment["path_length_m"], .04)

    def test_short_contacts_excluded(self):
        rows = [{"time_s": 3 + i * .02, "FL_contact": int(i != 2), "FL_x_m": .01*i, "FL_y_m": 0.} for i in range(5)]
        self.assertEqual(stance_segments(rows, "FL"), [])

    def test_rolling_surface_point_has_zero_slip(self):
        # radius .1, v_x=.2, omega_y=2: bottom point velocity cancels.
        velocity = np.array([0, 2, 0, .2, 0, 0])
        speed = tangent_speed(velocity, np.array([0, 0, .1]), np.zeros(3), np.array([0, 0, 1]))
        self.assertAlmostEqual(speed, 0.)

    def test_surface_tangent_speed_excludes_normal_motion(self):
        speed = tangent_speed(np.array([0, 0, 0, .3, .4, 9]), np.zeros(3), np.zeros(3), np.array([0, 0, 1]))
        self.assertAlmostEqual(speed, .5)


if __name__ == "__main__":
    unittest.main()
