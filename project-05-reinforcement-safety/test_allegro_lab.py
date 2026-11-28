import unittest

import numpy as np

from allegro_lab import Hand, SLIP_GAIN, THUMB_PROXIMAL, axis_angle, episode, rotation_error_deg, slip_correction


class AllegroTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hand = Hand()

    @classmethod
    def tearDownClass(cls):
        cls.hand.close()

    def test_wrist_frame_rotation_distance(self):
        wrist = axis_angle([0, 1, 0], 90)
        relative = axis_angle([1, 0, 0], 20)
        self.assertAlmostEqual(rotation_error_deg(wrist.T @ (wrist @ relative), np.eye(3)), 20)
        self.assertAlmostEqual(rotation_error_deg(relative, relative), 0, places=5)

    def test_correction_deadband_and_limit(self):
        self.assertEqual(slip_correction(.0008, SLIP_GAIN), 0)
        self.assertAlmostEqual(slip_correction(.005, SLIP_GAIN), -.112)
        self.assertEqual(slip_correction(.05, SLIP_GAIN), -.25)

    def test_matched_load_and_zero_gain(self):
        baseline, rows = episode(self.hand, "pull13", "hold", 5000000)
        feedback, repaired = episode(self.hand, "pull13", "feedback", 5000000)
        _, zero = episode(self.hand, "pull13", "feedback", 5000000, gain=0)
        self.assertEqual(zero, rows)
        self.assertEqual([r["force_world_n"] for r in rows], [r["force_world_n"] for r in repaired])
        self.assertEqual(feedback["load_steps"], 15)
        self.assertAlmostEqual(feedback["impulse_y_n_s"], 3.9)
        self.assertFalse(baseline["criteria"]["limited_drift"])
        self.assertTrue(feedback["success"])

    def test_finger_only_reorientation(self):
        hold, baseline = episode(self.hand, "reorient", "hold", 5000000)
        roll, moved = episode(self.hand, "reorient", "roll", 5000000)
        self.assertFalse(hold["criteria"]["target_held"])
        self.assertTrue(roll["success"])
        for fixed, moving in zip(baseline, moved, strict=True):
            a, b = fixed["q_ref_rad"], moving["q_ref_rad"]
            self.assertTrue(all(x == y for i, (x, y) in enumerate(zip(a, b)) if i != THUMB_PROXIMAL))
            self.assertEqual(moving["force_world_n"], [0., 0., 0.])
            self.assertEqual(moving["external_torque_world_nm"], [0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
