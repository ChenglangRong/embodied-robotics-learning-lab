import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_mujoco import build_sim_model, run, tip_from_mujoco


class TestUrdfMujoco(unittest.TestCase):
    def test_urdf_joint_chain_matches_hand_calculation(self):
        with tempfile.TemporaryDirectory() as folder:
            model = build_sim_model(Path(folder) / "derived.xml")
            self.assertEqual((model.nq, model.nv, model.nu), (2, 2, 2))
            data = mujoco.MjData(model)
            data.qpos[:] = [np.pi / 2, np.pi / 2]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(tip_from_mujoco(data), [-0.4, 0.6, 0.0], atol=1e-12)

    def test_fixed_seed_run_is_reproducible(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = run(7, (0.55, 0.55), Path(first), False)
            b = run(7, (0.55, 0.55), Path(second), False)
            self.assertEqual(a, b)
            self.assertTrue(a["success"])
            self.assertLess(a["max_fk_gap_m"], 1e-10)


if __name__ == "__main__":
    unittest.main()
