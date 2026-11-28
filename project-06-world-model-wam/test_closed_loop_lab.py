import unittest

import numpy as np

from closed_loop_lab import (ArmPhysics, BLOCK, EVAL, TRAIN, VALIDATION, branches, choose_plan, dataset,
                             state, tips, weights_from_pilot)


class ClosedLoopTests(unittest.TestCase):
    def test_batch_fk_matches_scalar_for_every_candidate(self):
        joints = np.array([[.2, .7], [-.4, -1.3], [.1, .05]])
        np.testing.assert_allclose(tips(joints), [ArmPhysics.fk(q) for q in joints], atol=1e-14)

    def test_episode_splits_disjoint(self):
        groups = [set(TRAIN), set(VALIDATION), *(set(seeds) for seeds in EVAL.values())]
        self.assertEqual(len(set.union(*groups)), sum(map(len, groups)))

    def test_dls_window_action_timing_and_applied_reference(self):
        env = ArmPhysics()
        try:
            data = dataset(env, [4100])
            initial, seq = data["initial"][0], data["action"][0].reshape(BLOCK, 2)
            expected = initial + data["future"][0].reshape(BLOCK, 4)*[.1, .1, 1, 1]
            env.reset(4100)
            for i, command in enumerate(seq):
                _, _, _, _, info = env.step(command)
                np.testing.assert_allclose(state(env), expected[i], atol=1e-12)
                np.testing.assert_array_equal(info["applied_velocity"], data["applied"][0, i])
            env.reset(4100)
            shifted_states = []
            for command in np.roll(seq, 1, axis=0):
                env.step(command)
                shifted_states.append(state(env))
            self.assertGreater(np.linalg.norm(shifted_states-expected), 1e-5)
        finally:
            env.close()

    def test_masked_candidates_tie_and_action_alignment_matters(self):
        env = ArmPhysics(split="test")
        weights = weights_from_pilot()
        try:
            env.reset(4500)
            action, index, costs = choose_plan(env, weights["mpc_masked"])
            self.assertEqual(index, 0)
            np.testing.assert_array_equal(action, [0, 0])
            np.testing.assert_allclose(costs, costs[0])
            rows = branches(env, 4500, {"action": weights["mpc_action"], "shifted": weights["mpc_action"]})
            aligned, shifted = ([r["tip_error_mm"][-1] for r in rows if r["model"] == name]
                                for name in ("action", "shifted"))
            self.assertLess(np.mean(aligned), np.mean(shifted))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
