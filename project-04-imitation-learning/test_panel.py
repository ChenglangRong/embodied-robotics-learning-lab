"""Headless callback tests; actual native button operation is a separate check."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
from learning import TwoLinkReachEnv, MLP
from panel import LearningPanel


def fixture(name="expert"):
    env,ghost=TwoLinkReachEnv(),TwoLinkReachEnv()
    for arm in (env,ghost):arm.reset(seed=11,goal=(.55,.55))
    return SimpleNamespace(env=env,ghost=ghost,ghost_done=False,trace=[],done=False,
                           policy_name=SimpleNamespace(get=lambda:name),model=None,
                           running=True,draw=Mock(),show_state=Mock(),set_status=Mock())


class CallbackTests(unittest.TestCase):
    def test_single_step_pauses_and_records_applied_velocity(self):
        view=fixture()
        q=view.env.q.copy()
        LearningPanel.step(view,manual=True)
        self.assertFalse(view.running)
        self.assertEqual(view.env.step_count,1)
        row=view.trace[0]
        velocity=np.array([row["qdot1_rad_s"],row["qdot2_rad_s"]])
        np.testing.assert_allclose(view.env.q,q+view.env.cfg.dt*velocity)
        np.testing.assert_array_equal(view.env.q,view.ghost.q)
        view.draw.assert_called_once()

    def test_zero_controller_records_timeout(self):
        view=fixture("zero")
        q=view.env.q.copy()
        for _ in range(600):LearningPanel.step(view)
        self.assertTrue(view.done)
        self.assertFalse(view.running)
        self.assertEqual(len(view.trace),600)
        np.testing.assert_array_equal(view.env.q,q)
        view.set_status.assert_called()

    def test_missing_model_leaves_state_unchanged(self):
        view=fixture("bc")
        LearningPanel.step(view)
        self.assertEqual(view.env.step_count,0)
        self.assertEqual(view.trace,[])

    def test_learned_controller_uses_model_output(self):
        view=fixture("bc")
        view.model=MLP(1,width=4)
        q=view.env.q.copy()
        velocity=np.clip(view.model(view.env),-2,2)
        LearningPanel.step(view)
        np.testing.assert_allclose(view.env.q,q+.02*velocity)

    def test_finished_episode_receives_no_more_steps(self):
        view=fixture()
        view.done=True
        LearningPanel.step(view)
        self.assertEqual(view.env.step_count,0)
        view.draw.assert_not_called()


if __name__=="__main__":unittest.main()
