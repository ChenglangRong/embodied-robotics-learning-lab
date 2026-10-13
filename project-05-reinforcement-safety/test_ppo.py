import tempfile
from pathlib import Path
import unittest
import numpy as np
from ppo import Agent, gae, log_probability
from physical_env import ArmPhysics, Pendulum, PhysicsConfig


class PPOTests(unittest.TestCase):
    def test_gae_truncation_bootstrap_and_reset_boundary(self):
        r = np.array([[1.], [2.], [3.]])
        v, nv = np.array([[.5], [.7], [.2]]), np.array([[.7], [5.], [9.]])
        adv, ret = gae(r, v, nv, np.array([[False], [False], [True]]), np.array([[False], [True], [False]]), .9, .8)
        np.testing.assert_allclose(adv[:, 0], [5.306, 5.8, 2.8])
        np.testing.assert_allclose(ret, adv+v)

    def test_transformed_density(self):
        a = np.linspace(-.99999, .99999, 20000)
        z = np.arctanh(a)[:, None]
        p = np.exp(log_probability(z, np.zeros_like(z), np.zeros(1)))
        self.assertAlmostEqual(float(np.trapz(p, a)), 1., places=4)
        self.assertTrue(np.isfinite(log_probability(np.array([[1000., -1000.]]), np.zeros((1, 2)), np.zeros(2))).all())

    def test_analytic_actor_gradient(self):
        agent = Agent(3, 2, 2)
        x = agent.rng.normal(size=(8, 3))
        _, z, lp = agent.sample(x)
        old = lp+np.linspace(-.3, .3, 8)
        adv = np.array([1, -1, .5, -.5, 2, -2, .3, -.3])
        _, gradients, _ = agent.actor_loss_grad(x, z, old, adv, entropy_coef=.01)
        for k, parameter in agent.actor.p.items():
            indices = [0, parameter.size//2, parameter.size-1]
            for i in set(indices):
                original = parameter.flat[i]
                parameter.flat[i] = original+1e-6
                plus = agent.actor_loss_grad(x, z, old, adv, entropy_coef=.01)[0]
                parameter.flat[i] = original-1e-6
                minus = agent.actor_loss_grad(x, z, old, adv, entropy_coef=.01)[0]
                parameter.flat[i] = original
                self.assertAlmostEqual((plus-minus)/2e-6, gradients[k].flat[i], places=6, msg=k)

    def test_clipped_branch_gradient(self):
        agent = Agent(2, 1, 3)
        x = np.zeros((2, 2))
        _, z, lp = agent.sample(x)
        old = lp-np.log(np.array([1.5, .5]))
        _, gradient, diag = agent.actor_loss_grad(x, z, old, np.array([1., -1.]))
        self.assertTrue(all(np.allclose(g, 0) for g in gradient.values()))
        self.assertEqual(diag["clip_fraction"], 1.)

    def test_save_load_and_action_bounds(self):
        agent = Agent(11, 2)
        x = agent.rng.normal(size=(30, 11))
        action, _, _ = agent.sample(x)
        self.assertTrue((np.abs(action)<=1).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.npz"
            agent.save(path)
            loaded = Agent.load(path)
            np.testing.assert_array_equal(agent.sample(x, True)[0], loaded.sample(x, True)[0])

    def test_physical_reset(self):
        env = ArmPhysics()
        a, _ = env.reset(42)
        env.step([0, 0])
        b, _ = env.reset(42)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (11,))
        env.close()

    def test_inertia_and_guard(self):
        env = ArmPhysics(PhysicsConfig(mass_scale=1.5))
        base = ArmPhysics()
        np.testing.assert_allclose(env.m.body_mass[2:4], base.m.body_mass[2:4]*1.5)
        np.testing.assert_allclose(env.m.body_inertia[2:4], base.m.body_inertia[2:4]*1.5)
        env.d.qpos[:] = [2.3, 0]
        env.d.qvel[:] = 0
        _, _, _, _, info = env.step([1, 0])
        self.assertTrue(info["guard_intervention"])
        self.assertEqual(info["applied_velocity"][0], 0)
        self.assertLessEqual(info["torque_peak_nm"], 6)
        env.close()
        base.close()

    def test_physical_step_has_inertia(self):
        env = ArmPhysics()
        env.reset(0)
        before = env.d.qpos.copy()
        env.step([.5, -.5])
        self.assertGreater(np.linalg.norm(env.d.qpos-before), 0)
        self.assertFalse(np.allclose(env.d.qpos-before, [.02, -.02]))
        env.close()

    def test_physical_terminal_reset_and_dls(self):
        env = ArmPhysics(PhysicsConfig(horizon=1))
        _, _, _, truncated, _ = env.step([0, 0])
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            env.step([0, 0])
        env.reset(3)
        self.assertTrue(np.isfinite(env.step(env.dls())[0]).all())
        env.close()

    def test_pendulum_contract(self):
        env = Pendulum()
        a, _ = env.reset(5)
        b, _ = env.reset(5)
        np.testing.assert_array_equal(a, b)
        obs, r, _, _, info = env.step([.2])
        self.assertEqual(obs.shape, (3,))
        self.assertAlmostEqual(r*10, info["raw_reward"])
        env.close()


if __name__ == "__main__":
    unittest.main()
