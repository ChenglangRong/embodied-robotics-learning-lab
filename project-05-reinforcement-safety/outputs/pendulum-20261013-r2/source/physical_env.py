"""Continuous goal-conditioned MuJoCo environment; never teleports during step."""
from dataclasses import asdict, dataclass
from pathlib import Path
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT/"model/arm2d-rl.xml"


@dataclass(frozen=True)
class PhysicsConfig:
    dt: float = .02
    substeps: int = 5
    horizon: int = 150
    tolerance: float = .045
    velocity_limit: float = 2.0
    torque_limit: float = 6.0
    guard: bool = True
    mass_scale: float = 1.0


class ArmPhysics:
    obs_dim, action_dim = 11, 2
    def __init__(self, cfg=PhysicsConfig(), split="train"):
        if cfg.dt != .004*cfg.substeps or cfg.horizon < 1 or cfg.mass_scale <= 0:
            raise ValueError("dt/substeps/model timestep contract and positive horizon/mass required")
        self.cfg, self.split = cfg, split
        self.m = mujoco.MjModel.from_xml_string(MODEL.read_text(encoding="utf-8"))
        self.m.body_mass[2:4] *= cfg.mass_scale
        self.m.body_inertia[2:4] *= cfg.mass_scale
        self.d = mujoco.MjData(self.m)
        mujoco.mj_setConst(self.m, self.d)
        self.tip_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self.renderer = None
        self.reset(0)

    def reset(self, seed=None):
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        qgoal = np.array([self.rng.uniform(-.7, .7), self.rng.uniform(.4, 1.8)])
        if self.split == "ood":
            qgoal[1] = -qgoal[1]
        self.goal = self.fk(qgoal)
        self.d.qpos[:] = np.clip(qgoal+self.rng.uniform(-.6, .6, 2), -2.3, 2.3)
        self.d.mocap_pos[0] = [*self.goal, .1]
        self.steps, self.dwell, self.ended = 0, 0, False
        mujoco.mj_forward(self.m, self.d)
        return self.observe(), {"goal": self.goal.tolist()}

    @staticmethod
    def fk(q):
        return np.array([.6*np.cos(q[0])+.4*np.cos(q.sum()), .6*np.sin(q[0])+.4*np.sin(q.sum())])

    def observe(self):
        q = self.d.qpos
        error = self.goal-self.d.site_xpos[self.tip_id, :2]
        return np.r_[np.sin(q), np.cos(q), self.d.qvel/4, self.goal, error, self.dwell/5]

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("two finite normalized action components required")
        if self.steps >= self.cfg.horizon or getattr(self, "ended", False):
            raise RuntimeError("reset required")
        raw_velocity = self.cfg.velocity_limit*np.clip(action, -1, 1)
        velocity = raw_velocity.copy()
        if self.cfg.guard:
            velocity[(np.abs(self.d.qpos)>2.2)&(self.d.qpos*velocity>0)] = 0
        previous_error = np.linalg.norm(self.goal-self.d.site_xpos[self.tip_id, :2])
        max_torque = 0.0
        for _ in range(self.cfg.substeps):
            torque = np.clip(3*(velocity-self.d.qvel), -self.cfg.torque_limit, self.cfg.torque_limit)
            self.d.ctrl[:] = torque
            max_torque = max(max_torque, float(np.max(np.abs(torque))))
            mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self.steps += 1
        error = float(np.linalg.norm(self.goal-self.d.site_xpos[self.tip_id, :2]))
        self.dwell = self.dwell+1 if error < self.cfg.tolerance and np.linalg.norm(self.d.qvel)<.4 else 0
        success = self.dwell >= 5
        violation = bool(np.any(np.abs(self.d.qpos)>2.5) or not np.isfinite(self.d.qpos).all())
        terminated = bool(success or violation)
        truncated = self.steps >= self.cfg.horizon and not terminated
        self.ended = terminated or truncated
        reward = 5*(previous_error-error)-2*error-.01-.002*float(np.sum(velocity**2))+3*success-3*violation
        info = {"success": bool(success), "violation": violation, "error_m": error,
                "guard_intervention": bool(np.any(velocity != raw_velocity)), "torque_peak_nm": max_torque,
                "raw_velocity": raw_velocity.tolist(), "applied_velocity": velocity.tolist(),
                "q": self.d.qpos.tolist(), "dq": self.d.qvel.tolist(), "goal": self.goal.tolist()}
        return self.observe(), reward, terminated, truncated, info

    def dls(self):
        q = self.d.qpos
        j = np.array([[-.6*np.sin(q[0])-.4*np.sin(q.sum()), -.4*np.sin(q.sum())],
                      [.6*np.cos(q[0])+.4*np.cos(q.sum()), .4*np.cos(q.sum())]])
        desired = 4*(self.goal-self.d.site_xpos[self.tip_id, :2])
        velocity = j.T@np.linalg.solve(j@j.T+.05**2*np.eye(2), desired)
        return np.clip(velocity/self.cfg.velocity_limit, -1, 1)

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, width=480, height=360)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [.25, .05, .1]
        camera.distance, camera.azimuth, camera.elevation = 2.0, 90, -80
        self.renderer.update_scene(self.d, camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer:
            self.renderer.close()


class Pendulum:
    obs_dim, action_dim = 3, 1
    def __init__(self):
        import gymnasium as gym
        self.env = gym.make("Pendulum-v1")

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        obs[2] /= 8
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(2*np.clip(action, -1, 1))
        obs[2] /= 8
        return obs, reward/10, terminated, truncated, {"raw_reward": reward, **info}

    def close(self):
        self.env.close()
