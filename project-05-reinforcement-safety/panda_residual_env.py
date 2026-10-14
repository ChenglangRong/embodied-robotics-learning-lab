"""Panda placement residual; frozen physical grasp/transport prefix and position servos."""
from dataclasses import dataclass, asdict
import hashlib
import importlib.util
from pathlib import Path
import sys
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
LAB_SOURCE = ROOT/"lab.py" if (ROOT/"lab.py").exists() else ROOT.parent/"project-03-robot-platforms/lab.py"
LAB_SPEC = importlib.util.spec_from_file_location("panda_nominal_lab", LAB_SOURCE)
lab = importlib.util.module_from_spec(LAB_SPEC)
sys.modules[LAB_SPEC.name] = lab
LAB_SPEC.loader.exec_module(lab)

PREFIX_CACHE = {}


@dataclass(frozen=True)
class Config:
    dt: float = .04
    residual_m: float = .08
    slew_m: float = .008
    filter_enabled: bool = True
    tolerance_m: float = .035
    horizon: int = 125


class PandaResidual:
    obs_dim, action_dim = 27, 2

    def __init__(self, cfg=Config()):
        self.cfg = cfg
        if cfg.dt != .04 or cfg.horizon != 125 or not 0 < cfg.residual_m <= .12 or cfg.slew_m <= 0:
            raise ValueError("Frozen .04s/125-step protocol; residual in (0,.12] m; positive slew")
        self.m, self.d, qa, va, self.model_hash = lab.build("panda")
        self.c = lab.Controller("panda", self.m, self.d, qa, va)
        self.obj = self.m.body("object").id
        self.tcp = self.m.site("tcp").id
        self.og = self.m.geom("object_geom").id
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        # Every reset starts with an identical verified grasp and transport.
        self.prefix_state = np.empty(mujoco.mj_stateSize(self.m, self.state_spec))
        if self.model_hash not in PREFIX_CACHE:
            for _ in range(4500):
                if _ % 5 == 0:
                    self.c.step()
                mujoco.mj_step(self.m, self.d)
            mujoco.mj_forward(self.m, self.d)
            mujoco.mj_getState(self.m, self.d, self.prefix_state, self.state_spec)
            self.prefix_plan = self.c.plan.qpos.copy()
            PREFIX_CACHE[self.model_hash] = (self.prefix_state.copy(), self.prefix_plan.copy())
        else:
            self.prefix_state[:], self.prefix_plan = PREFIX_CACHE[self.model_hash][0], PREFIX_CACHE[self.model_hash][1].copy()
            mujoco.mj_setState(self.m, self.d, self.prefix_state, self.state_spec)
            mujoco.mj_forward(self.m, self.d)
        self.prefix_contacts = self.contacts()
        self.prefix_object = self.d.xpos[self.obj].copy()
        if not self.prefix_contacts.all() or self.prefix_object[2] < .38:
            raise RuntimeError("Frozen prefix failed bilateral grasp/lift gate")
        self.renderer = None
        self.reset(0)

    def contacts(self):
        found = np.zeros(2, dtype=bool)
        for i, contact in enumerate(self.d.contact):
            if self.og not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.og else contact.geom1
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, int(self.m.geom_bodyid[other])) or ""
            force = np.zeros(6)
            mujoco.mj_contactForce(self.m, self.d, i, force)
            for j, finger in enumerate(("left_finger", "right_finger")):
                found[j] |= finger in name and abs(force[0]) > .01
        return found

    def reset(self, seed=None, split="test"):
        if seed is not None or not hasattr(self, "rng"):
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_setState(self.m, self.d, self.prefix_state, self.state_spec)
        self.c.plan.qpos[:] = self.prefix_plan
        mujoco.mj_forward(self.m, self.d)
        angle = self.rng.uniform(-np.pi, np.pi)
        radius = self.rng.uniform(.04, .065) if split != "ood" else self.rng.uniform(.075, .095)
        self.goal = np.r_[np.array([.5, -.19])+radius*np.array([np.cos(angle), np.sin(angle)]), .275]
        self.m.site("goal").pos[:2] = self.goal[:2]
        self.offset = np.zeros(2)
        self.steps, self.ended, self.rows = 0, False, []
        self.rel_start = self.relative()
        self.previous_error = self.error()
        return self.observe(), {"goal": self.goal.tolist(), "split": split}

    def relative(self):
        return self.d.site_xmat[self.tcp].reshape(3, 3).T @ (self.d.xpos[self.obj]-self.d.site_xpos[self.tcp])

    def error(self):
        return float(np.linalg.norm(self.goal[:2]-self.d.xpos[self.obj, :2]))

    def observe(self):
        return np.r_[self.d.qpos[:7]/3, self.d.qvel[:7]/3,
                     (self.goal[:2]-[.5, -.19])/.08,
                     (self.goal[:2]-self.d.xpos[self.obj, :2])/.08,
                     (self.d.xpos[self.obj, 2]-.275)/.2,
                     self.relative()/.1, self.contacts().astype(float), self.offset/.08, self.steps/125]

    def analytical(self):
        return np.clip((self.goal[:2]-[.5, -.19])/self.cfg.residual_m, -1, 1)

    def step(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (2,) or not np.isfinite(a).all():
            raise ValueError("Two finite normalized Cartesian residuals required")
        if self.ended:
            raise RuntimeError("reset required")
        requested = np.clip(a, -1, 1)*self.cfg.residual_m
        applied = self.offset+np.clip(requested-self.offset, -self.cfg.slew_m, self.cfg.slew_m) if self.cfg.filter_enabled else requested
        self.offset = applied
        t = self.d.time
        z = .48+lab.smooth((t-9)/1.6)*(.285-.48) if t < 12 else .285+lab.smooth((t-12)/1.6)*(.48-.285)
        xyz = np.r_[np.array([.5, -.19])+applied, z]
        self.c.target[:7] = self.c.ik(xyz)
        self.d.ctrl[:7] = self.c.target[:7]
        self.d.ctrl[7] = 0 if t < 11 else 255
        max_force = 0.
        for _ in range(20):
            mujoco.mj_step(self.m, self.d)
            max_force = max(max_force, float(np.max(np.abs(self.d.actuator_force[:7]))))
        mujoco.mj_forward(self.m, self.d)
        self.steps += 1
        error = self.error()
        violation = bool(np.any(self.d.qpos[:7] < self.m.jnt_range[:7, 0]-.01) or np.any(self.d.qpos[:7] > self.m.jnt_range[:7, 1]+.01) or not np.isfinite(self.d.qpos).all())
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.obj, vel, 0)
        row = {"step": self.steps, "time_s": float(self.d.time), "phase": "place" if t<11 else "release" if t<12 else "retreat",
               "error_m": error, "object_xyz": self.d.xpos[self.obj].tolist(), "object_speed_m_s": float(np.linalg.norm(vel[3:])),
               "finger_contacts": self.contacts().tolist(), "bilateral_contact": bool(self.contacts().all()), "relative_drift_m": float(np.linalg.norm(self.relative()-self.rel_start)),
               "raw_action": a.tolist(), "requested_offset_m": requested.tolist(), "applied_offset_m": applied.tolist(),
               "filter_intervention": bool(np.any(np.abs(requested-applied)>1e-12)), "violation": violation,
               "actuator_force_peak_nm": max_force, "gripper_command": float(self.d.ctrl[7]), "q": self.d.qpos[:7].tolist()}
        self.rows.append(row)
        finished = self.steps == self.cfg.horizon
        metrics = self.assess() if finished else {}
        success = bool(metrics.get("success", False))
        reward = 20*(self.previous_error-error)-10*error-.002*float(np.sum(a*a))+5*success-5*violation
        self.previous_error = error
        self.ended = bool(finished or violation)
        # Completed scheduled release/settle is a task terminal, including failed placements.
        return self.observe(), reward, self.ended, False, {**row, "success": success, **metrics}

    def assess(self):
        transport = [r for r in self.rows if r["time_s"] <= 10.]
        final = self.rows[-25:]
        retention = float(np.mean([r["bilateral_contact"] for r in transport])) if transport else 0.
        drift = max((r["relative_drift_m"] for r in transport), default=0.)
        error = float(np.mean([r["error_m"] for r in final]))
        height = float(np.mean([r["object_xyz"][2] for r in final]))
        speed = float(np.mean([r["object_speed_m_s"] for r in final]))
        released = not any(any(r["finger_contacts"]) for r in final)
        criteria = {"placement": error <= self.cfg.tolerance_m, "table_height": abs(height-.275)<.015,
                    "settled": speed < .03, "released": released, "transport_contact": retention >= .9,
                    "transport_slip": drift < .04, "joint_bounds": not any(r["violation"] for r in self.rows)}
        return {"success": bool(all(criteria.values())), "criteria": criteria, "final_error_m": error,
                "final_height_m": height, "final_speed_m_s": speed, "transport_contact_fraction": retention, "transport_relative_drift_m": drift}

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, width=640, height=480)
        self.renderer.update_scene(self.d, camera=lab.camera("panda"))
        return self.renderer.render().copy()

    def close(self):
        if self.renderer:
            self.renderer.close()
