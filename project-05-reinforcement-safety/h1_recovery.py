"""H1 bipedal standing and push recovery: matched ablations on the pinned Menagerie model.

All controllers share the model, the stiff joint PD, torque limits, push schedule and reset
seeds, so every difference is attributable to one reference-generation rule. The module is an
analytic engineering baseline: no reinforcement-learning training and no stepping.

Calibration facts this file depends on (measured in the offline scripts, all on the pinned
commit 0059d4335f8156206f63a35662313385f7ad6d74):
  * the shipped h1.xml default pose (pelvis 0.98 m, hip/knee/ankle -0.4/0.8/-0.4) drives both
    feet about 10 cm into the floor, so the lab builds its own stance;
  * the foot sole is 0.112 m below the ankle joint when the ankle angle is zero, so a contact
    stance needs the pelvis near 1.063 m;
  * with the ankle joint as the only sagittal actuator, a *negative* ankle reference accelerates
    the centre of mass forward (d a_com / d q_ankle = -2.85 1/s^2 per rad, measured), which is the
    opposite of the human ankle-strategy intuition and is why the ablations below are explicit;
  * the reference leg pose is unstable without feedback and the CoM offset that the closed loop
    settles at is about +0.051 m ahead of the ankle joint.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
LAB_FILE = ROOT / "lab.py" if (ROOT / "lab.py").exists() else ROOT.parent / "project-03-robot-platforms/lab.py"
SPEC = importlib.util.spec_from_file_location("h1_nominal_lab", LAB_FILE)
lab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = lab
SPEC.loader.exec_module(lab)
from tabular_lab import save_csv, save_json

CONTROLLERS = ("fixed", "com", "com_no_damping", "wrong_sign")
SCENARIOS = ("none", "push_front", "strong_front", "push_back", "push_left")
SCENARIO_PUSH = {  # (world force N along +x or +y, duration s); onset is fixed for every case
    "none": (0., 0., .2),
    "push_front": (40., 0., .2),
    "strong_front": (60., 0., .2),
    "push_back": (-60., 0., .2),
    "push_left": (0., 120., .2),
}
# Measured before freezing (development seed 0, offline calibration scripts):
#   * the fixed reference loses the stance with no push at all;
#   * the CoM feedback holds 60 N sagittal pushes and fails at 80 N by saturating the ankle torque;
#   * the CoM strategy holds 120 N lateral pushes 5/5 and fails 4/5 at 140 N by losing support;
#   * the wrong-sign variant amplifies the error and falls in about 0.8 s;
#   * hip-roll lateral feedback changed the lateral error by under 3 mm and the threshold by 0 N,
#     so no case uses it; the lateral response is the ankle strategy acting through both feet.
# The scenario set brackets those edges, so each criterion separates at least one mechanism.
PUSH_ONSET_S = 2.0
FEEDBACK_START_S = .3
# Reference stance (calibrated offline): hip pitch, knee, ankle and the touch height.
REFERENCE_HIP = -.20
REFERENCE_KNEE = .45
REFERENCE_ANKLE = -.25
REFERENCE_PELVIS_Z = 1.06279
FEEDBACK_DELAY_STEPS = 0


@dataclass(frozen=True)
class Config:
    dt: float = .02
    substeps: int = 10
    horizon: int = 300
    push_onset_s: float = PUSH_ONSET_S
    feedback_start_s: float = FEEDBACK_START_S
    com_gain: float = 12.
    com_rate_gain: float = 2.4
    com_target_m: float = .05
    filter_alpha: float = .15
    reference_hip: float = REFERENCE_HIP
    reference_knee: float = REFERENCE_KNEE
    reference_ankle: float = REFERENCE_ANKLE
    reference_pelvis_z: float = REFERENCE_PELVIS_Z
    kp_leg: tuple = (160., 240., 300., 300., 100.)
    kd_scale: float = .6


def push_force(scenario, t, cfg):
    """World-frame horizontal force applied at the pelvis; zero outside the scheduled window."""
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    force_x, force_y, duration = SCENARIO_PUSH[scenario]
    if not cfg.push_onset_s <= t < cfg.push_onset_s + duration:
        return np.zeros(3)
    return np.array([force_x, force_y, 0.])


def reference_correction(mode, com_error, rate, cfg):
    """Sagittal ankle-reference correction (rad).

    `com` uses the centre-of-mass error and its measured rate; `com_no_damping` removes the rate
    term, which is the matched ablation for the damping contribution; `wrong_sign` flips the
    correction and deliberately removes the controller's stability.
    Measured plant sensitivity is negative: a negative ankle reference accelerates the CoM
    forward (d a_com / d q_ankle = -2.85 1/s^2 per rad), so the correction is added with a
    positive sign.
    """
    if mode not in CONTROLLERS:
        raise ValueError("unknown controller")
    if mode == "fixed":
        return 0.
    sign = -1. if mode == "wrong_sign" else 1.
    rate_term = 0. if mode == "com_no_damping" else cfg.com_rate_gain * rate
    return sign * (cfg.com_gain * com_error + rate_term)


# Offline calibration result kept with the code: hip-roll CoM feedback changed the lateral error
# by under 3 mm and moved the lateral failure threshold by 0 N, so no case uses it.


class H1Stand:
    """Physical H1 stand task with matched push-recovery controllers."""

    def __init__(self, cfg=Config()):
        values = [v for k, v in asdict(cfg).items() if k != "kp_leg"]
        if not np.isfinite(values).all() or cfg.dt != .02 or cfg.substeps != 10 or cfg.horizon != 300:
            raise ValueError("invalid config or changed time protocol")
        if min(cfg.com_gain, cfg.com_rate_gain) < 0 or not 0 < cfg.filter_alpha <= 1:
            raise ValueError("invalid gains")
        if min(cfg.reference_pelvis_z, cfg.reference_knee) <= 0 or len(cfg.kp_leg) != 5:
            raise ValueError("invalid reference stance")
        # The stance constants were calibrated offline against the pinned model; changing one
        # without re-calibrating changes the plant under every controller.
        if not np.allclose([cfg.reference_hip, cfg.reference_knee, cfg.reference_ankle, cfg.reference_pelvis_z],
                           [REFERENCE_HIP, REFERENCE_KNEE, REFERENCE_ANKLE, REFERENCE_PELVIS_Z], atol=1e-9):
            raise ValueError("reference stance does not match the calibrated constants")
        self.cfg = cfg
        self.m, self.d, self.qa, self.va, self.model_hash = lab.build("h1")
        self.pelvis = self.m.body("pelvis").id
        self.left_ankle = self.m.body("left_ankle_link").id
        self.right_ankle = self.m.body("right_ankle_link").id
        self.foot_geoms = [i for i in range(self.m.ngeom)
                           if self.m.geom_bodyid[i] in (self.left_ankle, self.right_ankle)
                           and self.m.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE]
        self.floor = self.m.geom("floor").id
        self.kp = np.array(list(cfg.kp_leg) * 2 + [180.] + [60.] * 8)
        self.kd = 2 * np.sqrt(self.kp) * cfg.kd_scale
        self.home = np.zeros(self.m.nu)
        for side in (0, 5):
            self.home[side + 2] = cfg.reference_hip
            self.home[side + 3] = cfg.reference_knee
            self.home[side + 4] = cfg.reference_ankle
        self.reset(0)
        self.renderer = None

    def reset(self, seed):
        """Small reset perturbations around the calibrated stance.

        The margin was measured offline: with joint noise of 3 mrad / 10 mm/s the reference pose
        tips before the CoM feedback can recover it, so the frozen protocol uses 1 mrad / 2 mm/s.
        """
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:7] = [rng.normal(0, .001), rng.normal(0, .001),
                           self.cfg.reference_pelvis_z + rng.normal(0, .0005), 1, 0, 0, 0]
        self.d.qpos[self.qa] = self.home + rng.normal(0, .001, self.m.nu)
        self.d.qvel[:6] = rng.normal(0, .002, 6)
        # Start exactly at the calibrated touch height with no pre-settle: any pre-settle or
        # hovering start lets the pose tip before the CoM feedback is allowed to act.
        mujoco.mj_forward(self.m, self.d)
        self.steps = 0
        self.rows = []
        self.filtered = np.zeros(2)
        self.ended = False
        self.settled_from = None
        return self.measure()

    def support_centre(self):
        """Midpoint of the two ankle joints, in world coordinates."""
        return .5 * (self.d.xpos[self.left_ankle] + self.d.xpos[self.right_ankle])

    def measure(self):
        rotation = self.d.xmat[self.pelvis].reshape(3, 3)
        com = self.d.subtree_com[self.pelvis].copy()
        support = self.support_centre()
        pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])))
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        return {"com": com, "support": support, "pitch": pitch, "roll": roll,
                "error_x": float(com[0] - support[0] - self.cfg.com_target_m),
                "error_y": float(com[1] - support[1]),
                "rate_x": float(self.d.qvel[0]), "rate_y": float(self.d.qvel[1]),
                "height": float(self.d.qpos[2])}

    def foot_contacts(self):
        contacts = np.zeros(2, dtype=bool)
        slips = []
        for i, c in enumerate(self.d.contact):
            if self.floor not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.floor else c.geom1
            if other not in self.foot_geoms:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.m, self.d, i, force)
            if abs(force[0]) <= 1.:
                continue
            contacts[0 if self.m.geom_bodyid[other] == self.left_ankle else 1] = True
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_GEOM, other, velocity, 0)
            at_contact = velocity[3:] + np.cross(velocity[:3], c.pos - self.d.geom_xpos[other])
            slips.append(float(np.linalg.norm(at_contact[:2])))
        return contacts, slips

    def step(self, mode, scenario):
        if self.ended:
            raise RuntimeError("reset required")
        t = self.steps * self.cfg.dt
        state = self.measure()
        active = t >= self.cfg.feedback_start_s
        if active:
            # Causal IIR on the CoM error; the raw error stays in the trajectory log.
            self.filtered += (-self.filtered + np.array([state["error_x"], state["error_y"]])) * self.cfg.filter_alpha
        sagittal = reference_correction(mode, float(self.filtered[0]), state["rate_x"], self.cfg) if active else 0.
        q = self.home.copy()
        for side in (0, 5):
            q[side + 4] += sagittal
        torque_peak = 0.
        saturated = 0
        for _ in range(self.cfg.substeps):
            raw = self.kp * (q - self.d.qpos[self.qa]) - self.kd * self.d.qvel[self.va] + self.d.qfrc_bias[self.va]
            torque = np.clip(raw, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])
            self.d.ctrl[:] = torque
            saturated += int(np.any(np.abs(raw - torque) > 1e-9))
            torque_peak = max(torque_peak, float(np.max(np.abs(torque))))
            self.d.xfrc_applied[self.pelvis] = [*push_force(scenario, t, self.cfg), 0, 0, 0]
            mujoco.mj_step(self.m, self.d)
        self.d.xfrc_applied[self.pelvis] = 0
        mujoco.mj_forward(self.m, self.d)
        self.steps += 1
        after = self.measure()
        contacts, slips = self.foot_contacts()
        fallen = bool(after["height"] < .62 or max(abs(after["pitch"]), abs(after["roll"])) > .55
                      or not np.isfinite(self.d.qpos).all())
        settled = bool(abs(after["error_x"]) <= .035 and abs(after["error_y"]) <= .03
                       and np.hypot(after["rate_x"], after["rate_y"]) <= .12)
        if settled and self.settled_from is None and t > self.cfg.push_onset_s + .5:
            self.settled_from = t
        if not settled and self.settled_from is not None and not self.cfg.push_onset_s <= t < self.cfg.push_onset_s + .25:
            self.settled_from = None
        row = {"step": self.steps, "time_s": self.steps * self.cfg.dt,
               "push_force_n": push_force(scenario, t, self.cfg).tolist(),
               "com_x_m": float(after["com"][0]), "com_y_m": float(after["com"][1]),
               "support_x_m": float(after["support"][0]), "support_y_m": float(after["support"][1]),
               "com_error_x_m": after["error_x"], "com_error_y_m": after["error_y"],
               "com_rate_x_m_s": after["rate_x"], "com_rate_y_m_s": after["rate_y"],
               "pelvis_height_m": after["height"], "pitch_rad": after["pitch"], "roll_rad": after["roll"],
               "ankle_correction_rad": sagittal,
               "filtered_com_error_x_m": float(self.filtered[0]), "filtered_com_error_y_m": float(self.filtered[1]),
               "foot_contacts": contacts.tolist(), "contact_slip_m_s": slips,
               "torque_peak_nm": torque_peak, "saturated_substeps": saturated, "fallen": fallen,
               "joint_reference_rad": q.tolist(), "joint_position_rad": self.d.qpos[self.qa].tolist(),
               "joint_velocity_rad_s": self.d.qvel[self.va].tolist(), "applied_torque_nm": self.d.ctrl.tolist()}
        self.rows.append(row)
        self.ended = fallen or self.steps == self.cfg.horizon
        return row

    def assess(self, scenario):
        # Displacement caused by the push, measured against the mean CoM position in the
        # 0.5 s before onset, so a standing offset does not dominate the metric.
        pre = [r for r in self.rows if self.cfg.push_onset_s - .5 <= r["time_s"] < self.cfg.push_onset_s]
        base = (float(np.mean([r["com_x_m"] for r in pre])), float(np.mean([r["com_y_m"] for r in pre]))) if pre else None
        after = [r for r in self.rows if r["time_s"] >= self.cfg.push_onset_s]
        push_peak = max((float(np.hypot(r["com_x_m"] - base[0], r["com_y_m"] - base[1])) for r in after), default=None) if base else None
        peak = max((float(np.hypot(r["com_error_x_m"], r["com_error_y_m"])) for r in self.rows), default=None)
        max_lateral = max((abs(r["com_error_y_m"]) for r in after), default=None)
        tail = self.rows[-25:]
        tail_error = float(np.hypot(abs(np.mean([r["com_error_x_m"] for r in tail])),
                                    abs(np.mean([r["com_error_y_m"] for r in tail]))))
        tail_rate = float(np.hypot(np.mean([r["com_rate_x_m_s"] for r in tail]),
                                   np.mean([r["com_rate_y_m_s"] for r in tail])))
        window = [r for r in self.rows if r["time_s"] >= self.cfg.push_onset_s]
        slips = [s for r in window for s in r["contact_slip_m_s"]]
        recovered = None if self.settled_from is None else self.settled_from - self.cfg.push_onset_s
        complete = self.steps == self.cfg.horizon
        criteria = {"complete": complete,
                    "upright": not any(r["fallen"] for r in self.rows),
                    "min_height": min(r["pelvis_height_m"] for r in self.rows) >= .62,
                    "max_tilt": max(max(abs(r["pitch_rad"]), abs(r["roll_rad"])) for r in self.rows) <= .55,
                    "recovered": self.settled_from is not None,
                    "recovery_time": recovered is not None and recovered <= 2.5,
                    "final_stability": abs(tail[-1]["com_error_x_m"]) <= .035 and abs(tail[-1]["com_error_y_m"]) <= .03
                                        and tail_rate <= .12 and tail_error <= .035,
                    "pushed_by": bool(push_force(scenario, self.cfg.push_onset_s, self.cfg).any())
                                or scenario == "none",
                    "slip": bool(slips) and float(np.mean(slips)) <= .25,
                    "double_support": all(sum(r["foot_contacts"]) == 2 for r in window)}
        return {"success": bool(all(criteria.values())), "criteria": criteria, "steps": self.steps,
                "peak_com_error_m": peak, "push_peak_com_error_m": push_peak, "max_lateral_error_m": max_lateral,
                "pre_push_com_x_m": base[0] if base else None,
                "final_com_error_m": float(np.hypot(tail[-1]["com_error_x_m"], tail[-1]["com_error_y_m"])),
                "tail_com_error_m": tail_error, "tail_com_rate_m_s": tail_rate,
                "settled_com_offset_m": float(tail[-1]["com_error_x_m"] + self.cfg.com_target_m),
                "recovery_time_s": recovered,
                "min_height_m": min(r["pelvis_height_m"] for r in self.rows),
                "max_pitch_rad": max(abs(r["pitch_rad"]) for r in self.rows),
                "max_roll_rad": max(abs(r["roll_rad"]) for r in self.rows),
                "max_abs_ankle_correction_rad": max(abs(r["ankle_correction_rad"]) for r in self.rows),
                "mean_contact_slip_m_s": float(np.mean(slips)) if slips else None,
                "saturation_fraction": sum(r["saturated_substeps"] for r in self.rows) / (self.cfg.substeps * len(self.rows)),
                "lost_double_support_samples": sum(1 for r in window if sum(r["foot_contacts"]) != 2),
                "displacement_m": float(np.hypot(self.d.qpos[0], self.d.qpos[1]))}

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, width=640, height=480)
        camera = lab.camera("h1")
        camera.lookat[:2] = self.d.qpos[:2]
        self.renderer.update_scene(self.d, camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer:
            self.renderer.close()


def episode(env, mode, scenario, seed, render=False, viewer=False):
    env.reset(seed)
    frames = []
    handle = None
    if viewer:
        from mujoco import viewer as mv
        handle = mv.launch_passive(env.m, env.d)
        handle.cam.distance = 3.
        handle.cam.azimuth = 125
        handle.cam.elevation = -18
    try:
        for i in range(env.cfg.horizon):
            tick = time.monotonic()
            row = env.step(mode, scenario)
            if render and i % 5 == 0:
                im = Image.fromarray(env.render())
                draw = ImageDraw.Draw(im)
                draw.text((10, 10), f"{mode} {scenario} t={row['time_s']:.2f}s push={np.linalg.norm(row['push_force_n']):.0f}N", fill="white")
                draw.text((10, 28), f"CoM err x {row['com_error_x_m']*100:+.1f} cm y {row['com_error_y_m']*100:+.1f} cm", fill="white")
                draw.text((10, 46), f"pitch {np.degrees(row['pitch_rad']):+.1f} deg ankle correction {np.degrees(row['ankle_correction_rad']):+.1f} deg", fill="white")
                frames.append(im)
            if handle:
                handle.cam.lookat[:] = [env.d.qpos[0], env.d.qpos[1], .9]
                handle.sync()
                if not handle.is_running():
                    break
                time.sleep(max(0, env.cfg.dt - (time.monotonic() - tick)))
            if env.ended:
                break
        return {"controller": mode, "scenario": scenario, "seed": seed, **env.assess(scenario)}, list(env.rows), frames
    finally:
        if handle:
            handle.close()


def run(out, cfg, seeds, render=True):
    if out.exists():
        raise FileExistsError("new output folder required")
    out.mkdir(parents=True)
    source = out / "source"
    source.mkdir()
    for name in ("h1_recovery.py", "tabular_lab.py"):
        (source / name).write_bytes((ROOT / name).read_bytes())
    (source / "lab.py").write_bytes(LAB_FILE.read_bytes())
    assets = {p.relative_to(lab.CACHE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (lab.CACHE / "unitree_h1").rglob("*") if p.is_file()}
    assert subprocess.check_output(["git", "-C", str(lab.CACHE), "rev-parse", "HEAD"], text=True).strip() == lab.PIN
    protocol = {"date": "2026-10-16", "config": asdict(cfg), "seeds": seeds, "scenarios": list(SCENARIOS),
                "controllers": list(CONTROLLERS), "model_commit": lab.PIN, "model_assets_sha256": assets,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},
                "dependencies": {"numpy": np.__version__, "mujoco": mujoco.__version__},
                "acceptance": {"horizon_steps": 300, "episode_s": 6.0, "push_onset_s": PUSH_ONSET_S,
                               "feedback_start_s": FEEDBACK_START_S, "min_pelvis_height_m": .62,
                               "max_tilt_rad": .55, "com_target_m": cfg.com_target_m,
                               "recovery_window_m": .035, "recovery_rate_m_s": .12,
                               "recovery_time_s": 2.5, "mean_contact_slip_m_s": .25},
                "scenario_push": {k: {"force_x_n": v[0], "force_y_n": v[1], "duration_s": v[2]} for k, v in SCENARIO_PUSH.items()},
                "development_seeds": [0],
                "reset_scope": "Pelvis XY normal .001 m, height normal .0005 m around the calibrated 1.06279 m stance, "
                               "joint noise .001 rad, world velocity .002 m/s for six axes, no pre-settle; "
                               "flat floor, both feet in contact from the first step, push scheduled at 2.0 s",
                "calibration": {"reference_hip": cfg.reference_hip, "reference_knee": cfg.reference_knee,
                                "reference_ankle": cfg.reference_ankle, "reference_pelvis_z": cfg.reference_pelvis_z,
                                "measured_ankle_command_to_com_acceleration": {"value": -2.85, "unit": "1/s^2 per rad",
                                                                               "note": "negative: a negative ankle reference accelerates the CoM forward"},
                                "note": "the shipped h1.xml default pose penetrates the floor by about 0.10 m and is not used"},
                "scope": "Analytic standing/recovery baseline on the pinned H1 model, no RL training, no stepping; "
                         "matched push force, duration, reset seed and torque limits across controllers; "
                         "thresholds frozen before the five held-out seeds"}
    save_json(out / "protocol.json", protocol)
    cases = []
    env = H1Stand(cfg)
    try:
        with gzip.open(out / "trajectories.jsonl.gz", "wt", encoding="utf-8") as stream:
            for mode in CONTROLLERS:
                for scenario in SCENARIOS:
                    for seed in seeds:
                        record, rows, frames = episode(env, mode, scenario, seed,
                                                       render and seed == seeds[0] and scenario in ("strong_front", "push_left")
                                                       and mode in ("fixed", "com"))
                        cases.append(record)
                        stream.write(json.dumps({"case": record, "trajectory": rows}) + "\n")
                        if frames:
                            demos = out / "demos"
                            demos.mkdir(exist_ok=True)
                            name = f"{mode}-{scenario}"
                            frames[0].save(demos / f"{name}.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
                            frames[-1].save(demos / f"{name}.png")
                            save_csv(demos / f"{name}.csv", rows)
                        print(f"{mode} {scenario} seed{seed} success={record['success']} "
                              f"peak={record['peak_com_error_m']:.3f} settled={record['settled_com_offset_m']:+.4f} "
                              f"t_rec={record['recovery_time_s']} fail={[k for k, v in record['criteria'].items() if not v]}", flush=True)
    finally:
        env.close()
    save_json(out / "cases.json", cases)
    aggregate = {mode: {"successes": sum(c["success"] for c in cases if c["controller"] == mode),
                        "cases": sum(c["controller"] == mode for c in cases),
                        "falls": sum(not c["criteria"]["upright"] for c in cases if c["controller"] == mode)}
                 for mode in CONTROLLERS}
    save_json(out / "summary.json", {"aggregate": aggregate, "evaluation_cases": len(cases),
                                     "retained_failure_cases": sum(not c["success"] for c in cases),
                                     "task_status": "Per-scenario physical outcomes retained; RL residual recovery and stepping remain separate tasks"})
    save_json(out / "manifest.json", [{"path": p.relative_to(out).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                      for p in sorted(out.rglob("*")) if p.is_file() and "__pycache__" not in p.parts])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controller", choices=CONTROLLERS, default="com")
    parser.add_argument("--scenario", choices=SCENARIOS, default="strong_front")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3000000, 3000001, 3000002, 3000003, 3000004])
    parser.add_argument("--com-gain", type=float, default=12.)
    parser.add_argument("--com-rate-gain", type=float, default=2.4)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    config = Config(com_gain=args.com_gain, com_rate_gain=args.com_rate_gain)
    if args.demo:
        environment = H1Stand(config)
        try:
            print(json.dumps(episode(environment, args.controller, args.scenario, args.seed, viewer=True)[0], indent=2))
        finally:
            environment.close()
    elif args.output:
        if len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
            parser.error("unique nonnegative seeds")
        run(args.output, config, args.seeds, not args.no_render)
    else:
        parser.error("--output or --demo required")
