"""Physical Allegro grasp experiments with a fixed wrist during evaluation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
LAB_FILE = ROOT / "lab.py" if (ROOT / "lab.py").is_file() else ROOT.parent / "project-03-robot-platforms/lab.py"
spec = importlib.util.spec_from_file_location("allegro_nominal_lab", LAB_FILE)
lab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lab)

SUBSTEPS = 10
DT = .02
HORIZON = 200
PULSE = slice(50, 65)
THUMB_PROXIMAL = 13
SLIP_GAIN = 28.
SLIP_DEADBAND_M = .001
MAX_CORRECTION_RAD = .25
ROLL_DELTA_RAD = -.4
ROLL_AXIS = (-.25, -.49, .86)
ROLL_TARGET_DEG = 20.
TASKS = {"no_load": 0., "pull9": 9., "pull13": 13., "stress17": 17., "reorient": 0.}
SLIP_MODES = ("hold", "feedback")
ROLL_MODES = ("hold", "roll", "reverse")


def rotation_error_deg(actual, target):
    cosine = (np.trace(target.T @ actual) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def axis_angle(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    quat = np.zeros(4)
    matrix = np.zeros(9)
    mujoco.mju_axisAngle2Quat(quat, axis / np.linalg.norm(axis), np.radians(degrees))
    mujoco.mju_quat2Mat(matrix, quat)
    return matrix.reshape(3, 3)


def slip_correction(drift_m, gain):
    return -float(np.clip(gain * (drift_m - SLIP_DEADBAND_M), 0, MAX_CORRECTION_RAD))


class Hand:
    def __init__(self):
        self.m, self.d, self.qa, self.va, _ = lab.build("allegro")
        self.palm = self.m.body("palm").id
        self.obj = self.m.body("object").id
        self.geom = self.m.geom("object_geom").id
        self.tray = self.m.geom("tray_geom").id
        self.floor = self.m.geom("floor").id
        self.wrist = int(self.m.body("palm").mocapid[0])
        thumb_joint = self.m.joint(self.m.actuator_trnid[THUMB_PROXIMAL, 0]).name
        if thumb_joint != "thj1":
            raise ValueError(f"Unexpected thumb actuator mapping: {thumb_joint}")
        joint = self.m.joint("object_free").id
        self.object_q = int(self.m.jnt_qposadr[joint])
        self.initial_q = self.d.qpos.copy()
        self.nominal = lab.Controller("allegro", self.m, self.d, self.qa, self.va)
        self.closed = np.asarray(lab.DEFAULT_OPTIONS["allegro"]["closed"])
        self.renderer = None

    def reset(self, seed):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.initial_q
        rng = np.random.default_rng(seed)
        self.d.qpos[self.object_q:self.object_q+2] += rng.uniform(-.001, .001, 2)
        self.d.qpos[self.qa] += rng.uniform(-.004, .004, 16)
        mujoco.mj_forward(self.m, self.d)
        # This common preparation uses the existing prescribed lift and tray withdrawal.
        for _ in range(3500):
            self.nominal.step()
            mujoco.mj_step(self.m, self.d)
        self.d.mocap_pos[self.wrist, 2] = .31
        self.d.ctrl[:] = self.closed
        for _ in range(1000):
            mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self.start_position, self.start_rotation = self.pose()

    def pose(self):
        wrist_frame = self.d.xmat[self.palm].reshape(3, 3).T
        return (wrist_frame @ (self.d.xpos[self.obj] - self.d.xpos[self.palm]),
                wrist_frame @ self.d.xmat[self.obj].reshape(3, 3))

    def contacts(self):
        normals = dict.fromkeys(("ff", "mf", "rf", "th", "palm"), 0.)
        supported = False
        friction = []
        wrench = np.zeros(6)
        for i, c in enumerate(self.d.contact):
            if self.geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.geom else c.geom1
            mujoco.mj_contactForce(self.m, self.d, i, wrench)
            if wrench[0] <= .01:
                continue
            if other in (self.tray, self.floor):
                supported = True
            name = self.m.body(self.m.geom_bodyid[other]).name
            group = "palm" if name == "palm" else name[:2]
            if group in normals:
                normals[group] += float(wrench[0])
                friction.append(float(c.friction[0]))
        return normals, supported, friction

    def advance(self, action, force):
        self.d.ctrl[:] = np.clip(action, *self.m.actuator_ctrlrange.T)
        for _ in range(SUBSTEPS):
            self.d.xfrc_applied[self.obj] = [*force, 0., 0., 0.]
            mujoco.mj_step(self.m, self.d)
        # Evaluate contacts with the same external load as the integrated interval.
        mujoco.mj_forward(self.m, self.d)

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.m, width=640, height=480)
        camera = lab.camera("allegro")
        camera.lookat[:] = [.025, 0, .33]
        camera.distance = .48
        camera.elevation = -30
        self.renderer.update_scene(self.d, camera=camera)
        # Nonphysical colored markers expose the orientation of the symmetric ball.
        mark_pose(self.renderer.scene, self.d.xpos[self.obj], self.d.xmat[self.obj])
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def episode(hand, task, mode, seed, gain=SLIP_GAIN, on_step=None):
    modes = ROLL_MODES if task == "reorient" else SLIP_MODES
    if task not in TASKS or mode not in modes or not np.isfinite(gain) or gain < 0:
        raise ValueError("invalid task, controller or gain")
    hand.reset(seed)
    target = axis_angle(ROLL_AXIS, ROLL_TARGET_DEG) @ hand.start_rotation if task == "reorient" else hand.start_rotation
    wrist_p = hand.d.xpos[hand.palm].copy()
    wrist_r = hand.d.xmat[hand.palm].copy()
    anchor = None
    rows = []
    for step in range(HORIZON):
        position, _ = hand.pose()
        if step == PULSE.start:
            anchor = position.copy()
        drift = 0. if anchor is None else float(anchor[1] - position[1])
        correction = 0.
        if mode == "feedback" and PULSE.start <= step < 80:
            correction = slip_correction(drift, gain)
        if mode in ("roll", "reverse"):
            direction = 1 if mode == "roll" else -1
            correction = direction * ROLL_DELTA_RAD * float(lab.smooth((step - 35) / 35))
        action = hand.closed.copy()
        action[THUMB_PROXIMAL] += correction
        force = [0., TASKS[task] if PULSE.start <= step < PULSE.stop else 0., 0.]
        hand.advance(action, force)
        position, rotation = hand.pose()
        normals, supported, friction = hand.contacts()
        row = {"step": step + 1, "time_s": (step+1)*DT, "force_world_n": hand.d.xfrc_applied[hand.obj, :3].tolist(),
               "external_torque_world_nm": hand.d.xfrc_applied[hand.obj, 3:].tolist(),
               "palm_relative_position_m": position.tolist(), "palm_relative_rotation": rotation.ravel().tolist(),
               "world_height_m": float(hand.d.xpos[hand.obj, 2]),
               "shift_m": float(np.linalg.norm(position - hand.start_position)),
               "drift_y_m": 0. if anchor is None else float(anchor[1] - position[1]),
               "target_error_deg": rotation_error_deg(rotation, target),
               "rotation_from_start_deg": rotation_error_deg(rotation, hand.start_rotation),
               "wrist_shift_m": float(np.linalg.norm(hand.d.xpos[hand.palm] - wrist_p)),
               "wrist_rotation_matrix_change": float(np.linalg.norm(hand.d.xmat[hand.palm] - wrist_r)),
               "finger_count": sum(normals[name] > .03 for name in ("ff", "mf", "rf", "th")),
               "normal_force_n": normals, "support_contact": supported, "contact_mu": friction,
               "feedback_input_m": drift, "thumb_correction_rad": correction,
               "q_ref_rad": hand.d.ctrl.tolist(), "q_rad": hand.d.qpos[hand.qa].tolist(),
               "qvel_rad_s": hand.d.qvel[hand.va].tolist(), "actuator_force_nm": hand.d.actuator_force.tolist()}
        rows.append(row)
        if on_step is not None and not on_step(hand, row):
            break
    return assess(task, mode, seed, rows), rows


def assess(task, mode, seed, rows):
    tail = rows[-25:]
    peak_drift = max(abs(row["drift_y_m"]) for row in rows)
    peak_shift = max(row["shift_m"] for row in rows)
    final_error = max(row["target_error_deg"] for row in tail)
    active = sum(any(row["force_world_n"]) for row in rows)
    impulse = DT * sum(row["force_world_n"][1] for row in rows)
    schedule_correct = all(row["force_world_n"] == [0., TASKS[task] if i in range(PULSE.start, PULSE.stop) else 0., 0.]
                           for i, row in enumerate(rows))
    two_finger_fraction = sum(row["finger_count"] >= 2 for row in rows)/len(rows)
    criteria = {
        "complete": len(rows) == HORIZON,
        "no_external_support": not any(row["support_contact"] for row in rows),
        "fixed_wrist": all(row["wrist_shift_m"] < 1e-10 and row["wrist_rotation_matrix_change"] < 1e-10 for row in rows),
        "retained": peak_shift <= .04 and all(.25 <= row["world_height_m"] <= .46 for row in rows),
        "multi_finger": two_finger_fraction >= .95 and all(row["finger_count"] >= 2 for row in tail),
        "force_schedule": schedule_correct and active == (15 if TASKS[task] else 0),
        "zero_external_torque": all(not any(row["external_torque_world_nm"]) for row in rows),
    }
    if task == "reorient":
        criteria["target_held"] = final_error <= 8 and min(row["rotation_from_start_deg"] for row in tail) >= 12
    else:
        criteria["limited_drift"] = peak_drift <= .014 and max(abs(row["drift_y_m"]) for row in tail) <= .010
    return {"task": task, "mode": mode, "seed": seed, "success": all(criteria.values()), "criteria": criteria,
            "steps": len(rows), "peak_drift_m": peak_drift, "peak_shift_m": peak_shift,
            "tail_max_goal_error_deg": final_error, "final_rotation_deg": rows[-1]["rotation_from_start_deg"],
            "final_drift_m": rows[-1]["drift_y_m"], "two_finger_fraction": two_finger_fraction,
            "peak_normal_force_n": max(sum(row["normal_force_n"].values()) for row in rows),
            "mean_normal_force_n": float(np.mean([sum(row["normal_force_n"].values()) for row in rows])),
            "load_steps": active, "impulse_y_n_s": impulse}


def mark_pose(scene, center, orientation):
    rotation = orientation.reshape(3, 3)
    for axis, rgba in enumerate(((.95, .1, .14, 1), (.09, .84, .37, 1), (.12, .56, 1., 1))):
        for sign in (1, -1):
            pos = center + sign * .039 * rotation[:, axis]
            size = .007 if sign > 0 else .004
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.full(3, size), pos, np.eye(3).ravel(), np.array(rgba, dtype=np.float32))
            scene.ngeom += 1
