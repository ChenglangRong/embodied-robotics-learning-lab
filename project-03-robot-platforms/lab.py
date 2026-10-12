"""Four physical robot experiments. Execution changes controls, never teleports robots."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from PIL import Image, ImageDraw

PIN = "0059d4335f8156206f63a35662313385f7ad6d74"
CACHE = Path(os.environ.get("ROBOT_MODEL_CACHE", str(Path.home() / ".cache/robot-learning/menagerie")))
SOURCES = {"panda": ("franka_emika_panda", "panda.xml"), "go2": ("unitree_go2", "go2.xml"),
           "h1": ("unitree_h1", "h1.xml"), "allegro": ("wonik_allegro", "right_hand.xml")}
DEFAULT_OPTIONS = {
    "panda": {"close_ctrl": 0},
    "go2": {"amp": .055, "lift": .07, "freq": 1.8, "kp": 100, "kd": 3.5},
    "h1": {"com_gain": 5., "velocity_gain": .8, "foot_offset": .03, "balance": 0., "bend": .15},
    "allegro": {"wrist_quat": "0 1 0 1", "object_pos": [.03, 0, .215], "radius": .04, "wrist_tilt": 0.,
                "closed": [-.2, .6, .6, .6, 0, .6, .6, .6, .2, .6, .6, .6, 1.1, .6, .9, .7]},
}


def add(parent, tag, **kwargs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in kwargs.items()})


def build(kind, options=None):
    options = options or {}
    folder, filename = SOURCES[kind]
    src = CACHE / folder / filename
    root = ET.fromstring(src.read_text(encoding="utf-8"))
    root.find("compiler").set("meshdir", (src.parent / "assets").as_posix())
    # MJCF home keys have fixed dimensions; reset the original robot explicitly below.
    key = root.find("keyframe")
    if key is not None:
        root.remove(key)
    option = root.find("option")
    if option is None:
        option = add(root, "option")
    option.set("timestep", "0.002")
    option.set("integrator", "implicitfast")
    visual = add(root, "visual")
    add(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.4 0.4 0.4", specular="0.2 0.2 0.2")
    add(visual, "global", offwidth="640", offheight="480")
    asset = root.find("asset")
    add(asset, "texture", name="lab_checker", type="2d", builtin="checker", rgb1=".16 .21 .27", rgb2=".24 .30 .36", width="256", height="256")
    add(asset, "material", name="lab_floor", texture="lab_checker", texrepeat="4 4", reflectance=".1")
    world = root.find("worldbody")
    add(world, "light", pos="1 -2 3", dir="-.2 .5 -1", diffuse=".8 .8 .8")
    add(world, "geom", name="floor", type="plane", size="4 4 .1", material="lab_floor", friction="1 .01 .001")
    if kind == "panda":
        hand = root.find(".//body[@name='hand']")
        add(hand, "site", name="tcp", pos="0 0 .1034", size=".008", rgba="0 1 0 1")
        add(world, "geom", name="table", type="box", pos=".5 0 .125", size=".24 .36 .125", rgba=".5 .57 .65 1", friction="1 .02 .001")
        obj = add(world, "body", name="object", pos=".5 .12 .276")
        add(obj, "freejoint", name="object_free")
        add(obj, "geom", name="object_geom", type="box", size=".025 .025 .025", mass=".06", rgba="1 .42 .12 1", friction="1 .02 .002", condim="6")
        add(world, "site", name="goal", type="box", pos=".5 -.19 .252", size=".055 .055 .002", rgba=".2 .9 .45 .6")
        # Finger-pad contacts need lateral friction to support a lifted object.
        for geom in hand.iter("geom"):
            geom.set("friction", "1 .02 .002")
            geom.set("condim", "4")
    if kind == "allegro":
        palm = root.find(".//body[@name='palm']")
        palm.set("quat", options.get("wrist_quat", "0 1 0 1"))
        palm.set("pos", "0 0 .15")
        palm.set("mocap", "true")
        # A prescribed wrist fixture, with dynamically simulated finger joints and object.
        objxyz = options.get("object_pos", [.03, 0, .215])
        radius = options.get("radius", .04)
        tray = add(world, "body", name="tray", mocap="true", pos=f"{objxyz[0]} {objxyz[1]} {objxyz[2] - radius - .006}")
        add(tray, "geom", name="tray_geom", type="box", size=".014 .014 .006", rgba=".45 .55 .7 .5")
        obj = add(world, "body", name="object", pos=" ".join(map(str, objxyz)))
        add(obj, "freejoint", name="object_free")
        add(obj, "geom", name="object_geom", type="sphere", size=radius, mass=".04", rgba="1 .52 .12 1", friction="1.5 .02 .002", condim="6")
        for geom in palm.iter("geom"):
            geom.set("friction", "1.5 .02 .002")
            geom.set("condim", "4")
        default = root.find("default")
        for pos in default.iter("position"):
            pos.set("kp", "3")
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    qadr = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    vadr = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    if kind == "panda":
        qadr = np.arange(8)
        vadr = np.arange(8)
        data.qpos[:9] = [0, 0, 0, -1.57079, 0, 1.57079, -.7853, .04, .04]
        data.ctrl[:] = [0, 0, 0, -1.57079, 0, 1.57079, -.7853, 255]
    elif kind == "go2":
        data.qpos[:7] = [0, 0, .30, 1, 0, 0, 0]
        data.qpos[qadr] = np.tile([0, .9, -1.8], 4)
    elif kind == "h1":
        data.qpos[:7] = [0, 0, .98, 1, 0, 0, 0]
        data.qpos[qadr] = [0, 0, -.4, .8, -.4] * 2 + [0] * 9
    else:
        data.qpos[qadr] = [0, .15, .15, .15] * 3 + [.6, .2, .2, .2]
        data.ctrl[:] = data.qpos[qadr]
    mujoco.mj_forward(model, data)
    return model, data, qadr, vadr, hashlib.sha256(xml.encode()).hexdigest()


def smooth(a):
    a = np.clip(a, 0, 1)
    return a * a * (3 - 2 * a)


class Controller:
    def __init__(self, kind, model, data, qadr, vadr, options=None):
        self.kind, self.m, self.d, self.qa, self.va = kind, model, data, qadr, vadr
        self.options = options or {}
        self.home = data.qpos[qadr].copy()
        self.target = self.home.copy()
        self.plan = mujoco.MjData(model)
        self.plan.qpos[:] = data.qpos
        self.stage = "settle"
        self.wrist_home = data.mocap_quat.copy()
        if kind == "panda":
            self.tcp = model.site("tcp").id
            self.plan.qpos[:7] = [0, -.4, 0, -2, 0, 1.7, .7]
            self.waypoints = [(0, np.array([.5, .12, .48]), 255), (2, np.array([.5, .12, .285]), 255),
                              (4, np.array([.5, .12, .285]), 0), (5.5, np.array([.5, .12, .48]), 0),
                              (7, np.array([.5, -.19, .48]), 0), (9, np.array([.5, -.19, .285]), 0),
                              (11, np.array([.5, -.19, .285]), 255), (12, np.array([.5, -.19, .48]), 255)]
            self.initial_tcp = data.site_xpos[self.tcp].copy()

    def ik(self, xyz):
        # Numerical 6D IK in planning-only data; physical data evolves through mj_step.
        desired = np.diag([1., -1., -1.])
        for _ in range(12):
            mujoco.mj_forward(self.m, self.plan)
            rot = self.plan.site_xmat[self.tcp].reshape(3, 3)
            er = .5 * sum((np.cross(rot[:, i], desired[:, i]) for i in range(3)))
            error = np.r_[xyz - self.plan.site_xpos[self.tcp], er]
            jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
            mujoco.mj_jacSite(self.m, self.plan, jp, jr, self.tcp)
            J = np.vstack([jp[:, :7], jr[:, :7]])
            delta = J.T @ np.linalg.solve(J @ J.T + .002 * np.eye(6), error)
            self.plan.qpos[:7] += np.clip(delta, -.12, .12)
            self.plan.qpos[:7] = np.clip(self.plan.qpos[:7], self.m.jnt_range[:7, 0] + .01, self.m.jnt_range[:7, 1] - .01)
        return self.plan.qpos[:7].copy()

    def step(self):
        t = self.d.time
        if self.kind == "panda":
            idx = max(i for i, (ts, _, _) in enumerate(self.waypoints) if ts <= t)
            ts, xyz, gripper = self.waypoints[idx]
            prev = self.initial_tcp if idx == 0 else self.waypoints[idx - 1][1]
            duration = 1.6 if idx in [0, 1, 3, 4, 5, 7] else .7
            xyz = prev + smooth((t - ts) / duration) * (xyz - prev)
            self.target[:7] = self.ik(xyz)
            self.d.ctrl[:7] = self.target[:7]
            self.d.ctrl[7] = self.options.get("close_ctrl", 0) if gripper == 0 else gripper
            self.stage = ["approach", "descend", "grasp", "lift", "transport", "place", "release", "retreat"][idx]
        elif self.kind == "go2":
            q = self.home.copy()
            amp = self.options.get("amp", .055)
            freq = self.options.get("freq", 1.8)
            phase = 2 * np.pi * freq * max(0, t - 2)
            ramp = smooth((t - 2) / 1.5) * (1 - smooth((t - 9) / 1.5))
            for i, offset in enumerate([0, np.pi, np.pi, 0]):
                p = phase + offset
                x = ramp * amp * np.sin(p)
                z = -.265 + ramp * self.options.get("lift", .07) * max(0, np.cos(p))
                knee = -math.acos(np.clip((x * x + z * z - 2 * .213**2) / (2 * .213**2), -1, 1))
                q[3 * i + 1] = math.atan2(-x, -z) - knee / 2
                q[3 * i + 2] = knee
            self.target = q
            self.d.ctrl[:] = np.clip(self.options.get("kp", 100) * (q - self.d.qpos[self.qa]) - self.options.get("kd", 3.5) * self.d.qvel[self.va], self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])
            self.stage = "stand" if t < 2 or t > 10.5 else "diagonal gait"
        elif self.kind == "h1":
            q = self.home.copy()
            bend = self.options.get("bend", .15) * np.sin(np.pi * np.clip((t - 2) / 6, 0, 1)) ** 2
            for off in [0, 5]:
                q[off + 2] -= bend
                q[off + 3] += 2 * bend
                q[off + 4] -= bend
            arm = .3 * np.sin(np.pi * np.clip((t - 3) / 5, 0, 1)) ** 2
            q[11] = -arm
            q[15] = -arm
            q[14] = .3 * arm
            q[18] = .3 * arm
            # PD tracking plus bias compensation, with measured torso pitch feedback.
            R = self.d.xmat[self.m.body("pelvis").id].reshape(3, 3)
            pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
            feet_x = .5 * (self.d.xpos[self.m.body("left_ankle_link").id, 0] + self.d.xpos[self.m.body("right_ankle_link").id, 0])
            com_x = self.d.subtree_com[self.m.body("pelvis").id, 0]
            correction = self.options.get("com_gain", 5.) * (com_x - feet_x - self.options.get("foot_offset", .03)) + self.options.get("velocity_gain", .8) * self.d.qvel[0]
            for off in [0, 5]:
                q[off + 4] += self.options.get("balance", .0) * pitch + correction
            kp = np.array([160, 240, 300, 300, 100] * 2 + [180] + [60] * 8)
            kd = 2 * np.sqrt(kp) * .6
            self.target = q
            torque = kp * (q - self.d.qpos[self.qa]) - kd * self.d.qvel[self.va] + self.d.qfrc_bias[self.va]
            self.d.ctrl[:] = np.clip(torque, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])
            self.stage = "balance" if t < 2 or t > 8 else "squat + reach"
        else:
            closed = np.array(self.options.get("closed", [-.2, .6, .6, .6, 0, .6, .6, .6, .2, .6, .6, .6, 1.1, .6, .9, .7]))
            self.target = self.home + smooth((t - 1) / 2) * (closed - self.home)
            self.d.ctrl[:] = self.target
            lift = .16 * smooth((t - 4) / 3)
            self.d.mocap_pos[self.m.body("palm").mocapid, 2] = .15 + lift
            support_z = self.options.get("object_pos", [.03, 0, .215])[2] - self.options.get("radius", .04) - .006
            self.d.mocap_pos[self.m.body("tray").mocapid, 2] = support_z - .3 * smooth((t - 3.5) / 1)
            angle = self.options.get("wrist_tilt", 0.) * smooth((t - 7) / 1)
            wrist_id = int(self.m.body("palm").mocapid[0])
            mujoco.mju_mulQuat(self.d.mocap_quat[wrist_id], np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0]), self.wrist_home[wrist_id])
            self.stage = "close" if t < 3.5 else "remove support + lift"


def camera(kind):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = 125, -22
    if kind == "panda":
        cam.lookat[:], cam.distance = [.35, 0, .4], 1.7
    elif kind == "go2":
        cam.lookat[:], cam.distance = [0, 0, .24], 1.7
    elif kind == "h1":
        cam.lookat[:], cam.distance = [0, 0, .8], 3
    else:
        cam.lookat[:], cam.distance = [.02, 0, .24], .7
        cam.azimuth = 145
    return cam


def sample(kind, m, d, controller):
    finger_groups = set()
    contact_force = 0.
    foot_contact = [0] * 4
    foot_clearance = [0.] * 4
    for i, c in enumerate(d.contact):
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        contact_force += abs(force[0])
        for g in [c.geom1, c.geom2]:
            b = m.geom_bodyid[g]
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if kind == "allegro" and m.geom("object_geom").id in [c.geom1, c.geom2]:
                if name[:3] in ["ff_", "mf_", "rf_", "th_"]:
                    finger_groups.add(name[:2])
    if kind == "go2":
        floor = m.geom("floor").id
        for i, name in enumerate(["FL", "FR", "RL", "RR"]):
            gid = m.geom(name).id
            foot_contact[i] = int(any(gid in [c.geom1, c.geom2] and floor in [c.geom1, c.geom2] for c in d.contact))
            foot_clearance[i] = float(d.geom_xpos[gid, 2] - m.geom_size[gid, 0])
    if kind in ["panda", "allegro"]:
        pos = d.xpos[m.body("object").id]
        height = pos[2]
        tilt = 0
        object_geom = m.geom("object_geom").id
        contacts = sum(object_geom in [c.geom1, c.geom2] and m.geom_bodyid[c.geom1 if c.geom2 == object_geom else c.geom2] not in [0, m.body("object").id] and "tray" not in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[c.geom1 if c.geom2 == object_geom else c.geom2]) or "") for c in d.contact)
    else:
        pos = d.qpos[:3]
        height = pos[2]
        R = d.xmat[m.body("base" if kind == "go2" else "pelvis").id].reshape(3, 3)
        tilt = math.acos(float(np.clip(R[2, 2], -1, 1)))
        floor = m.geom("floor").id
        contacts = sum(floor in [c.geom1, c.geom2] for c in d.contact)
    error = np.sqrt(np.mean((controller.target - d.qpos[controller.qa]) ** 2))
    # Panda's final actuator is a tendon, so its control units differ from joint angles.
    return [d.time, *pos, height, tilt, contacts, float(error), controller.stage,
            len(finger_groups), contact_force, *foot_contact, *foot_clearance,
            *d.qpos[controller.qa], *d.qvel[controller.va], *d.ctrl, *controller.target]


def assess(kind, rows):
    a = np.array([r[:8] for r in rows], dtype=float)
    t, z, tilt, contact = a[:, 0], a[:, 4], a[:, 5], a[:, 6]
    final = a[t > t[-1] - 1]
    metrics = {"max_height_m": float(z.max()), "final_height_m": float(final[:, 4].mean()),
               "max_tilt_deg": float(np.rad2deg(tilt.max())), "contact_samples": int((contact > 0).sum())}
    if kind == "panda":
        err = np.linalg.norm(final[:, 1:3].mean(axis=0) - [.5, -.19])
        metrics.update(placement_error_m=float(err), lift_m=float(z.max() - z[0]))
        passed = z.max() > .38 and err < .035 and abs(final[:, 4].mean() - .275) < .015
    elif kind == "go2":
        displacement = a[-1, 1] - a[0, 1]
        metrics.update(forward_displacement_m=float(displacement), min_base_height_m=float(z[t > 1].min()))
        clearances = np.array([r[15:19] for r in rows])
        metrics.update(max_foot_clearance_m=float(clearances[t > 2].max()),
                       swing_samples=int((clearances[t > 2].max(axis=1) > .02).sum()))
        passed = displacement > .08 and z[t > 1].min() > .20 and tilt.max() < .6 and metrics["swing_samples"] > 10
    elif kind == "h1":
        dip = float(np.max(z[(t > 1) & (t < 2)]) - np.min(z[(t > 4) & (t < 6)]))
        metrics.update(squat_depth_m=dip, min_pelvis_height_m=float(z.min()))
        passed = z.min() > .7 and tilt.max() < .25 and dip > .025
    else:
        fingers = np.array([r[9] for r in rows])
        held = (t > 7) & (contact >= 2) & (fingers >= 2) & (z > .27)
        metrics.update(held_fraction_after_lift=float(held[t > 7].mean()),
                       max_contacting_fingers=int(fingers[t > 7].max()),
                       support="palm-assisted caging with prescribed wrist fixture")
        passed = held[t > 7].mean() > .9
    return {"task": kind, "passed": bool(passed), "metrics": metrics, "menagerie_commit": PIN,
            "mujoco_version": mujoco.__version__, "scope": "deterministic physical baseline; no learned policy; no real robot claim"}


def run(kind, output=None, render=False, viewer=False, options=None, duration=None, observer=None):
    options = {**DEFAULT_OPTIONS[kind], **(options or {})}
    m, d, qa, va, model_hash = build(kind, options)
    ctrl = Controller(kind, m, d, qa, va, options)
    end = duration or {"panda": 14, "go2": 12, "h1": 10, "allegro": 11}[kind]
    frames, rows = [], []
    renderer = mujoco.Renderer(m, height=480, width=640) if render else None
    cam = camera(kind)
    view = None
    if viewer:
        from mujoco import viewer as mjviewer
        view = mjviewer.launch_passive(m, d)
        view.cam.lookat[:], view.cam.distance = cam.lookat, cam.distance
        view.cam.azimuth, view.cam.elevation = cam.azimuth, cam.elevation
    next_frame, step = 0., 0
    started = time.perf_counter()
    while d.time < end:
        if step % 5 == 0:
            ctrl.step()
        mujoco.mj_step(m, d)
        if step % 10 == 0:
            rows.append(sample(kind, m, d, ctrl))
            if observer is not None:
                observer(m, d, ctrl)
        if renderer and d.time >= next_frame:
            if kind == "go2":
                cam.lookat[0:2] = d.qpos[:2]
            renderer.update_scene(d, camera=cam)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 640, 35), fill=(14, 22, 33))
            draw.text((12, 10), f"{kind.upper()} | t={d.time:4.1f}s | {ctrl.stage}", fill="white")
            frames.append(frame)
            next_frame += .1
        if view:
            if not view.is_running():
                break
            if step % 10 == 0:
                if kind == "go2":
                    view.cam.lookat[0:2] = d.qpos[:2]
                view.sync()
            remaining = started + d.time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
        step += 1
    if renderer:
        renderer.close()
    if view:
        deadline = time.perf_counter() + 15
        while view.is_running() and time.perf_counter() < deadline:
            view.sync()
            time.sleep(.02)
        view.close()
    result = assess(kind, rows) if d.time >= end - m.opt.timestep else {"task": kind, "passed": False, "metrics": {}, "menagerie_commit": PIN}
    result["completed_duration_s"] = float(d.time)
    if d.time < end - m.opt.timestep:
        result["passed"] = False
        result["termination"] = "viewer closed before task completion"
    else:
        result["termination"] = "completed" if result["passed"] else "task criteria failed"
    if output:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "x_m", "y_m", "z_m", "height_m", "tilt_rad", "contacts", "joint_rmse_rad", "stage",
                             "contacting_fingers", "total_normal_contact_force_N", *[f"foot_{x}_contact" for x in ["FL", "FR", "RL", "RR"]],
                             *[f"foot_{x}_clearance_m" for x in ["FL", "FR", "RL", "RR"]],
                             *[f"joint_{i}_q" for i in range(len(qa))], *[f"joint_{i}_dq" for i in range(len(va))],
                             *[f"actuator_{i}_ctrl" for i in range(m.nu)], *[f"joint_{i}_target" for i in range(len(qa))]])
            writer.writerows(rows)
        (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        config = {"task": kind, "duration_s": end, "dt_s": .002, "controller_dt_s": .01, "options": options or {},
                  "model_xml_sha256": model_hash, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "menagerie_commit": PIN, "seed": 0}
        model_source = CACHE / SOURCES[kind][0] / SOURCES[kind][1]
        config["upstream_model_xml_sha256"] = hashlib.sha256(model_source.read_bytes()).hexdigest()
        config["joint_names"] = [m.joint(int(m.actuator_trnid[i, 0])).name if kind != "panda" or i < 7 else "finger_joint1 (tendon)" for i in range(m.nu)]
        config["actuator_names"] = [m.actuator(i).name for i in range(m.nu)]
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if frames:
            frames[0].save(out / "demo.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
            frames[len(frames) // 2].save(out / "preview.png")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("robot", choices=list(SOURCES))
    ap.add_argument("--output", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--options", default="{}", help="JSON controller parameters")
    args = ap.parse_args()
    print(json.dumps(run(args.robot, args.output, args.render, args.viewer, json.loads(args.options)), indent=2))
