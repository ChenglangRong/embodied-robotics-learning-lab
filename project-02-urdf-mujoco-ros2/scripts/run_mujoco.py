"""Load the shared URDF into MuJoCo, add torque actuators, and run a reach task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "project-01-simulation-software" / "src"))
from robot_lab.planar_arm import Config, TwoLinkReachEnv, dls_action


def build_sim_model(destination: Path, goal: tuple[float, float] = (0.55, 0.55)) -> mujoco.MjModel:
    """Convert the teaching URDF to MJCF and add two explicit torque motors."""
    source = ROOT / "model" / "arm2d.urdf"
    # MuJoCo's Windows file-path API may fail on non-ASCII workspace names.
    base_model = mujoco.MjModel.from_xml_string(source.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="robot_lab_mjcf_") as temporary:
        ascii_path = Path(temporary) / "model.xml"
        mujoco.mj_saveLastXML(str(ascii_path), base_model)
        tree = ET.parse(ascii_path)
    root = tree.getroot()
    urdf = ET.parse(source).getroot()
    tool_origin = urdf.find("./joint[@name='tool_fixed']/origin")
    link2_body = root.find(".//body[@name='link2']")
    if tool_origin is None or link2_body is None:
        raise ValueError("URDF and compiled MJCF must contain tool_fixed and link2")
    ET.SubElement(link2_body, "site", {
        "name": "tool_tip", "pos": tool_origin.attrib["xyz"], "size": "0.01"
    })
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("compiled MJCF must contain worldbody")
    ET.SubElement(worldbody, "site", {
        "name": "goal_marker", "pos": f"{goal[0]} {goal[1]} 0",
        "size": "0.025", "rgba": "1 0.1 0.1 1"
    })
    actuators = ET.SubElement(root, "actuator")
    for joint in ("shoulder", "elbow"):
        ET.SubElement(actuators, "motor", {
            "name": f"{joint}_motor", "joint": joint,
            "gear": "1", "ctrllimited": "true", "ctrlrange": "-3 3",
        })
    xml_text = ET.tostring(root, encoding="unicode")
    destination.write_text(xml_text, encoding="utf-8")
    return mujoco.MjModel.from_xml_string(xml_text)


def tip_from_mujoco(data: mujoco.MjData) -> np.ndarray:
    return data.site("tool_tip").xpos.copy()


def run(seed: int, goal: tuple[float, float], output: Path, viewer_enabled: bool, render_gif: bool = False) -> dict:
    model = build_sim_model(output / "derived_scene.xml", goal)
    data = mujoco.MjData(model)
    cfg = Config()
    controller_model = TwoLinkReachEnv(cfg)
    controller_model.reset(seed=seed, goal=goal)
    initial_q = controller_model.q.copy()
    data.qpos[:] = controller_model.q
    mujoco.mj_forward(model, data)
    if model.nq != 2 or model.nv != 2 or model.nu != 2:
        raise RuntimeError(f"expected 2 positions, velocities and motors; got {model.nq}/{model.nv}/{model.nu}")
    physics_dt = float(model.opt.timestep)
    substeps = max(1, round(cfg.dt / physics_dt))
    if not np.isclose(substeps * physics_dt, cfg.dt):
        raise RuntimeError("physics timestep must divide controller timestep exactly")
    rows: list[dict] = []
    termination = "max_steps"
    viewer = None
    renderer = None
    frames: list[Image.Image] = []
    if viewer_enabled:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(model, data)
    if render_gif:
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [0.4, 0.2, 0.0]
        camera.distance = 2.2
        camera.azimuth = 90
        camera.elevation = -89
    try:
        for step in range(cfg.max_steps):
            tick = time.perf_counter()
            controller_model.q = data.qpos.copy()
            desired_qdot = np.clip(dls_action(controller_model), -cfg.velocity_limit, cfg.velocity_limit)
            for _ in range(substeps):
                data.ctrl[:] = np.clip(1.5 * (desired_qdot - data.qvel) - 0.2 * data.qvel, -3.0, 3.0)
                mujoco.mj_step(model, data)
            # mj_step integrates qpos last; refresh body transforms for the logged state.
            mujoco.mj_forward(model, data)
            tip = tip_from_mujoco(data)
            analytic_tip = controller_model.position(data.qpos)
            error = float(np.linalg.norm(np.asarray(goal) - tip[:2]))
            rows.append({
                "step": step + 1, "sim_time_s": float(data.time),
                "qpos1_rad": float(data.qpos[0]), "qpos2_rad": float(data.qpos[1]),
                "qvel1_rad_s": float(data.qvel[0]), "qvel2_rad_s": float(data.qvel[1]),
                "ctrl1_Nm": float(data.ctrl[0]), "ctrl2_Nm": float(data.ctrl[1]),
                "desired_qdot1_rad_s": float(desired_qdot[0]),
                "desired_qdot2_rad_s": float(desired_qdot[1]),
                "tool_x_m": float(tip[0]), "tool_y_m": float(tip[1]),
                "fk_gap_m": float(np.linalg.norm(tip[:2] - analytic_tip)),
                "error_m": error,
            })
            if renderer is not None and step % 2 == 0:
                renderer.update_scene(data, camera=camera)
                frames.append(Image.fromarray(renderer.render().copy()))
            if viewer is not None:
                viewer.sync()
                remaining = cfg.dt - (time.perf_counter() - tick)
                if remaining > 0:
                    time.sleep(remaining)
                if not viewer.is_running():
                    termination = "viewer_closed"
                    break
            if error <= cfg.tolerance:
                termination = "success"
                break
    finally:
        if viewer is not None:
            viewer.close()
        if renderer is not None:
            renderer.close()
    summary = {
        "seed": seed, "initial_q_rad": initial_q.tolist(),
        "goal_m": list(goal), "steps": len(rows),
        "success": termination == "success", "termination": termination,
        "final_error_m": rows[-1]["error_m"],
        "max_fk_gap_m": max(r["fk_gap_m"] for r in rows),
        "physics_dt_s": physics_dt, "control_dt_s": cfg.dt,
        "mujoco_version": mujoco.__version__, "python_version": platform.python_version(),
        "urdf_sha256": hashlib.sha256((ROOT / "model" / "arm2d.urdf").read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps({
        "controller": asdict(cfg),
        "goal_m": list(goal), "seed": seed,
        "control_law": "DLS desired joint velocity; clipped velocity servo torque",
        "motor_torque_limit_Nm": 3.0,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "controller_sha256": hashlib.sha256(
            (ROOT.parent / "project-01-simulation-software" / "src" / "robot_lab" / "planar_arm.py").read_bytes()
        ).hexdigest(),
    }, indent=2), encoding="utf-8")
    if frames:
        frames[0].save(output / "animation.gif", save_all=True,
                       append_images=frames[1:], duration=80, loop=0)
        frames[-1].save(output / "final_frame.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--goal-x", type=float, default=0.55)
    parser.add_argument("--goal-y", type=float, default=0.55)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "first-run")
    parser.add_argument("--viewer", action="store_true", help="show live MuJoCo GUI")
    parser.add_argument("--render-gif", action="store_true", help="save an offscreen MuJoCo animation")
    args = parser.parse_args()
    print(json.dumps(run(args.seed, (args.goal_x, args.goal_y), args.output,
                         args.viewer, args.render_gif), indent=2))


if __name__ == "__main__":
    main()
