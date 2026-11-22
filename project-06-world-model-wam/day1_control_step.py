"""Replay three one-step controls under the same two-link MuJoCo reset."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import mujoco
import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT.parent / "project-05-reinforcement-safety"))
import physical_env

RESET_SEED = 4000
ACTIONS = {"positive": [0.5, 0.0], "negative": [-0.5, 0.0], "zero": [0.0, 0.0]}


def snapshot(env, observation):
    return {
        "time_s": float(env.d.time),
        "q_rad": env.d.qpos.tolist(),
        "dq_rad_s": env.d.qvel.tolist(),
        "observation": observation.tolist(),
    }


def run():
    env = physical_env.ArmPhysics(split="test")
    try:
        obs0, reset_info = env.reset(RESET_SEED)
        initial = snapshot(env, obs0)
        cases = {}
        for name, action in ACTIONS.items():
            obs, _ = env.reset(RESET_SEED)
            if snapshot(env, obs) != initial:
                raise RuntimeError(f"{name} did not start from the frozen state")
            next_obs, reward, terminated, truncated, info = env.step(action)
            final = snapshot(env, next_obs)
            if not np.isclose(final["time_s"] - initial["time_s"], env.cfg.dt):
                raise RuntimeError(f"{name} did not advance one control step")
            cases[name] = {
                "action": action,
                "velocity_ref_rad_s": info["raw_velocity"],
                "applied_velocity_ref_rad_s": info["applied_velocity"],
                "final": final,
                "reward": float(reward),
                "error_m": info["error_m"],
                "torque_peak_nm": info["torque_peak_nm"],
                "guard_intervention": info["guard_intervention"],
                "success": info["success"],
                "violation": info["violation"],
                "terminated": terminated,
                "truncated": truncated,
            }
        return {
            "reset_seed": RESET_SEED,
            "data_scope": "Development demonstration, no training or held-out evaluation",
            "environment_split": env.split,
            "config": asdict(env.cfg),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "mujoco": mujoco.__version__,
            },
            "sha256": {
                "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "environment": hashlib.sha256(Path(physical_env.__file__).read_bytes()).hexdigest(),
                "model_xml": hashlib.sha256(physical_env.MODEL.read_bytes()).hexdigest(),
            },
            "goal_m": reset_info["goal"],
            "initial": initial,
            "cases": cases,
        }
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run(), indent=2, allow_nan=False)
    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8", newline="\n")
