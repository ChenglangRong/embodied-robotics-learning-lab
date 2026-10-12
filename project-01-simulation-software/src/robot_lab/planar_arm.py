from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    link_lengths: tuple[float, float] = (0.6, 0.4)
    dt: float = 0.02
    max_steps: int = 600
    tolerance: float = 1e-3
    gain: float = 4.0
    damping: float = 0.05
    velocity_limit: float = 2.0


class TwoLinkReachEnv:
    def __init__(self, config: Config = Config()):
        self.cfg = config
        self.q = np.zeros(2, dtype=float)
        self.goal = np.zeros(2, dtype=float)
        self.step_count = 0

    def reset(self, seed: int = 0, goal: tuple[float, float] = (0.55, 0.55)) -> dict:
        rng = np.random.default_rng(seed)
        self.q = rng.uniform(-0.5, 0.5, size=2)
        self.goal = np.asarray(goal, dtype=float)
        self.step_count = 0
        return self.observe()

    def position(self, q: np.ndarray | None = None) -> np.ndarray:
        q = self.q if q is None else np.asarray(q, dtype=float)
        l1, l2 = self.cfg.link_lengths
        return np.array([
            l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1]),
            l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1]),
        ])

    def jacobian(self, q: np.ndarray | None = None) -> np.ndarray:
        q = self.q if q is None else np.asarray(q, dtype=float)
        l1, l2 = self.cfg.link_lengths
        s12, c12 = np.sin(q.sum()), np.cos(q.sum())
        return np.array([
            [-l1 * np.sin(q[0]) - l2 * s12, -l2 * s12],
            [l1 * np.cos(q[0]) + l2 * c12, l2 * c12],
        ])

    def observe(self) -> dict:
        return {"q": self.q.copy(), "goal": self.goal.copy(), "position": self.position()}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain two finite joint velocities in rad/s")
        clipped = np.clip(action, -self.cfg.velocity_limit, self.cfg.velocity_limit)
        self.q = self.q + self.cfg.dt * clipped
        self.step_count += 1
        error = float(np.linalg.norm(self.goal - self.position()))
        success = error <= self.cfg.tolerance
        timeout = self.step_count >= self.cfg.max_steps
        info = {
            "error": error,
            "success": success,
            "timeout": timeout,
            "action_clipped": bool(np.any(action != clipped)),
            "applied_action": clipped.copy(),
        }
        return self.observe(), -error, bool(success or timeout), info


def dls_action(env: TwoLinkReachEnv) -> np.ndarray:
    error = env.goal - env.position()
    v = env.cfg.gain * error
    j = env.jacobian()
    lam2 = env.cfg.damping**2
    return j.T @ np.linalg.solve(j @ j.T + lam2 * np.eye(2), v)


def run_episode(seed: int, goal: tuple[float, float], cfg: Config) -> tuple[list[dict], dict]:
    env = TwoLinkReachEnv(cfg)
    env.reset(seed=seed, goal=goal)
    initial_q = env.q.copy()
    initial_error = float(np.linalg.norm(env.goal - env.position()))
    rows: list[dict] = []
    termination = "max_steps"
    for _ in range(cfg.max_steps):
        before = env.position().copy()
        raw = dls_action(env)
        _, _, done, info = env.step(raw)
        after = env.position().copy()
        rows.append({
            "step": env.step_count,
            "time_s": env.step_count * cfg.dt,
            "q1": env.q[0], "q2": env.q[1],
            "x": after[0], "y": after[1],
            "goal_x": env.goal[0], "goal_y": env.goal[1],
            "error": info["error"],
            "raw_action_norm": float(np.linalg.norm(raw)),
            "applied_qdot1": info["applied_action"][0],
            "applied_qdot2": info["applied_action"][1],
            "action_clipped": info["action_clipped"],
            "delta_position": float(np.linalg.norm(after - before)),
        })
        if info["success"]:
            termination = "success"
            break
        if done:
            break
    summary = {
        "seed": seed,
        "initial_q": initial_q.tolist(),
        "initial_error": initial_error,
        "goal": list(goal),
        "steps": len(rows),
        "success": termination == "success",
        "termination": termination,
        "final_error": rows[-1]["error"],
        "clipped_fraction": float(np.mean([r["action_clipped"] for r in rows])),
    }
    radius = float(np.linalg.norm(env.goal))
    inner, outer = abs(cfg.link_lengths[0] - cfg.link_lengths[1]), sum(cfg.link_lengths)
    summary["geometric_reachable"] = bool(inner <= radius <= outer)
    summary["geometric_error_lower_bound"] = float(max(inner - radius, radius - outer, 0.0))
    return rows, summary


def save_run(rows: list[dict], summary: dict, cfg: Config, output: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    with (output / "trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    xy = np.array([[r["x"], r["y"]] for r in rows])
    plt.figure(figsize=(6, 5))
    plt.plot(xy[:, 0], xy[:, 1], label="end-effector")
    plt.scatter([summary["goal"][0]], [summary["goal"][1]], marker="x", s=80, label="goal")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"termination={summary['termination']}, error={summary['final_error']:.4f} m")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "trajectory.png", dpi=160)
    plt.close()
