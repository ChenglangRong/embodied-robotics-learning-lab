"""Finite kinematic arm MDP: sampled Q-learning, model oracle and joint projection.

The learner receives only (joint1 index, joint2 index); the goal stays fixed.
TimeLimit is an external collection budget, so truncation preserves bootstrap.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
ACTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [0, 0]], dtype=int)
ACTION_NAMES = ["q1+", "q1-", "q2+", "q2-", "hold"]


@dataclass(frozen=True)
class Config:
    bins: int = 21
    gamma: float = 0.97
    alpha: float = 0.15
    episodes: int = 6000
    max_steps: int = 80
    tolerance: float = 0.045
    step_cost: float = 0.02
    success_bonus: float = 2.0
    goal_shift: float = 0.0

    def __post_init__(self):
        if self.bins < 5 or self.episodes < 1 or self.max_steps < 1:
            raise ValueError("bins >= 5, episodes >= 1, max_steps >= 1 required")
        if not (0 <= self.gamma < 1 and 0 < self.alpha <= 1 and self.tolerance > 0):
            raise ValueError("gamma in [0,1), alpha in (0,1], positive tolerance required")
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("finite configuration required")


def position(q):
    q = np.asarray(q)
    return np.stack((0.6*np.cos(q[..., 0])+0.4*np.cos(q.sum(axis=-1)),
                     0.6*np.sin(q[..., 0])+0.4*np.sin(q.sum(axis=-1))), axis=-1)


class ArmMDP:
    """Exact lattice dynamics, with an absorbing violation state outside the grid."""
    def __init__(self, cfg=Config(), filtered=True):
        self.cfg, self.filtered = cfg, filtered
        self.n = cfg.bins
        self.axes = (np.linspace(-1.5, 1.5, self.n), np.linspace(0, 2.4, self.n))
        ij = np.indices((self.n, self.n)).reshape(2, -1).T
        self.q = np.column_stack([self.axes[k][ij[:, k]] for k in range(2)])
        self.goal = position(self.q[int(0.7*(self.n-1))*self.n+int(0.7*(self.n-1))])
        self.goal = self.goal + np.array([cfg.goal_shift, 0.0])
        self.error = np.linalg.norm(position(self.q)-self.goal, axis=1)
        self.terminal = np.r_[self.error <= cfg.tolerance, True]
        self.failure_state = self.n*self.n
        proposal = ij[:, None, :]+ACTIONS[None, :, :]
        bad = np.any((proposal < 0) | (proposal >= self.n), axis=2)
        bounded = np.clip(proposal, 0, self.n-1)
        dest = bounded[..., 0]*self.n+bounded[..., 1]
        reward = -self.error[dest]-cfg.step_cost
        reward += cfg.success_bonus*self.terminal[dest]
        if not filtered:
            dest = np.where(bad, self.failure_state, dest)
            reward = np.where(bad, -2.0-cfg.step_cost, reward)
        self.dest = np.vstack([dest, np.full((1, 5), self.failure_state)])
        self.reward = np.vstack([reward, np.zeros((1, 5))])
        self.intervention = np.vstack([bad if filtered else np.zeros_like(bad), np.zeros((1, 5), bool)])
        self.violation = np.vstack([bad if not filtered else np.zeros_like(bad), np.zeros((1, 5), bool)])
        self.starts = np.flatnonzero(~self.terminal[:-1])
        self.reset(int(self.starts[0]))

    def reset(self, state):
        if state not in self.starts:
            raise ValueError("start must be a nonterminal grid state")
        self.state, self.steps = int(state), 0
        return self.state

    def step(self, action):
        if self.terminal[self.state] or self.steps >= self.cfg.max_steps:
            raise RuntimeError("reset required after episode completion")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < 5:
            raise ValueError("action must be an integer in [0,4]")
        before = self.state
        q = self.q[before].copy()
        q += ACTIONS[action]*np.array([self.axes[0][1]-self.axes[0][0], self.axes[1][1]-self.axes[1][0]])
        nxt = int(self.dest[before, action])
        self.state, self.steps = nxt, self.steps+1
        if nxt != self.failure_state:
            q = self.q[nxt].copy()
        terminated = bool(self.terminal[nxt])
        truncated = self.steps >= self.cfg.max_steps and not terminated
        info = {"success": terminated and nxt != self.failure_state,
                "violation": bool(self.violation[before, action]),
                "intervention": bool(self.intervention[before, action]),
                "error": float(np.linalg.norm(position(q)-self.goal)), "q": q.tolist()}
        return nxt, float(self.reward[before, action]), terminated, truncated, info


def td_target(reward, next_q, terminated, gamma):
    return reward+gamma*(not terminated)*float(np.max(next_q))


def value_iteration(env, tolerance=1e-10):
    v = np.zeros(env.failure_state+1)
    residuals = []
    for _ in range(2000):
        q = env.reward+env.cfg.gamma*v[env.dest]*(~env.terminal[env.dest])
        q[env.terminal] = 0
        new = q.max(axis=1)
        residual = float(np.max(np.abs(new-v)))
        residuals.append(residual)
        v = new
        if residual < tolerance:
            break
    q = env.reward+env.cfg.gamma*v[env.dest]*(~env.terminal[env.dest])
    q[env.terminal] = 0
    return q, residuals


def train(cfg, seed, progress=None):
    env, rng = ArmMDP(cfg), np.random.default_rng(seed)
    q = np.zeros((env.failure_state+1, 5))
    visits = np.zeros_like(q, dtype=np.int64)
    logs = []
    steps_total = 0
    for episode in range(cfg.episodes):
        state = env.reset(int(rng.choice(env.starts)))
        epsilon = max(0.08, 0.9*(1-episode/cfg.episodes))
        total, interventions, success = 0.0, 0, False
        for _ in range(cfg.max_steps):
            if rng.random() < epsilon:
                action = int(rng.integers(5))
            else:
                action = int(rng.choice(np.flatnonzero(q[state] == q[state].max())))
            nxt, reward, terminated, truncated, info = env.step(action)
            visits[state, action] += 1
            q[state, action] += cfg.alpha*(td_target(reward, q[nxt], terminated, cfg.gamma)-q[state, action])
            total += reward
            interventions += info["intervention"]
            steps_total += 1
            state = nxt
            if terminated or truncated:
                success = info["success"]
                break
        logs.append({"episode": episode+1, "environment_steps": steps_total, "return": total,
                     "success": success, "steps": env.steps, "epsilon": epsilon,
                     "interventions": interventions})
        if progress and (episode+1) % 100 == 0:
            progress(episode+1, cfg.episodes)
    return q, visits, logs


def rollout(cfg, q, start, filtered=True, random_seed=None):
    env, rng = ArmMDP(cfg, filtered), np.random.default_rng(random_seed)
    state = env.reset(int(start))
    rows = []
    total = 0.0
    for step in range(cfg.max_steps):
        action = int(rng.integers(5)) if random_seed is not None else int(np.argmax(q[state]))
        before = state
        state, reward, terminated, truncated, info = env.step(action)
        total += reward
        rows.append({"step": step+1, "state": before, "action": action, "next_state": state,
                     "reward": reward, "return": total, "terminated": terminated, "truncated": truncated,
                     **info})
        if terminated or truncated:
            break
    result = {"start": int(start), "success": info["success"], "steps": len(rows),
              "return": total, "final_error_m": info["error"],
              "interventions": sum(r["intervention"] for r in rows),
              "violations": sum(r["violation"] for r in rows),
              "termination": "success" if info["success"] else "violation" if info["violation"] else "truncated"}
    return rows, result


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_arm(cfg, row, caption, size=(680, 460)):
    im = Image.new("RGB", size, "#f0fdfa")
    draw = ImageDraw.Draw(im)
    q = np.array(row["q"])
    elbow = np.array([0.6*np.cos(q[0]), 0.6*np.sin(q[0])])
    def pixel(p):
        return (int(size[0]/2+210*p[0]), int(size[1]/2-210*p[1]))
    draw.line([pixel([0, 0]), pixel(elbow), pixel(position(q))], fill="#134e4a", width=9)
    for p in ([0, 0], elbow, position(q)):
        x, y = pixel(p)
        draw.ellipse((x-7, y-7, x+7, y+7), fill="#0d9488")
    x, y = pixel(ArmMDP(cfg).goal)
    draw.line((x-8, y-8, x+8, y+8), fill="#b45309", width=3)
    draw.line((x-8, y+8, x+8, y-8), fill="#b45309", width=3)
    draw.text((16, 16), caption, fill="#134e4a")
    draw.text((16, 40), f"step={row['step']}  action={ACTION_NAMES[row['action']]}  error={row['error']:.4f}m", fill="#134e4a")
    draw.text((16, size[1]-28), f"projection={row['intervention']} violation={row['violation']} | kinematic grid", fill="#134e4a")
    return im


def curve_image(logs, path):
    im = Image.new("RGB", (760, 400), "white")
    d = ImageDraw.Draw(im)
    d.text((24, 16), "Training success per 100 episodes (exploration enabled)", fill="#134e4a")
    d.line((60, 60, 60, 340, 720, 340), fill="#134e4a", width=2)
    d.text((8, 58), "100%", fill="#134e4a")
    d.text((20, 328), "0%", fill="#134e4a")
    d.text((530, 366), "x: episode; each line = seed", fill="#134e4a")
    for seed, rows in logs.items():
        points = []
        for i in range(0, len(rows), 100):
            mean = np.mean([r["success"] for r in rows[i:i+100]])
            points.append((60+660*(i+1)/len(rows), 340-280*mean))
        color = ["#0f766e", "#1d4ed8", "#b45309", "#9333ea", "#be123c"][seed % 5]
        if len(points) > 1:
            d.line(points, fill=color, width=2)
        d.text((80+seed*115, 42), f"seed {seed}", fill=color)
    im.save(path)


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run(output, cfg, seeds):
    if output.exists():
        raise FileExistsError("Choose a new output directory to preserve existing evidence")
    output.mkdir(parents=True)
    env = ArmMDP(cfg)
    oracle, residuals = value_iteration(env)
    save_json(output/"config.json", asdict(cfg))
    save_json(output/"value-iteration.json", {"residuals": residuals, "known_model_oracle": True})
    np.savez(output/"oracle.npz", q=oracle)
    policies = {"untrained": np.zeros_like(oracle), "random": np.zeros_like(oracle), "model_oracle": oracle}
    logs_all = {}
    training_steps = {}
    for seed in seeds:
        print(f"TRAIN seed={seed}", flush=True)
        q, visits, logs = train(cfg, seed, lambda e, n: print(f"seed={seed} episodes={e}/{n}", flush=True) if e % 1000 == 0 else None)
        np.savez(output/f"q-seed-{seed}.npz", q=q, visits=visits)
        save_csv(output/f"training-seed-{seed}.csv", logs)
        policies[f"q_seed_{seed}"] = q
        logs_all[seed] = logs
        training_steps[str(seed)] = logs[-1]["environment_steps"]
    curve_image(logs_all, output/"learning-curve.png")
    cases = []
    for name, q in policies.items():
        for start in env.starts:
            _, result = rollout(cfg, q, start, random_seed=10000+int(start) if name == "random" else None)
            cases.append({"policy": name, "filter": True, **result})
    for seed in seeds:
        q = policies[f"q_seed_{seed}"]
        for start in env.starts:
            _, result = rollout(cfg, q, start, filtered=False)
            cases.append({"policy": f"q_seed_{seed}", "filter": False, **result})
    save_csv(output/"evaluation.csv", cases)
    aggregate = {}
    for name in policies:
        group = [r for r in cases if r["policy"] == name and r["filter"]]
        aggregate[name] = {"successes": sum(r["success"] for r in group), "cases": len(group),
                           "success_rate": float(np.mean([r["success"] for r in group])),
                           "violations": sum(r["violations"] for r in group)}
    demos = output/"demos"
    demos.mkdir()
    for name in ("untrained", "random", f"q_seed_{seeds[0]}", "model_oracle"):
        rows, result = rollout(cfg, policies[name], int(env.starts[0]), random_seed=10000+int(env.starts[0]) if name == "random" else None)
        save_csv(demos/f"{name}.csv", rows)
        frames = [draw_arm(cfg, r, name) for r in rows]
        frames[0].save(demos/f"{name}.gif", save_all=True, append_images=frames[1:], duration=140, loop=0)
        save_json(demos/f"{name}.json", result)
    # Direct counterexample: same outbound proposal, enabled projection vs bypass.
    safety = []
    for filtered in (True, False):
        check = ArmMDP(cfg, filtered)
        check.reset(0)
        nxt, reward, terminated, truncated, info = check.step(1)
        safety.append({"filter": filtered, "action": "q1-", "next_state": nxt,
                       "reward": reward, "terminated": terminated, "truncated": truncated, **info})
    save_json(output/"safety-counterexample.json", safety)
    # Deliberate goal change: goal is absent from observation, so no goal generalization claim.
    shifted = Config(**{**asdict(cfg), "goal_shift": 0.25})
    failures = []
    for start in ArmMDP(shifted).starts[::10]:
        _, result = rollout(shifted, policies[f"q_seed_{seeds[0]}"], start)
        failures.append(result)
    save_json(output/"changed-goal.json", {"config": asdict(shifted), "cases": failures,
              "scope": "Changed MDP and unobserved goal; diagnostic transfer failure, outside the trained task."})
    summary = {"algorithm": "tabular Q-learning", "training_seeds": seeds,
               "source_sha256": source_hash(), "dependencies": {"python": platform.python_version(), "numpy": np.__version__},
               "training_environment_steps": training_steps, "aggregate": aggregate,
               "evaluation_scope": "Exhaustive nonterminal starts in the SAME finite fixed-goal MDP; training may visit every evaluated state.",
               "software_status": "verification pending", "task_status": "see per-policy outcomes",
               "claim_boundary": "Kinematic grid and deterministic joint projection; complex robot RL and physical safety remain separate projects.",
               "command": subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), "--output", str(output), "--episodes", str(cfg.episodes), "--seeds", *map(str, seeds)])}
    save_json(output/"summary.json", summary)
    (output/"source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    manifest = [{"path": p.relative_to(output).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(output.rglob("*")) if p.is_file()]
    save_json(output/"manifest.json", manifest)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)) or any(s < 0 for s in args.seeds):
        parser.error("distinct nonnegative seeds required")
    run(args.output, Config(episodes=args.episodes), args.seeds)
