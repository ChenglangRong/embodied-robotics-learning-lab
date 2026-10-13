"""Train/evaluate PPO on Pendulum and physical goal-conditioned reaching."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from physical_env import ArmPhysics, MODEL, Pendulum, PhysicsConfig
from ppo import Agent, gae
from tabular_lab import save_csv, save_json

ROOT = Path(__file__).resolve().parent
SOURCES = ["ppo.py", "physical_env.py", "continuous.py", "model/arm2d-rl.xml"]


def make_env(kind, split="test", guard=True, mass_scale=1.):
    return Pendulum() if kind == "pendulum" else ArmPhysics(PhysicsConfig(guard=guard, mass_scale=mass_scale), split)


def episode(kind, policy, seed, split="test", guard=True, mass_scale=1., render=False, viewer=False):
    env = make_env(kind, split, guard, mass_scale)
    obs, _ = env.reset(seed)
    rng = np.random.default_rng(seed+5000000)
    total, rows, frames = 0., [], []
    handle = None
    if viewer:
        import mujoco.viewer
        handle = mujoco.viewer.launch_passive(env.m, env.d)
        handle.cam.lookat[:] = [.25, .05, .1]
        handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = 2., 90, -75
    try:
        for step in range(200 if kind == "pendulum" else env.cfg.horizon):
            started = time.monotonic()
            if isinstance(policy, Agent):
                action = policy.sample(obs[None], deterministic=True)[0][0]
            elif policy == "dls":
                action = env.dls()
            elif policy == "random":
                action = rng.uniform(-1, 1, env.action_dim)
            else:
                action = np.zeros(env.action_dim)
            obs, reward, terminated, truncated, info = env.step(action)
            total += info.get("raw_reward", reward)
            rows.append({"step": step+1, "reward": reward, "return": total, "action": action.tolist(),
                         "terminated": terminated, "truncated": truncated, **info})
            if render and step % 5 == 0:
                im = Image.fromarray(env.render())
                draw = ImageDraw.Draw(im)
                draw.text((12, 12), f"MuJoCo step {step+1} | error {info['error_m']:.3f} m", fill="black")
                draw.text((12, 30), f"success={info['success']} guard={info['guard_intervention']}", fill="black")
                frames.append(im)
            if handle:
                handle.sync()
                if not handle.is_running():
                    break
                time.sleep(max(0, env.cfg.dt-(time.monotonic()-started)))
            if terminated or truncated:
                break
        result = {"reset_seed": seed, "split": split, "guard": guard, "mass_scale": mass_scale,
                  "return": total, "steps": len(rows), "success": info.get("success", False),
                  "final_error_m": info.get("error_m", None),
                  "violations": sum(r.get("violation", False) for r in rows),
                  "guard_interventions": sum(r.get("guard_intervention", False) for r in rows),
                  "torque_peak_nm": max(r.get("torque_peak_nm", 0) for r in rows),
                  "termination": "success" if info.get("success") else "violation" if info.get("violation") else "truncated" if truncated else "viewer_closed"}
        return result, rows, frames
    finally:
        if handle:
            handle.close()
        env.close()


def evaluate(kind, policy, seeds, split="test", guard=True, mass_scale=1.):
    return [episode(kind, policy, s, split, guard, mass_scale)[0] for s in seeds]


def train(kind, seed, steps, output, nenv=8, rollout=128, epochs=6):
    if steps <= 0 or steps % (nenv*rollout):
        raise ValueError("positive training steps divisible by 1024 required")
    envs = [make_env(kind, "train") for _ in range(nenv)]
    agent = Agent(envs[0].obs_dim, envs[0].action_dim, seed)
    obs = np.stack([env.reset(seed*1000+i)[0] for i, env in enumerate(envs)])
    logs, episode_returns, episode_success = [], [], []
    returns_in_progress = np.zeros(nenv)
    validation_seeds = list(range(900000, 900008))
    best_score, best_update = -float("inf"), 0
    started = time.perf_counter()
    try:
        for update in range(steps//(nenv*rollout)):
            data = {k: [] for k in ("obs", "z", "logp", "value", "reward", "next_value", "terminated", "truncated")}
            for _ in range(rollout):
                actions, z, lp = agent.sample(obs)
                data["obs"].append(obs.copy())
                data["z"].append(z)
                data["logp"].append(lp)
                data["value"].append(agent.value(obs))
                transitions = [env.step(a) for env, a in zip(envs, actions)]
                finals = np.stack([t[0] for t in transitions])
                rewards = np.array([t[1] for t in transitions])
                terms, truncs = np.array([t[2] for t in transitions]), np.array([t[3] for t in transitions])
                data["reward"].append(rewards)
                data["next_value"].append(agent.value(finals))
                data["terminated"].append(terms)
                data["truncated"].append(truncs)
                returns_in_progress += np.array([t[4].get("raw_reward", t[1]) for t in transitions])
                for i in np.flatnonzero(terms|truncs):
                    episode_returns.append(float(returns_in_progress[i]))
                    episode_success.append(transitions[i][4].get("success", False))
                    returns_in_progress[i] = 0
                    finals[i] = envs[i].reset()[0]
                obs = finals
            data = {k: np.array(v) for k, v in data.items()}
            adv, targets = gae(data["reward"], data["value"], data["next_value"], data["terminated"], data["truncated"])
            diag = agent.optimize(data["obs"].reshape(-1, envs[0].obs_dim), data["z"].reshape(-1, envs[0].action_dim),
                                  data["logp"].ravel(), adv.ravel(), targets.ravel(), epochs=epochs)
            logs.append({"update": update+1, "environment_steps": (update+1)*nenv*rollout,
                         "training_return_last50": float(np.mean(episode_returns[-50:])) if episode_returns else 0.,
                         "training_success_last50": float(np.mean(episode_success[-50:])) if episode_success else 0.,
                         **diag})
            if (update+1) % 8 == 0 or update+1 == steps//(nenv*rollout):
                validation = evaluate(kind, agent, validation_seeds)
                score = float(np.mean([r["return"] for r in validation])) if kind == "pendulum" else float(np.mean([r["success"] for r in validation]))-0.01*float(np.mean([r["final_error_m"] for r in validation]))
                if score > best_score:
                    best_score, best_update = score, update+1
                    agent.save(output/f"ppo-seed-{seed}.npz")
                print(f"{kind} seed={seed} steps={(update+1)*nenv*rollout}/{steps} validation={score:.4f}", flush=True)
        save_csv(output/f"training-seed-{seed}.csv", logs)
        return {"seed": seed, "environment_steps": steps, "physics_substeps": steps*5 if kind == "arm" else None,
                "validation_selected_update": best_update, "validation_score": best_score,
                "wall_seconds": time.perf_counter()-started, "epochs": epochs, "rollout": rollout, "environments": nenv}
    finally:
        for env in envs:
            env.close()


def run(kind, output, steps, seeds):
    if output.exists():
        raise FileExistsError("New directory required; preserve frozen experiments")
    output.mkdir(parents=True)
    snapshot = output/"source"
    snapshot.mkdir()
    source_hashes = {}
    for name in SOURCES:
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
        source_hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    protocol = {"kind": kind, "training_steps_per_seed": steps, "training_seeds": seeds,
                "validation_reset_seeds": list(range(900000, 900008)), "test_reset_seeds": list(range(1000000, 1000024)),
                "ood_reset_seeds": list(range(2000000, 2000024)), "physics_config": asdict(PhysicsConfig()),
                "algorithm": "NumPy tanh-Gaussian PPO; GAE gamma=.97 lambda=.95, clip=.2, Adam 3e-4, norm cap=.5, minibatch256, epochs6, empirical KL stop .03; latent Gaussian entropy coefficient0",
                "split_scope": "Disjoint reset seeds, overlapping ID task distributions; OOD mirrored elbow-target branch. Mass shift is separate, paired with ID resets.",
                "checkpoint_selection": "8 validation resets; score=success rate-.01*mean error for arm, mean raw return for Pendulum; test outcomes excluded from selection"}
    save_json(output/"protocol.json", protocol)
    training = [train(kind, seed, steps, output) for seed in seeds]
    env = make_env(kind)
    initial = Agent(env.obs_dim, env.action_dim, 777)
    env.close()
    policies = {"zero": "zero", "random": "random", "untrained": initial}
    if kind == "arm":
        policies["dls"] = "dls"
    for seed in seeds:
        policies[f"ppo_seed_{seed}"] = Agent.load(output/f"ppo-seed-{seed}.npz")
    all_cases, traces = [], []
    for name, policy in policies.items():
        conditions = [("test", True, 1.), ("ood", True, 1.), ("test", True, 1.5), ("test", False, 1.)] if kind == "arm" else [("test", True, 1.)]
        for split, guard, mass in conditions:
            split_seeds = protocol["ood_reset_seeds"] if split == "ood" else protocol["test_reset_seeds"]
            for reset_seed in split_seeds:
                result, rows, _ = episode(kind, policy, reset_seed, split, guard, mass)
                result["policy"] = name
                all_cases.append(result)
                if not result["success"] and kind == "arm":
                    traces.append({"case": result, "trajectory": rows})
    save_json(output/"cases.json", all_cases)
    with gzip.open(output/"failures.jsonl.gz", "wt", encoding="utf-8") as stream:
        for record in traces:
            stream.write(json.dumps(record)+"\n")
    aggregate = {}
    for name in policies:
        cases = [r for r in all_cases if r["policy"] == name and r["split"] == "test" and r["guard"] and r["mass_scale"] == 1.]
        aggregate[name] = {"successes": sum(r["success"] for r in cases), "cases": len(cases),
                           "mean_return": float(np.mean([r["return"] for r in cases])),
                           "mean_error_m": float(np.mean([r["final_error_m"] for r in cases])) if kind == "arm" else None,
                           "violations": sum(r["violations"] for r in cases)}
    if kind == "arm":
        demos = output/"demos"
        demos.mkdir()
        chosen = ["dls", "untrained", f"ppo_seed_{seeds[0]}"]
        for name in chosen:
            result, rows, frames = episode(kind, policies[name], protocol["test_reset_seeds"][0], render=True)
            save_csv(demos/f"{name}.csv", rows)
            save_json(demos/f"{name}.json", result)
            frames[0].save(demos/f"{name}.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    summary = {"kind": kind, "training": training, "aggregate": aggregate,
               "source_sha256": source_hashes, "evaluation_cases": len(all_cases), "retained_failure_traces": len(traces),
               "dependencies": {name: importlib.metadata.version(name) for name in ["numpy", "Pillow", "mujoco", "gymnasium"]},
               "python": platform.python_version(), "scope": "Goal-conditioned velocity actions, retained velocity servo and hard torque clipping; planar MuJoCo inertia, no contact manipulation or certified physical safety. GPT and pi05 inference are separate planned integrations.",
               "task_status": "Per-policy results retained; no automatic pass from software tests"}
    save_json(output/"summary.json", summary)
    manifest = [{"path": p.relative_to(output).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                for p in sorted(output.rglob("*")) if p.is_file()]
    save_json(output/"manifest.json", manifest)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["arm", "pendulum"], default="arm")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=65536)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--demo", type=Path)
    parser.add_argument("--policy", default="ppo_seed_0")
    parser.add_argument("--split", choices=["test", "ood"], default="test")
    parser.add_argument("--case", type=int, default=0)
    args = parser.parse_args()
    if args.demo:
        protocol = json.loads((args.demo/"protocol.json").read_text(encoding="utf-8"))
        if protocol["kind"] != "arm" or not 0 <= args.case < 24:
            parser.error("Live demonstration requires arm run and case0..23")
        policy = Agent.load(args.demo/(args.policy.replace("ppo_seed_", "ppo-seed-")+".npz")) if args.policy.startswith("ppo_seed_") else args.policy
        result, _, _ = episode("arm", policy, protocol[f"{args.split}_reset_seeds"][args.case], args.split, viewer=True)
        print(json.dumps(result, indent=2))
    elif args.output:
        if len(args.seeds) != len(set(args.seeds)) or any(s < 0 for s in args.seeds):
            parser.error("Distinct nonnegative seeds required")
        run(args.kind, args.output, args.steps, args.seeds)
    else:
        parser.error("--output or --demo required")
