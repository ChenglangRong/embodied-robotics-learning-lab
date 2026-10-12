"""From-scratch BC and budget-matched, subsampled DAgger in a kinematic arm."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import copy
import csv
from dataclasses import asdict
import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "project-01-simulation-software/src"))
from robot_lab.planar_arm import Config, TwoLinkReachEnv, dls_action


def encode(env):
    q = env.q
    return np.array([np.sin(q[0]), np.cos(q[0]), np.sin(q[1]), np.cos(q[1]),
                     *(env.goal - env.position())], dtype=float)


def specifications(n, seed, split, negative=False):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(n):
        r = rng.uniform(.40, .85)
        angle = rng.uniform(.10, 1.30) * (-1 if negative else 1)
        result.append({"id": f"{split}-{i:04d}", "split": split,
                       "reset_seed": int(rng.integers(0, 2**31-1)),
                       "goal": [float(r*np.cos(angle)), float(r*np.sin(angle))]})
    return result


def protocol():
    return {"environment": asdict(Config()), "observation_dim": 6, "action_dim": 2,
            "action_unit": "rad/s", "data_seed": 20261012,
            "train": specifications(120, 101, "train"), "validation": specifications(24, 202, "validation"),
            "test": specifications(24, 303, "test"), "ood": specifications(24, 404, "ood", negative=True),
            "ood_definition": "negative target angles [-1.30,-0.10]; same radial and reset distributions",
            "split_rule": "whole episode; distinct ids/reset seeds; fit normalization on train only",
            "checkpoint_rule": "lowest validation teacher-forced action MSE; test/OOD excluded",
            "dagger": {"rollouts": 16, "query_every_steps": 5, "executed_policy": "frozen base BC",
                       "iterations": 1, "variant": "subsampled, pure learner rollout aggregation"}}


def expert(env):
    return np.clip(dls_action(env), -env.cfg.velocity_limit, env.cfg.velocity_limit)


def collect(specs, behavior=expert, query_every=1, budget=None):
    if query_every < 1 or (budget is not None and budget < 1):
        raise ValueError("Query interval and label budget must be positive")
    if behavior is expert and query_every != 1:
        raise ValueError("Expert execution requires one label per step; subsampling requires a separate behavior policy")
    records = {key: [] for key in ("obs", "action", "raw_expert_action", "executed_action", "next_obs", "pre_q", "next_q",
                                  "goal", "episode_id", "step", "success", "timeout")}
    episodes = []
    for spec in specs:
        env = TwoLinkReachEnv()
        env.reset(seed=spec["reset_seed"], goal=spec["goal"])
        count, budget_stop = 0, False
        for step in range(env.cfg.max_steps):
            obs, q = encode(env), env.q.copy()
            should_query = step % query_every == 0
            raw = dls_action(env) if should_query else None
            label = np.clip(raw, -env.cfg.velocity_limit, env.cfg.velocity_limit) if should_query else None
            # Expert mode reuses this very label, keeping the query ledger exact.
            action = label if behavior is expert else behavior(env)
            _, _, done, info = env.step(action)
            if should_query:
                values = (obs, label, raw, info["applied_action"], encode(env), q, env.q.copy(), env.goal.copy(), spec["id"],
                          step, info["success"], info["timeout"])
                for key, value in zip(records, values):
                    records[key].append(value)
                count += 1
            if budget is not None and len(records["obs"]) >= budget:
                budget_stop = not done
                break
            if done:
                break
        episodes.append({**spec, "steps": env.step_count, "labels": count, "success": bool(info["success"]),
                         "timeout": bool(info["timeout"]), "budget_truncated": budget_stop,
                         "final_error_m": float(info["error"])})
        if budget is not None and len(records["obs"]) >= budget:
            break
    arrays = {key: np.asarray(value) for key, value in records.items()}
    return arrays, episodes


class MLP:
    """6 -> width -> width -> 2, tanh hidden units, linear velocity output."""
    def __init__(self, seed=0, width=48, mean=None, std=None):
        self.seed, self.width = seed, width
        self.mean = np.zeros(6) if mean is None else np.asarray(mean).copy()
        self.std = np.ones(6) if std is None else np.asarray(std).copy()
        rng = np.random.default_rng(seed)
        dims = [6, width, width, 2]
        self.params = {}
        for i, (n, m) in enumerate(zip(dims, dims[1:])):
            self.params[f"W{i}"] = rng.normal(0, np.sqrt(2/(n+m)), (n, m))
            self.params[f"b{i}"] = np.zeros(m)
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = copy.deepcopy(self.m)
        self.t = 0

    def forward(self, obs):
        a0 = (np.asarray(obs) - self.mean) / self.std
        a1 = np.tanh(a0 @ self.params["W0"] + self.params["b0"])
        a2 = np.tanh(a1 @ self.params["W1"] + self.params["b1"])
        prediction = 2 * (a2 @ self.params["W2"] + self.params["b2"])
        return prediction, (a0, a1, a2)

    def loss_grad(self, obs, targets):
        pred, (a0, a1, a2) = self.forward(obs)
        error = pred - targets
        derivative = 2 * error / error.size * 2  # MSE derivative and output scale.
        grads = {"W2": a2.T @ derivative, "b2": derivative.sum(axis=0)}
        derivative = (derivative @ self.params["W2"].T) * (1-a2*a2)
        grads.update(W1=a1.T @ derivative, b1=derivative.sum(axis=0))
        derivative = (derivative @ self.params["W1"].T) * (1-a1*a1)
        grads.update(W0=a0.T @ derivative, b0=derivative.sum(axis=0))
        return float(np.mean(error*error)), grads

    def update(self, grads, lr=.001):
        self.t += 1
        norm = np.sqrt(sum(float((g*g).sum()) for g in grads.values()))
        scale = min(1., 10./max(norm, 1e-12))
        for key, gradient in grads.items():
            g = gradient * scale
            self.m[key] = .9*self.m[key] + .1*g
            self.v[key] = .999*self.v[key] + .001*g*g
            mh = self.m[key] / (1-.9**self.t)
            vh = self.v[key] / (1-.999**self.t)
            self.params[key] -= lr * mh / (np.sqrt(vh)+1e-8)

    def raw_action(self, env):
        return self.forward(encode(env))[0]

    def __call__(self, env):
        return self.raw_action(env)

    def save(self, path):
        np.savez_compressed(path, **self.params, mean=self.mean, std=self.std,
                            seed=self.seed, width=self.width)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            result = cls(int(data["seed"]), int(data["width"]), data["mean"], data["std"])
            result.params = {k: data[k].copy() for k in result.params}
        return result


def mse(model, data):
    return float(np.mean((model.forward(data["obs"])[0]-data["action"])**2))


def fit(model, data, validation, steps, rng_seed, batch_size=256):
    rng = np.random.default_rng(rng_seed)
    log, best, best_value, best_step = [], copy.deepcopy(model), float("inf"), 0
    for step in range(1, steps+1):
        ids = rng.integers(0, len(data["obs"]), size=batch_size)
        _, grads = model.loss_grad(data["obs"][ids], data["action"][ids])
        model.update(grads)
        if step == 1 or step % 200 == 0 or step == steps:
            validation_loss = mse(model, validation)
            log.append({"update": step, "train_mse": mse(model, data), "validation_mse": validation_loss})
            if validation_loss < best_value:
                best, best_value, best_step = copy.deepcopy(model), validation_loss, step
            print(f"    update {step}/{steps}: val_action_mse={validation_loss:.6g}", flush=True)
    return best, log, best_step


def aggregate(a, b):
    return {key: np.concatenate([a[key], b[key]], axis=0) for key in a}


def rollout(spec, policy):
    env = TwoLinkReachEnv()
    env.reset(seed=spec["reset_seed"], goal=spec["goal"])
    rows = []
    for step in range(env.cfg.max_steps):
        raw = np.asarray(policy(env))
        _, _, done, info = env.step(raw)
        rows.append({"step": step, "time_s": env.step_count*env.cfg.dt, "q1": env.q[0], "q2": env.q[1],
                     "x_m": env.position()[0], "y_m": env.position()[1], "error_m": info["error"],
                     "qdot1_rad_s": info["applied_action"][0], "qdot2_rad_s": info["applied_action"][1],
                     "action_clipped": info["action_clipped"]})
        if done:
            break
    result = {**spec, "success": bool(info["success"]), "termination": "success" if info["success"] else "timeout",
              "steps": env.step_count, "final_error_m": float(info["error"]),
              "clipped_fraction": float(np.mean([r["action_clipped"] for r in rows]))}
    return result, rows


def evaluate(specs, policy, path=None):
    results, chosen_failure = [], False
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)
    for i, spec in enumerate(specs):
        result, trace = rollout(spec, policy)
        results.append(result)
        if path and (i == 0 or (not result["success"] and not chosen_failure)):
            write_csv(Path(path)/f"{spec['id']}.csv", trace)
            if not result["success"]:
                chosen_failure = True
    return {"episodes": len(results), "successes": sum(r["success"] for r in results),
            "success_rate": float(np.mean([r["success"] for r in results])),
            "mean_final_error_m": float(np.mean([r["final_error_m"] for r in results])),
            "mean_clipped_fraction": float(np.mean([r["clipped_fraction"] for r in results])),
            "per_episode": results}


def latency(model, obs):
    for _ in range(50):
        model.forward(obs)
    samples = []
    for _ in range(300):
        start = time.perf_counter_ns()
        model.forward(obs)
        samples.append((time.perf_counter_ns()-start)/1e6)
    return {"median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)),
            "includes": "single-observation NumPy normalization and network forward; no rendering, IPC or actuation"}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_experiment(output=None, mode="quick", base_steps=None, branch_steps=None, collect_only=False):
    started = time.perf_counter()
    output = Path(output) if output else ROOT/"outputs/runs"/datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    specs = protocol()
    save_json(output/"protocol.json", specs)
    datasets, episode_metadata = {}, {}
    for split in ("train", "validation", "test", "ood"):
        print(f"COLLECT {split}: {len(specs[split])} complete expert episodes", flush=True)
        datasets[split], episode_metadata[split] = collect(specs[split])
        np.savez_compressed(output/f"data-{split}.npz", **datasets[split])
    collection_seconds = time.perf_counter() - started
    save_json(output/"expert-episodes.json", episode_metadata)
    result = {"status": "collected" if collect_only else "running", "mode": mode,
              "date": datetime.datetime.now().astimezone().isoformat(), "python": sys.version,
              "numpy": np.__version__, "platform": platform.platform(), "protocol": specs,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "environment_source_sha256": hashlib.sha256((ROOT.parent/"project-01-simulation-software/src/robot_lab/planar_arm.py").read_bytes()).hexdigest(),
              "dataset_sha256": {s: hashlib.sha256((output/f"data-{s}.npz").read_bytes()).hexdigest() for s in datasets},
              "training_filter": "all expert labels retained, including expert timeout episodes",
              "data_counts": {s: len(d["obs"]) for s, d in datasets.items()}, "results": {},
              "timing_seconds": {"initial_collection_and_save": collection_seconds}}
    save_json(output/"summary.json", result)
    if collect_only:
        print(f"DATA {output}", flush=True)
        return output
    base_steps = base_steps or (600 if mode == "quick" else 1600)
    branch_steps = branch_steps or (400 if mode == "quick" else 1200)
    seeds = [0] if mode == "quick" else list(range(5))
    result["training_config"] = {"seeds": seeds, "base_updates": base_steps, "branch_updates": branch_steps,
                                 "width": 48, "batch_size": 256, "learning_rate": .001,
                                 "loss": "mean squared raw action velocity error", "optimizer": "Adam, global gradient norm clip 10"}
    train, validation = datasets["train"], datasets["validation"]
    mean, std = train["obs"].mean(axis=0), np.maximum(train["obs"].std(axis=0), 1e-3)
    result["baselines"] = {}
    for name, policy in (("expert", expert), ("zero", lambda env: np.zeros(2))):
        result["baselines"][name] = {s: evaluate(specs[s], policy, output/"traces"/name/s) for s in ("test", "ood")}
    for seed in seeds:
        print(f"TRAIN seed={seed}: base BC", flush=True)
        model = MLP(seed, mean=mean, std=std)
        stage_started = time.perf_counter()
        base, history, selected = fit(model, train, validation, base_steps, seed+10)
        training_seconds = {"bc": time.perf_counter() - stage_started}
        folder = output/f"seed-{seed}"
        folder.mkdir()
        write_csv(folder/"bc-loss.csv", history)
        models, selections = {"bc": base}, {"bc": selected}
        # Query trajectories and new expert trajectories use training-only specs.
        dagger_specs = specifications(16, 5000+seed, f"dagger-s{seed}")
        stage_started = time.perf_counter()
        queried, queried_episodes = collect(dagger_specs, behavior=base, query_every=5)
        dagger_seconds = time.perf_counter() - stage_started
        query_count = len(queried["obs"])
        fresh_specs = specifications(512, 7000+seed, f"fresh-s{seed}")
        stage_started = time.perf_counter()
        fresh, fresh_episodes = collect(fresh_specs, budget=query_count)
        fresh_seconds = time.perf_counter() - stage_started
        if len(fresh["obs"]) != query_count:
            raise RuntimeError("Fresh expert label budget was not matched")
        np.savez_compressed(folder/"dagger-labels.npz", **queried)
        np.savez_compressed(folder/"fresh-expert-labels.npz", **fresh)
        save_json(folder/"query-episodes.json", {"dagger": queried_episodes, "fresh": fresh_episodes})
        save_json(folder/"query-ledger.json", {"new_dagger_labels": query_count, "new_fresh_expert_labels": len(fresh["obs"]),
                  "new_more_steps_labels": 0, "queries_every_steps": 5,
                  "dagger_environment_steps": sum(e["steps"] for e in queried_episodes),
                  "fresh_environment_steps": sum(e["steps"] for e in fresh_episodes),
                  "dagger_collection_seconds": dagger_seconds, "fresh_collection_seconds": fresh_seconds,
                  "compute_matching": "equal minibatch updates/batch size for the three post-BC branches; wall time and environment steps reported separately"})
        for name, data in (("bc_more_steps", train), ("bc_more_data", aggregate(train, fresh)), ("dagger", aggregate(train, queried))):
            print(f"TRAIN seed={seed}: {name}, extra_labels={0 if name=='bc_more_steps' else query_count}", flush=True)
            stage_started = time.perf_counter()
            trained, history, selected = fit(copy.deepcopy(base), data, validation, branch_steps, seed+20)
            training_seconds[name] = time.perf_counter() - stage_started
            models[name], selections[name] = trained, selected
            write_csv(folder/f"{name}-loss.csv", history)
        result["results"][str(seed)] = {}
        for name, trained in models.items():
            trained.save(folder/f"{name}.npz")
            metrics = {"teacher_forced_test_mse": mse(trained, datasets["test"]),
                       "teacher_forced_ood_mse": mse(trained, datasets["ood"]),
                       "latency": latency(trained, datasets["test"]["obs"][0]),
                       "training_seconds": training_seconds[name], "selected_optimizer_update": trained.t,
                       "selected_stage_update": selections[name], "extra_expert_labels": 0 if name in ("bc", "bc_more_steps") else query_count}
            for split in ("test", "ood"):
                trace_path = folder/"traces"/name/split if seed == seeds[0] else None
                metrics[split] = evaluate(specs[split], trained, trace_path)
            result["results"][str(seed)][name] = metrics
            print(f"EVAL seed={seed} {name}: ID={metrics['test']['success_rate']:.3f}, OOD={metrics['ood']['success_rate']:.3f}", flush=True)
        save_json(output/"summary.json", result)
    result["aggregate"] = {}
    for name in ("bc", "bc_more_steps", "bc_more_data", "dagger"):
        result["aggregate"][name] = {}
        for split in ("test", "ood"):
            rates = [r[name][split]["success_rate"] for r in result["results"].values()]
            errors = [r[name][split]["mean_final_error_m"] for r in result["results"].values()]
            result["aggregate"][name][split] = {"per_seed_success_rate": rates, "mean_success_rate": float(np.mean(rates)),
                 "sample_std_success_rate": float(np.std(rates, ddof=1)) if len(rates)>1 else None,
                 "mean_final_error_m": float(np.mean(errors)), "training_seed_count": len(rates)}
    result["status"] = "training_completed"
    result["scope"] = "kinematic two-link benchmark; one subsampled DAgger iteration; no contact, camera policy or real robot claim"
    save_json(output/"summary.json", result)
    from visuals import create_outputs
    create_outputs(output)
    result["status"] = "completed"
    result["timing_seconds"]["total_including_rendering"] = time.perf_counter() - started
    save_json(output/"summary.json", result)
    (ROOT/"outputs").mkdir(exist_ok=True)
    (ROOT/"outputs/latest.txt").write_text(str(output.resolve()), encoding="utf-8")
    print(f"COMPLETE {output.resolve()}", flush=True)
    return output


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--base-steps", type=int)
    parser.add_argument("--branch-steps", type=int)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    if any(n is not None and n < 1 for n in (args.base_steps, args.branch_steps)):
        parser.error("training step counts must be positive")
    run_experiment(args.output, args.mode, args.base_steps, args.branch_steps, args.collect_only)
