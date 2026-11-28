"""Two-link model-predictive control and a matched future-supervised action-block pilot."""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

from action_conditioned_pilot import ArmPhysics, physical_env, state
import mujoco
import numpy as np
from ppo import Network

ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "outputs/action-pilot-20261127/report.json"
TRAIN = tuple(range(4100, 4140))
VALIDATION = tuple(range(4200, 4210))
EVAL = {"test": tuple(range(4500, 4520)), "ood": tuple(range(4600, 4620))}
FIT_SEEDS = (0, 1, 2)
STEPS = 800
BLOCK = 5
ACTION_DIM = BLOCK * 2
# Stop first, then four cardinal and four diagonal bounded references.
CANDIDATES = np.array([[0, 0], [.8, 0], [-.8, 0], [0, .8], [0, -.8],
                       [.8, .8], [.8, -.8], [-.8, .8], [-.8, -.8]])
POLICIES = ("dls", "mpc_masked", "mpc_action")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    paths = (Path(__file__), ROOT / "action_conditioned_pilot.py", Path(physical_env.__file__),
             physical_env.MODEL, Path(sys.modules["ppo"].__file__), PILOT)
    return {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in paths}


def planner_rollout(initial, actions, weights, cfg):
    """Batch over candidate sequences; reuse the model's requested-action input contract."""
    x = np.broadcast_to(initial, (len(actions), len(initial))).copy()
    prediction = []
    for u in actions.transpose(1, 0, 2):
        # Encode the same guard before predicting a transition near a joint bound.
        applied = np.where((abs(x[:, :2]) > 2.2) & (x[:, :2] * u > 0) & cfg.guard, 0., u)
        x = x + np.column_stack((np.ones(len(x)), x, applied)) @ weights
        prediction.append(x.copy())
    return np.stack(prediction, axis=1)


def candidates(horizon):
    return np.repeat(CANDIDATES[:, None, :], horizon, axis=1)


def tips(joints):
    q1, q2 = np.moveaxis(joints, -1, 0)
    heading = q1 + q2
    return np.stack((.6*np.cos(q1)+.4*np.cos(heading), .6*np.sin(q1)+.4*np.sin(heading)), axis=-1)


def score(predictions, goal):
    positions = tips(predictions[:, -1, :2])
    final_speed = predictions[:, -1, 2:]
    return np.sum((positions-goal)**2, axis=1) + .003*np.sum(final_speed**2, axis=1)


def choose_plan(env, weights):
    menu = candidates(BLOCK)
    prediction = planner_rollout(state(env), menu, weights, env.cfg)
    costs = score(prediction, env.goal)
    # Collapse roundoff ties so the action-blind model's choice remains deterministic.
    index = int(np.flatnonzero(costs <= costs.min() + 1e-12)[0])
    return menu[index, 0], index, costs.tolist()


def dataset(env, seeds):
    windows = {"obs": [], "action": [], "future": [], "initial": [], "applied": [], "seed": [], "step": []}
    for seed in seeds:
        obs, _ = env.reset(seed)
        initial = state(env)
        trajectory = []
        while not env.ended:
            action = env.dls()
            after, _, _, _, info = env.step(action)
            trajectory.append((obs, action, state(env), initial, info["applied_velocity"]))
            obs, initial = after, state(env)
        for t in range(len(trajectory)-BLOCK+1):
            rows = trajectory[t:t+BLOCK]
            windows["obs"].append(rows[0][0])
            windows["action"].append(np.asarray([row[1] for row in rows]).ravel())
            # Normalize angular position and velocity increments with fixed physical scales.
            windows["future"].append(((np.asarray([row[2] for row in rows])-rows[0][3]) /
                                       np.array([.1, .1, 1., 1.])).ravel())
            windows["initial"].append(rows[0][3])
            windows["applied"].append([row[4] for row in rows])
            windows["seed"].append(seed)
            windows["step"].append(t)
    return {name: np.asarray(values) for name, values in windows.items()}


def train(data, validation, seed, aux, updates):
    model = Network(11, ACTION_DIM + 4*BLOCK, np.random.default_rng(seed))
    mean = data["obs"].mean(axis=0)
    scale = np.maximum(data["obs"].std(axis=0), .05)
    design = (data["obs"]-mean)/scale
    rng = np.random.default_rng(seed+700)
    for _ in range(updates):
        ids = rng.integers(0, len(design), size=64)
        prediction, cache = model.forward(design[ids])
        action_error = prediction[:, :ACTION_DIM] - data["action"][ids]
        future_error = prediction[:, ACTION_DIM:] - data["future"][ids]
        grad = np.column_stack((2*action_error/action_error.size, 2*aux*future_error/future_error.size))
        model.update(model.backward(cache, grad), .001)
    prediction, _ = model.forward((validation["obs"]-mean)/scale)
    metrics = {"validation_action_mae": float(np.mean(abs(prediction[:, :ACTION_DIM]-validation["action"]))),
               "validation_scaled_future_rmse": float(np.sqrt(np.mean((prediction[:, ACTION_DIM:]-validation["future"])**2)))}
    return {"parameters": model.p, "mean": mean, "scale": scale}, metrics


def execute(env, seed, method, artifact=None, model=None):
    obs, _ = env.reset(seed)
    initial = state(env).tolist()
    rows = []
    while not env.ended:
        before = state(env)
        choice, costs = None, None
        if method == "dls":
            action = env.dls()
        elif method.startswith("mpc_"):
            action, choice, costs = choose_plan(env, artifact)
        else:
            command, _ = model.forward((obs-artifact["mean"])/artifact["scale"])
            action = np.clip(command[:2], -1, 1)
        obs, _, terminated, truncated, info = env.step(action)
        rows.append({"step": env.steps, "x": before.tolist(), "action": action.tolist(),
                     "x_next": state(env).tolist(), "applied_rad_s": info["applied_velocity"],
                     "guard": info["guard_intervention"], "success": info["success"], "violation": info["violation"],
                     "terminated": terminated, "truncated": truncated, "goal_error_m": info["error_m"],
                     "choice": choice, "candidate_costs": costs})
    case = {"split": env.split, "seed": seed, "method": method, "initial": initial, "goal": env.goal.tolist(),
            "success": rows[-1]["success"], "guard_steps": sum(row["guard"] for row in rows), "steps": len(rows),
            "final_error_mm": 1000*rows[-1]["goal_error_m"],
            "termination": "success" if rows[-1]["success"] else "violation" if rows[-1]["violation"] else "timeout"}
    return case, rows


def branches(env, seed, models):
    """Matched open-loop branches with a varying action sequence, plus a shifted-action negative control."""
    results = []
    menu = candidates(BLOCK)
    # Switch the sign at step 3 to make a one-step shift detectable.
    menu[:, 2:] *= -1
    for index, actions in enumerate(menu):
        env.reset(seed)
        x0 = state(env)
        truth = []
        for action in actions:
            env.step(action)
            truth.append(state(env))
        for name, weights in models.items():
            fed = np.roll(actions, 1, axis=0) if name == "shifted" else actions
            estimated = planner_rollout(x0, fed[None], weights, env.cfg)[0]
            error = 1000*np.linalg.norm(tips(estimated[:, :2])-tips(np.asarray(truth)[:, :2]), axis=1)
            results.append({"split": env.split, "seed": seed, "candidate": index, "model": name,
                            "actions": actions.tolist(), "initial": x0.tolist(), "observed": np.asarray(truth).tolist(),
                            "predicted": estimated.tolist(), "tip_error_mm": error.tolist()})
    return results


def aggregate(cases, predictions):
    output = {}
    for mode in dict.fromkeys(row["split"] for row in cases):
        rows = [row for row in cases if row["split"] == mode]
        entry = {}
        for name in dict.fromkeys(row["method"] for row in rows):
            group = [r for r in rows if r["method"] == name]
            entry[name] = {"successes": sum(r["success"] for r in group), "cases": len(group),
                           "mean_final_error_mm": float(np.mean([r["final_error_mm"] for r in group])),
                           "violations": sum(r["termination"] == "violation" for r in group),
                           "guard_steps": sum(r["guard_steps"] for r in group)}
        entry["aux_pairs"] = {}
        for seed in FIT_SEEDS:
            pair = {r["seed"]: r["success"] for r in rows if r["method"] == f"bc_{seed}"}
            aux = {r["seed"]: r["success"] for r in rows if r["method"] == f"aux_{seed}"}
            entry["aux_pairs"][str(seed)] = {"wins": sum(aux[k] and not pair[k] for k in pair),
                                            "losses": sum(pair[k] and not aux[k] for k in pair)}
        entry["prediction"] = {}
        for name in ("masked", "action", "shifted"):
            errors = np.asarray([r["tip_error_mm"] for r in predictions if r["split"] == mode and r["model"] == name])
            entry["prediction"][name] = {"branches": len(errors), "step1_mm": float(errors[:, 0].mean()),
                                         "step5_mm": float(errors[:, -1].mean())}
        output[mode] = entry
    return output


def evaluate(folder, weights, networks, eval_seeds):
    cases, predictions = [], []
    with gzip.open(folder / "trajectories.jsonl.gz", "wt", encoding="utf-8", newline="\n") as stream:
        for split, seeds in eval_seeds.items():
            env = ArmPhysics(split=split)
            try:
                for seed in seeds:
                    predictions.extend(branches(env, seed, {"masked": weights["mpc_masked"],
                                                               "action": weights["mpc_action"], "shifted": weights["mpc_action"]}))
                    for name in (*POLICIES, *networks):
                        artifact, model = networks[name] if name in networks else (weights.get(name), None)
                        case, rows = execute(env, seed, name, artifact, model)
                        cases.append(case)
                        stream.write(json.dumps({"case": case, "steps": rows}, separators=(",", ":"), allow_nan=False)+"\n")
            finally:
                env.close()
    write(folder / "predictions.json", predictions)
    write(folder / "cases.json", cases)
    return aggregate(cases, predictions)


def load_network(artifact):
    model = Network(11, ACTION_DIM + 4*BLOCK, np.random.default_rng(0))
    model.p = {name: np.asarray(value) for name, value in artifact["parameters"].items()}
    return artifact, model


def weights_from_pilot():
    weights = read(PILOT)["weights"]
    return {"mpc_masked": np.asarray(weights["state_only"]), "mpc_action": np.asarray(weights["state_action"])}


def run(folder, aux, updates, quick):
    folder.mkdir(parents=True, exist_ok=False)
    train_seeds, validation_seeds = (TRAIN[:4], VALIDATION[:2]) if quick else (TRAIN, VALIDATION)
    eval_seeds = {split: seeds[:2] if quick else seeds for split, seeds in EVAL.items()}
    env = ArmPhysics()
    try:
        data, val = dataset(env, train_seeds), dataset(env, validation_seeds)
    finally:
        env.close()
    np.savez_compressed(folder / "expert_windows.npz", **{f"{split}_{k}": v for split, subset in
                        (("train", data), ("val", val)) for k, v in subset.items()})
    fitted, validation = {}, {}
    for seed in FIT_SEEDS:
        for prefix, loss_weight in (("bc", 0.), ("aux", aux)):
            name = f"{prefix}_{seed}"
            artifact, metrics = train(data, val, seed, loss_weight, updates)
            fitted[name] = {k: ({p: a.tolist() for p, a in v.items()} if k == "parameters" else v.tolist())
                            for k, v in artifact.items()}
            validation[name] = metrics
    write(folder / "weights.json", fitted)
    metrics = evaluate(folder, weights_from_pilot(), {k: load_network(v) for k, v in fitted.items()}, eval_seeds)
    write(folder / "report.json", {"dependencies": {"mujoco": mujoco.__version__, "numpy": np.__version__,
                                                     "python": sys.version.split()[0]},
                                   "sources": provenance(), "environment": asdict(env.cfg),
                                   "train_seeds": train_seeds, "validation_seeds": validation_seeds,
                                   "evaluation_seeds": eval_seeds, "training_seeds": FIT_SEEDS, "updates": updates,
                                   "aux_weight": aux, "train_windows": len(data["obs"]), "validation_windows": len(val["obs"]),
                                   "action_chunk": BLOCK, "executed_per_plan": 1, "candidate_actions": CANDIDATES.tolist(),
                                   "validation": validation, "metrics": metrics,
                                   "outputs_sha256": {p.name: sha(p) for p in sorted(folder.iterdir())}})
    return metrics


def compare(actual, expected, label):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"keys differ at {label}")
        for key in expected:
            compare(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"sequence differs at {label}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{label}[{i}]")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, atol=1e-7, rtol=1e-7):
            raise ValueError(f"numeric mismatch at {label}: {actual} vs {expected}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"outcome mismatch at {label}: {actual} vs {expected}")


def verify(frozen, output):
    report = read(frozen / "report.json")
    if report["sources"] != provenance():
        raise ValueError("source or prior model hash differs")
    for name, digest in report["outputs_sha256"].items():
        if sha(frozen / name) != digest:
            raise ValueError(f"frozen artifact changed: {name}")
    fitted = read(frozen / "weights.json")
    output.mkdir(parents=True, exist_ok=False)
    metrics = evaluate(output, weights_from_pilot(), {k: load_network(v) for k, v in fitted.items()}, report["evaluation_seeds"])
    compare(metrics, report["metrics"], "metrics")
    compare(read(output / "predictions.json"), read(frozen / "predictions.json"), "predictions")
    paths = [folder / "trajectories.jsonl.gz" for folder in (output, frozen)]
    count = 0
    with gzip.open(paths[0], "rt", encoding="utf-8") as first, gzip.open(paths[1], "rt", encoding="utf-8") as second:
        for count, (a, b) in enumerate(zip(first, second, strict=True), start=1):
            compare(json.loads(a), json.loads(b), f"case{count}")
    result = {"status": "PASS", "closed_loop_replayed": count,
              "prediction_records": len(read(output / "predictions.json")), "retained_failures":
                  sum(not row["success"] for row in read(output / "cases.json")),
              "numeric_atol": 1e-7, "numeric_rtol": 1e-7, "scope": "frozen weights, no retraining during replay"}
    write(output / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--aux", type=float, default=.25)
    parser.add_argument("--updates", type=int, default=STEPS)
    args = parser.parse_args()
    if not np.isfinite(args.aux) or args.aux < 0 or args.updates < 1:
        parser.error("choose finite nonnegative auxiliary weight and positive training updates")
    result = verify(args.verify, args.output) if args.verify else run(args.output, args.aux, args.updates, args.quick)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
