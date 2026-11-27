"""Compare fixed linear dynamics models on matched two-link MuJoCo branches."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

import mujoco
import numpy as np
import PIL
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT.parent / "project-05-reinforcement-safety"))
import physical_env
from physical_env import ArmPhysics


TRAIN = range(4100, 4140)
EVALUATION = {"id": range(4300, 4320), "ood_negative_elbow": range(4400, 4420)}
TRAIN_STEPS = 25
HORIZON = 5
RIDGE = 0.01
ACTIONS = {"zero": [0, 0], "q1+": [.6, 0], "q1-": [-.6, 0],
           "q2+": [0, .6], "q2-": [0, -.6]}
MODELS = ("persistence", "state_only", "state_action")


def state(env):
    return np.r_[env.d.qpos, env.d.qvel]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def regressors(x, action, model):
    # Both fits have identical feature slots. The ablation masks the action.
    u = action if model == "state_action" else np.zeros(2)
    return np.r_[1.0, x, u]


def collect_training(folder):
    states, controls, targets = [], [], []
    env = ArmPhysics(split="train")
    try:
        with (folder / "training.csv").open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, lineterminator="\n")
            writer.writerow(["seed", "step", "q1", "q2", "dq1", "dq2", "u1", "u2",
                             "applied_dq_ref1", "applied_dq_ref2", "next_q1", "next_q2",
                             "next_dq1", "next_dq2", "goal_x", "goal_y", "guard",
                             "terminated", "truncated"])
            for seed in TRAIN:
                env.reset(seed)
                rng = np.random.default_rng(10000 + seed)
                for step in range(TRAIN_STEPS):
                    x = state(env)
                    u = rng.uniform(-.8, .8, 2)
                    _, _, terminated, truncated, info = env.step(u)
                    y = state(env)
                    writer.writerow([seed, step, *x, *u, *info["applied_velocity"], *y,
                                     *env.goal, info["guard_intervention"], terminated, truncated])
                    states.append(x)
                    controls.append(u)
                    targets.append(y - x)
                    if terminated or truncated:
                        break
    finally:
        env.close()

    weights = {}
    for model in MODELS[1:]:
        design = np.stack([regressors(x, u, model) for x, u in zip(states, controls)])
        penalty = np.eye(design.shape[1]) * RIDGE
        penalty[0, 0] = 0
        weights[model] = np.linalg.solve(design.T @ design + penalty,
                                          design.T @ np.asarray(targets))
    return weights, len(states)


def simulate_branch(env, seed, action):
    env.reset(seed)
    truth = [state(env)]
    applied = []
    for step in range(HORIZON):
        _, _, terminated, truncated, info = env.step(action)
        applied.append(info["applied_velocity"])
        truth.append(state(env))
        if (terminated or truncated) and step + 1 < HORIZON:
            raise RuntimeError(f"Branch ended before the frozen comparison horizon: {seed}")
    return np.stack(truth), applied


def evaluate(folder, weights):
    aggregate = {}
    rankings = []
    last_cases = {}
    with (folder / "branches.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(["split", "seed", "action", "step", "goal_x", "goal_y",
                         "applied_dq_ref1", "applied_dq_ref2", "true_q1", "true_q2",
                         "true_dq1", "true_dq2", "model", "pred_q1", "pred_q2",
                         "pred_dq1", "pred_dq2", "tip_error_mm"])
        for split, seeds in EVALUATION.items():
            env = ArmPhysics(split="ood" if split == "ood_negative_elbow" else "test")
            errors = {model: {h: [] for h in (1, HORIZON)} for model in MODELS}
            try:
                for seed in seeds:
                    distances = {model: [] for model in (*MODELS, "oracle")}
                    initial = None
                    for action_name, action in ACTIONS.items():
                        truth, applied = simulate_branch(env, seed, action)
                        if initial is not None:
                            np.testing.assert_array_equal(initial, truth[0])
                        initial = truth[0]
                        distances["oracle"].append(float(1000 * np.linalg.norm(
                            ArmPhysics.fk(truth[-1, :2]) - env.goal)))
                        for model in MODELS:
                            x = initial.copy()
                            for h in range(1, HORIZON + 1):
                                if model != "persistence":
                                    x = x + regressors(x, action, model) @ weights[model]
                                tip = ArmPhysics.fk(x[:2])
                                error = float(1000 * np.linalg.norm(tip - ArmPhysics.fk(truth[h, :2])))
                                writer.writerow([split, seed, action_name, h, *env.goal,
                                                 *applied[h - 1], *truth[h], model, *x, error])
                                if h in errors[model]:
                                    errors[model][h].append(error)
                            distances[model].append(float(1000 * np.linalg.norm(tip - env.goal)))
                    oracle = np.asarray(distances["oracle"])
                    for model in MODELS:
                        predicted = np.asarray(distances[model])
                        best = int(np.argmin(predicted))
                        regret = float(oracle[best] - oracle.min())
                        rankings.append({"split": split, "seed": seed, "model": model,
                                         "top_tie": int(np.sum(abs(predicted - predicted.min()) < 1e-10)) > 1,
                                         "chosen": list(ACTIONS)[best],
                                         "oracle": list(ACTIONS)[np.argmin(oracle)],
                                         "best_match": regret < 1e-10,
                                         "regret_mm": regret})
                        if model == "state_action" and (split not in last_cases or
                                regret > last_cases[split]["regret_mm"]):
                            last_cases[split] = {"seed": seed, "regret_mm": regret,
                                                 "distances_to_goal_mm": distances}
            finally:
                env.close()
            aggregate[split] = {}
            for model in MODELS:
                cases = [r for r in rankings if r["split"] == split and r["model"] == model]
                aggregate[split][model] = {
                    "mean_tip_error_mm": {h: float(np.mean(v)) for h, v in errors[model].items()},
                    "best_matches": sum(r["best_match"] for r in cases),
                    "top_ties": sum(r["top_tie"] for r in cases),
                    "resets": len(cases),
                    "mean_top1_regret_mm": float(np.mean([r["regret_mm"] for r in cases])),
                }
    with (folder / "ranking.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rankings[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rankings)
    return aggregate, last_cases


def draw_case(path, split, case):
    """Display the measured candidate ranking in the worst regret case."""
    img = Image.new("RGB", (680, 270), "white")
    draw = ImageDraw.Draw(img)
    draw.text((15, 12), f"Distance to goal after 100 ms | {split} | reset {case['seed']}",
              fill="black")
    draw.text((15, 32), "Measured (blue) and predicted (orange), millimetres. Lower is better.",
              fill="black")
    scores = case["distances_to_goal_mm"]
    bounds = np.array(scores["oracle"] + scores["state_action"])
    lo, hi = bounds.min() - 1, bounds.max() + 1
    draw.text((15, 53), f"Horizontal scale: {lo:.1f} to {hi:.1f} mm", fill="black")
    for i, name in enumerate(ACTIONS):
        y = 83 + 34 * i
        draw.text((15, y - 5), name, fill="black")
        draw.line((68, y, 501, y), fill="#c7cbd3")
        for model, offset, color in (("oracle", -4, "#215681"), ("state_action", 4, "#b95413")):
            value = scores[model][i]
            x = 68 + 433 * (value - lo) / (hi - lo)
            draw.ellipse((x - 3, y + offset - 3, x + 3, y + offset + 3), fill=color)
        draw.text((516, y - 5), f"{scores['oracle'][i]:.2f} / {scores['state_action'][i]:.2f}",
                  fill="black")
    draw.text((15, 251), f"Selected action excess distance: {case['regret_mm']:.3f} mm", fill="black")
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    folder = parser.parse_args().output_dir
    folder.mkdir(parents=True, exist_ok=True)
    weights, count = collect_training(folder)
    aggregate, cases = evaluate(folder, weights)
    worst_domain = max(cases, key=lambda name: cases[name]["regret_mm"])
    draw_case(folder / "worst-ranking.png", worst_domain, cases[worst_domain])
    report = {
        "scope": "Simulation only. Open-loop predictions and fixed-horizon candidate choice.",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "mujoco": mujoco.__version__, "pillow": PIL.__version__},
        "environment": asdict(physical_env.PhysicsConfig()),
        "training": {"reset_seeds": list(TRAIN), "action_rng": "default_rng(10000+reset_seed)",
                     "uniform_action": [-.8, .8], "max_steps_per_episode": TRAIN_STEPS,
                     "transitions": count, "ridge": RIDGE, "intercept_penalized": False,
                     "inputs": ["intercept", "q1", "q2", "dq1", "dq2", "u1", "u2"],
                     "output": "increment [q1, q2, dq1, dq2]"},
        "evaluation": {"seeds": {k: list(v) for k, v in EVALUATION.items()},
                       "ordered_candidates": ACTIONS, "horizon": HORIZON,
                       "tie_rule": "First minimal predicted goal distance (zero first).",
                       "metric_unit": "Euclidean end-effector distance in millimetres; equal branch weight."},
        "weights": {name: value.tolist() for name, value in weights.items()},
        "metrics": aggregate, "largest_regret_by_domain": cases,
        "sha256": {path.name: sha(path) for path in
                   [Path(__file__), Path(physical_env.__file__), physical_env.MODEL,
                    folder / "training.csv", folder / "branches.csv", folder / "ranking.csv"]},
    }
    save_json(folder / "report.json", report)
    print(json.dumps(aggregate, indent=2))
    print({k: (v["seed"], v["regret_mm"]) for k, v in cases.items()})


if __name__ == "__main__":
    main()
