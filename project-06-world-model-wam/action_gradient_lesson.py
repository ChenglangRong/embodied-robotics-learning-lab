"""Check one local action derivative in a frozen world model against two-link MuJoCo."""

import argparse
import json
from pathlib import Path
import platform
import sys

import mujoco
import numpy as np

import action_conditioned_pilot as pilot

ROOT = pilot.PROJECT.parent
sys.path.insert(0, str(ROOT / "project-01-simulation-software" / "src"))
from robot_lab import planar_arm

FROZEN = pilot.PROJECT / "outputs/action-pilot-20261127/report.json"
DEFAULT_OUTPUT = pilot.PROJECT / "outputs/action-gradient-lesson-20261128.json"
EPSILON = 0.001
STEP = 0.05


def model_distance_and_derivative(initial, goal, u1, weights, kinematics):
    """Differentiate distance with respect to a single u1 shared across all five steps."""
    x = initial.copy()
    sensitivity = np.zeros_like(x)
    action = [u1, 0.0]
    for _ in range(pilot.HORIZON):
        x += pilot.regressors(x, action, "state_action") @ weights
        # For the row-vector model, dx'/du = dx/du (I + W_state) + W_u1.
        sensitivity += sensitivity @ weights[1:5] + weights[5]
    tip_error = pilot.ArmPhysics.fk(x[:2]) - goal
    distance = np.linalg.norm(tip_error)
    if distance == 0:
        raise ValueError("Euclidean distance has no unique derivative at zero.")
    tip_derivative = kinematics.jacobian(x[:2]) @ sensitivity[:2]
    return 1000 * distance, 1000 * (tip_error / distance) @ tip_derivative


def measure_seed(env, seed, weights, kinematics):
    base, _ = pilot.simulate_branch(env, seed, [0.0, 0.0])
    initial, goal = base[0], env.goal.copy()
    baseline_mm = 1000 * np.linalg.norm(pilot.ArmPhysics.fk(base[-1, :2]) - goal)

    def measured_distance(u1):
        trajectory, _ = pilot.simulate_branch(env, seed, [u1, 0.0])
        np.testing.assert_array_equal(initial, trajectory[0])
        np.testing.assert_array_equal(goal, env.goal)
        return 1000 * np.linalg.norm(pilot.ArmPhysics.fk(trajectory[-1, :2]) - goal)

    predicted_mm, derivative = model_distance_and_derivative(initial, goal, 0.0, weights, kinematics)
    plus, minus = measured_distance(EPSILON), measured_distance(-EPSILON)
    central = (plus - minus) / (2 * EPSILON)
    fine = (measured_distance(EPSILON / 2) - measured_distance(-EPSILON / 2)) / EPSILON
    model_plus, _ = model_distance_and_derivative(initial, goal, EPSILON, weights, kinematics)
    model_minus, _ = model_distance_and_derivative(initial, goal, -EPSILON, weights, kinematics)
    model_central = (model_plus - model_minus) / (2 * EPSILON)
    np.testing.assert_allclose(derivative, model_central, rtol=1e-5, atol=1e-5)
    downhill = -STEP * np.sign(derivative)
    stepped_mm = measured_distance(downhill)
    return {
        "seed": seed, "initial_q_dq": initial.tolist(), "goal_xy_m": goal.tolist(),
        "model_baseline_mm": predicted_mm, "sim_baseline_mm": baseline_mm,
        "sim_plus_epsilon_mm": plus, "sim_minus_epsilon_mm": minus,
        "model_analytic": derivative, "model_central": model_central,
        "sim_central": central, "sim_half_epsilon_central": fine,
        "downhill_u1": downhill, "sim_downhill_mm": stepped_mm,
        "distance_saved_mm": baseline_mm - stepped_mm,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    output = parser.parse_args().output
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert frozen["evaluation"]["horizon"] == pilot.HORIZON
    weights = np.asarray(frozen["weights"]["state_action"])
    kinematics = planar_arm.TwoLinkReachEnv()
    groups = {}
    for split, seeds in frozen["evaluation"]["seeds"].items():
        env = pilot.ArmPhysics(split={"id": "test", "ood_negative_elbow": "ood"}[split])
        try:
            cases = [measure_seed(env, seed, weights, kinematics) for seed in seeds]
        finally:
            env.close()
        groups[split] = {
            "n": len(cases),
            "same_sign": sum(int(np.sign(c["model_analytic"]) == np.sign(c["sim_central"])) for c in cases),
            "mean_absolute_derivative_error": np.mean([abs(c["model_analytic"] - c["sim_central"]) for c in cases]),
            "max_sim_epsilon_halving_difference": max(abs(c["sim_central"] - c["sim_half_epsilon_central"]) for c in cases),
            "strict_distance_improvements": sum(int(c["distance_saved_mm"] > 0) for c in cases),
            "mean_distance_saved_mm": np.mean([c["distance_saved_mm"] for c in cases]),
            "cases": cases,
        }
    sources = [Path(__file__).resolve(), FROZEN, FROZEN.with_name("training.csv"),
               Path(pilot.__file__), Path(pilot.physical_env.__file__), pilot.physical_env.MODEL,
               Path(planar_arm.__file__)]
    report = {
        "scope": "Local open-loop action sensitivity in a two-link simulation. No policy training or humanoid evaluation.",
        "frozen_weights": FROZEN.relative_to(ROOT).as_posix() + "#weights.state_action",
        "command_from_repo_root": "python project-06-world-model-wam/action_gradient_lesson.py",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "mujoco": mujoco.__version__, "pillow": pilot.PIL.__version__},
        "config": {"baseline_u": [0.0, 0.0], "differentiated_action": "u1 (held constant at all steps); u2=0",
                   "action_unit": "dimensionless normalized joint velocity command",
                   "velocity_limit_rad_s": frozen["environment"]["velocity_limit"],
                   "control_dt_s": frozen["environment"]["dt"], "horizon": pilot.HORIZON,
                   "cost": "terminal Euclidean tip-to-goal distance (mm)",
                   "derivative_unit": "mm / normalized u1", "central_difference_epsilon": EPSILON,
                   "downhill_step_magnitude": STEP},
        "sha256": {p.relative_to(ROOT).as_posix(): pilot.sha(p) for p in sources},
        "groups": groups,
    }
    pilot.save_json(output, report)
    for name, summary in groups.items():
        print(name, {k: v for k, v in summary.items() if k != "cases"})


if __name__ == "__main__":
    main()
