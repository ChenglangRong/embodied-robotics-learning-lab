"""Hash, protocol and deterministic exhaustive policy replay verification."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from tabular_lab import ArmMDP, Config, rollout, source_hash, save_json, value_iteration


def verify(folder):
    folder = folder.resolve()
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    expected = {r["path"] for r in manifest}
    assert len(expected) == len(manifest)
    assert expected == {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json"}
    for row in manifest:
        path = (folder/row["path"]).resolve()
        assert path.is_relative_to(folder)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["path"]
    summary = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    assert summary["source_sha256"] == source_hash()
    assert hashlib.sha256((folder/"source-snapshot.py").read_bytes()).hexdigest() == source_hash()
    cfg = Config(**json.loads((folder/"config.json").read_text(encoding="utf-8")))
    env = ArmMDP(cfg)
    oracle, residuals = value_iteration(env)
    assert residuals[-1] < 1e-10
    np.testing.assert_allclose(np.load(folder/"oracle.npz")["q"], oracle, atol=1e-10, rtol=1e-10)
    policies = {"untrained": np.zeros_like(oracle), "random": np.zeros_like(oracle), "model_oracle": oracle}
    for seed in summary["training_seeds"]:
        with np.load(folder/f"q-seed-{seed}.npz") as model:
            assert model["q"].shape == oracle.shape and np.isfinite(model["q"]).all()
            policies[f"q_seed_{seed}"] = model["q"].copy()
            with (folder/f"training-seed-{seed}.csv").open(encoding="utf-8", newline="") as f:
                logs = list(csv.DictReader(f))
            assert len(logs) == cfg.episodes
            assert model["visits"].sum() == int(logs[-1]["environment_steps"]) == summary["training_environment_steps"][str(seed)]
    with (folder/"evaluation.csv").open(encoding="utf-8", newline="") as f:
        cases = list(csv.DictReader(f))
    keys = [(r["policy"], r["filter"], int(r["start"])) for r in cases]
    assert len(keys) == len(set(keys))
    wanted = {(name, "True", int(start)) for name in policies for start in env.starts}
    wanted |= {(f"q_seed_{seed}", "False", int(start)) for seed in summary["training_seeds"] for start in env.starts}
    assert set(keys) == wanted
    for row in cases:
        _, replay = rollout(cfg, policies[row["policy"]], int(row["start"]), filtered=row["filter"] == "True",
                            random_seed=10000+int(row["start"]) if row["policy"] == "random" else None)
        for name in ("success", "steps", "interventions", "violations", "termination", "start"):
            assert row[name] == str(replay[name]), (row["policy"], row["start"], name)
        for name in ("return", "final_error_m"):
            assert np.isclose(float(row[name]), replay[name], rtol=1e-10, atol=1e-10), name
    for name, aggregate in summary["aggregate"].items():
        group = [r for r in cases if r["policy"] == name and r["filter"] == "True"]
        successes = sum(r["success"] == "True" for r in group)
        assert aggregate["successes"] == successes and aggregate["cases"] == len(group)
        assert aggregate["success_rate"] == successes/len(group)
        assert aggregate["violations"] == sum(int(r["violations"]) for r in group)
    safety = json.loads((folder/"safety-counterexample.json").read_text(encoding="utf-8"))
    assert safety[0]["intervention"] and not safety[0]["violation"]
    assert safety[1]["violation"] and safety[1]["terminated"]
    changed = json.loads((folder/"changed-goal.json").read_text(encoding="utf-8"))
    shifted = Config(**changed["config"])
    for case in changed["cases"]:
        _, replay = rollout(shifted, policies[f"q_seed_{summary['training_seeds'][0]}"], case["start"])
        for name in ("start", "success", "steps", "interventions", "violations", "termination"):
            assert case[name] == replay[name]
        for name in ("return", "final_error_m"):
            assert np.isclose(case[name], replay[name], rtol=1e-10, atol=1e-10)
    return {"status": "PASS", "exhaustive_policy_replays": len(cases), "changed_goal_replays": len(changed["cases"]),
            "hashes": len(manifest), "training_rerun": False,
            "scope": "Saved-policy exhaustive replay, data/protocol/hash checks. Training regeneration and independent learner assessment are separate."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.folder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)
    print(json.dumps(report, indent=2))
