"""Replay frozen PPO runs, including every retained physical failure trace."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal(actual, expected, label):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), label
        for key in expected:
            equal(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), label
        for i, (a, b) in enumerate(zip(actual, expected)):
            equal(a, b, f"{label}[{i}]")
    elif isinstance(expected, float):
        assert np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-7, atol=1e-9), (label, actual, expected)
    else:
        assert actual == expected, (label, actual, expected)


def key(case):
    return (case["policy"], case["split"], case["guard"], case["mass_scale"], case["reset_seed"])


def verify(folder):
    manifest = read(folder/"manifest.json")
    paths = {r["path"] for r in manifest}
    assert len(paths) == len(manifest)
    assert paths == {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}
    for row in manifest:
        path = (folder/row["path"]).resolve()
        assert path.is_relative_to(folder) and digest(path) == row["sha256"], row["path"]
    summary, protocol = read(folder/"summary.json"), read(folder/"protocol.json")
    for name, sha in summary["source_sha256"].items():
        assert digest(folder/"source"/name) == sha, name
    # A fresh verifier process imports the exact historical implementation.
    sys.path.insert(0, str(folder/"source"))
    from continuous import episode
    from ppo import Agent
    cases = read(folder/"cases.json")
    kind, seeds = protocol["kind"], protocol["training_seeds"]
    policies = {"zero": "zero", "random": "random", "untrained": Agent(11 if kind == "arm" else 3, 2 if kind == "arm" else 1, 777)}
    if kind == "arm":
        policies["dls"] = "dls"
    for seed in seeds:
        policies[f"ppo_seed_{seed}"] = Agent.load(folder/f"ppo-seed-{seed}.npz")
    reset_sets = [set(protocol[f"{split}_reset_seeds"]) for split in ("validation", "test", "ood")]
    assert all(not reset_sets[i] & reset_sets[j] for i in range(3) for j in range(i))
    conditions = [("test", True, 1.), ("ood", True, 1.), ("test", True, 1.5), ("test", False, 1.)] if kind == "arm" else [("test", True, 1.)]
    expected = {(name, split, guard, mass, seed) for name in policies for split, guard, mass in conditions for seed in protocol[f"{split}_reset_seeds"]}
    assert len(cases) == len(expected) == summary["evaluation_cases"] and {key(c) for c in cases} == expected
    with gzip.open(folder/"failures.jsonl.gz", "rt", encoding="utf-8") as stream:
        traces = [json.loads(line) for line in stream]
    indexed = {key(t["case"]): t for t in traces}
    failed = {key(c) for c in cases if not c["success"]} if kind == "arm" else set()
    assert set(indexed) == failed and len(traces) == len(failed) == summary["retained_failure_traces"]
    for i, case in enumerate(cases):
        result, trajectory, _ = episode(kind, policies[case["policy"]], case["reset_seed"], case["split"], case["guard"], case["mass_scale"])
        result["policy"] = case["policy"]
        equal(result, case, f"case{i}")
        if key(case) in indexed:
            equal(indexed[key(case)]["case"], case, f"failure{i}")
            equal(trajectory, indexed[key(case)]["trajectory"], f"trace{i}")
    for name, stored in summary["aggregate"].items():
        selected = [c for c in cases if c["policy"] == name and c["split"] == "test" and c["guard"] and c["mass_scale"] == 1.]
        aggregate = {"successes": sum(c["success"] for c in selected), "cases": len(selected),
                     "mean_return": float(np.mean([c["return"] for c in selected])),
                     "mean_error_m": float(np.mean([c["final_error_m"] for c in selected])) if kind == "arm" else None,
                     "violations": sum(c["violations"] for c in selected)}
        equal(aggregate, stored, name)
    for training in summary["training"]:
        with (folder/f"training-seed-{training['seed']}.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert int(rows[-1]["environment_steps"]) == training["environment_steps"] == protocol["training_steps_per_seed"]
        assert len(rows) == training["environment_steps"]//1024
        assert 1 <= training["validation_selected_update"] <= len(rows)
        for row in rows:
            assert all(value in ("True", "False") or np.isfinite(float(value)) for value in row.values())
    return {"status": "PASS", "kind": kind, "replayed_cases": len(cases), "replayed_failure_trajectories": len(traces),
            "hashed_files": len(manifest), "implementation": "Frozen source snapshot; current helper fallback for historical snapshots",
            "current_source_differences": [name for name, sha in summary["source_sha256"].items() if digest(ROOT/name) != sha],
            "scope": "Stored-policy numerical replay and complete retained failure trajectories; training regeneration and learner competence assessed separately"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.folder.resolve())
    assert not args.output.resolve().is_relative_to(args.folder.resolve()), "Keep frozen manifest unchanged"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
