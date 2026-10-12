"""Re-run every saved physical case and audit frozen data/model provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from platform_learning import ROOT, PLATFORM_ROOT, DURATION, ReferenceMLP, PhaseBC, episode, save_json, lab


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run, output, replay=True, portable=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    revision = subprocess.check_output(["git", "-C", str(lab.CACHE), "rev-parse", "HEAD"], text=True).strip()
    assert revision == summary["menagerie_commit"], "Upstream model revision changed"
    model_changes = subprocess.check_output(["git", "-C", str(lab.CACHE), "status", "--porcelain", "--untracked-files=no"], text=True)
    assert not model_changes.strip(), "Preserve and audit modified upstream model files"
    sources = {"platform_learning.py": ROOT / "platform_learning.py", "learning.py": ROOT / "learning.py",
               "lab.py": PLATFORM_ROOT / "lab.py", "deep_practice.py": PLATFORM_ROOT / "deep_practice.py"}
    for name, expected in summary["source_sha256"].items():
        assert digest(run / "source-snapshot" / name) == expected, f"Snapshot changed: {name}"
        assert digest(sources[name]) == expected, f"Current source changed: {name}"
    started = time.perf_counter()
    mode = "integrity_only" if not replay else "portable_replay" if portable else "exact_replay"
    report = {"status": "running", "run": str(run), "verification_mode": mode,
              "model_revision": revision, "portable_metric_rtol": 1e-7 if portable else 0,
              "portable_metric_atol": 1e-9 if portable else 0, "checks": [], "replayed": []}
    save_json(output, report)
    for kind, record in summary["platforms"].items():
        folder = run / kind
        protocol = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
        datasets, all_ids = {}, set()
        all_seeds = []
        for split, specs in protocol["specs"].items():
            file = folder / f"data-{split}.npz"
            assert digest(file) == record["dataset_sha256"][split], f"Dataset changed: {kind}/{split}"
            with np.load(file, allow_pickle=False) as data:
                datasets[split] = {key: data[key].copy() for key in data.files}
            data = datasets[split]
            ids = {spec["id"] for spec in specs}
            assert set(data["episode_id"].tolist()) == ids
            assert all_ids.isdisjoint(ids), f"Episode leakage: {kind}/{split}"
            all_ids.update(ids)
            all_seeds.extend(spec["seed"] for spec in specs)
            assert len(data["obs"]) == record["dataset_rows"][split]
            assert data["obs"].shape[1] == record["observation_dim"]
            assert data["action"].shape[1] == record["action_dim"]
            for key in ("obs", "action", "next_obs", "pre_qpos", "next_qpos", "time_s"):
                assert np.isfinite(data[key]).all(), f"Invalid array: {kind}/{split}/{key}"
            for spec in specs:
                mask = data["episode_id"] == spec["id"]
                np.testing.assert_allclose(data["time_s"][mask], np.arange(0, DURATION[kind], .02), atol=1e-12)
        assert len(all_seeds) == len(set(all_seeds)), f"Reset seed overlap: {kind}"
        train = datasets["train"]
        phase = PhaseBC.load(folder / "phase-bc.npz")
        expected_phase = PhaseBC.fit(train, DURATION[kind])
        np.testing.assert_array_equal(phase.phase, expected_phase.phase)
        np.testing.assert_array_equal(phase.actions, expected_phase.actions)
        models = {}
        for seed in summary["training_seeds"]:
            model = ReferenceMLP.load(folder / f"bc-seed-{seed}.npz")
            for actual, expected in ((model.mean, train["obs"].mean(0)),
                                     (model.std, np.maximum(train["obs"].std(0), .001)),
                                     (model.action_mean, train["action"].mean(0)),
                                     (model.action_std, np.maximum(train["action"].std(0), .001))):
                np.testing.assert_array_equal(actual, expected)
            models[str(seed)] = model
        report["checks"].append({"platform": kind, "source_data_hashes": "PASS", "whole_episode_splits": "PASS",
                                 "train_only_normalization": "PASS", "train_only_phase_bc": "PASS"})
        evaluations = [(name, None, values) for name, values in record["baselines"].items()]
        evaluations += [(name, seed, values) for seed, row in record["results"].items()
                        for name, values in row["policies"].items()]
        for policy, seed, splits in evaluations:
            model = phase if policy == "phase_bc" else models.get(seed)
            for split, cases in splits.items():
                specs = protocol["specs"][split]
                assert len(cases) == len(specs)
                for expected, spec in zip(cases, specs):
                    assert expected["spec"] == spec
                    assert expected["passed"] == bool(expected["completed"] and all(expected["criteria"].values()))
                    if policy != "expert":
                        assert expected["expert_calls"] == 0
                    if not replay:
                        continue
                    actual, _ = episode(kind, spec, policy, model)
                    # Diagnostic source metadata may evolve; task and physics evidence must match exactly.
                    for key in ("passed", "completed", "criteria", "metrics", "expert_calls", "control_steps",
                                "base_model_xml_sha256", "action_clipped_fraction", "termination"):
                        context = f"Replay mismatch: {kind}/{policy}/{split}/{spec['id']}/{key}"
                        if portable and key == "base_model_xml_sha256":
                            # Raw XML contains a host-specific absolute meshdir. Pinned clean model
                            # assets, source and frozen specs are checked independently above.
                            continue
                        if portable and key == "metrics":
                            assert actual[key].keys() == expected[key].keys(), context
                            for metric, value in expected[key].items():
                                if isinstance(value, float):
                                    assert np.isclose(actual[key][metric], value, rtol=1e-7, atol=1e-9), context+"/"+metric
                                else:
                                    assert actual[key][metric] == value, context+"/"+metric
                        else:
                            assert actual[key] == expected[key], context
                    if policy != "expert":
                        assert actual["expert_calls"] == 0
                    report["replayed"].append({"platform": kind, "policy": policy, "seed": seed,
                                               "case": spec["id"], "passed": actual["passed"],
                                               "task_evidence_match": True, "exact_match": not portable,
                                               "recorded_raw_model_hash": expected["base_model_xml_sha256"],
                                               "current_raw_model_hash": actual["base_model_xml_sha256"]})
                saved_rate = record["aggregate"][policy][split]["per_seed_success_rate"]
                rate_index = 0 if seed is None else summary["training_seeds"].index(int(seed))
                assert saved_rate[rate_index] == float(np.mean([case["passed"] for case in cases]))
            print(f"{'REPLAYED' if replay else 'AUDITED STORED'} {kind} {policy} seed={seed}", flush=True)
            save_json(output, report)
    report.update(status="PASS", episodes=len(report["replayed"]), elapsed_seconds=time.perf_counter()-started,
                  verification_scope=("Data/source provenance, episode isolation, train normalization and stored summaries; no new physical replay." if not replay else
                                      "Pinned clean model assets, exact task criteria/status/counts, floating metrics rtol=1e-7 atol=1e-9; raw absolute-path XML hash reported separately." if portable else
                                      "Same installed simulator and evaluation implementation; exact stored-policy replay and data/provenance checks."))
    save_json(output, report)
    print(f"PASS {report['episodes']} physical episodes; {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--portable", action="store_true", help="Keep task decisions exact; tolerate small floating-point differences and report host-specific XML hashes")
    args = parser.parse_args()
    verify(args.run, args.output, replay=not args.skip_replay, portable=args.portable)
