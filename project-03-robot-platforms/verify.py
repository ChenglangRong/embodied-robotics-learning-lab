"""Run independent repeated baselines and specified failure controls."""
import hashlib
import argparse
import json
import datetime
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from lab import CACHE, PIN, SOURCES, Controller, build, run
from panel import FAILURE


def verify(output=None):
    actual = subprocess.check_output(["git", "-C", str(CACHE), "rev-parse", "HEAD"], text=True).strip()
    if actual != PIN:
        raise RuntimeError(f"Model revision mismatch: {actual}")
    report = {"model_commit": actual, "checks": [], "baselines": {}, "failure_controls": {},
              "python_version": sys.version, "platform": platform.platform(), "numpy_version": np.__version__,
              "validation_date": datetime.datetime.now().astimezone().isoformat(),
              "lab_source_sha256": hashlib.sha256((Path(__file__).parent / "lab.py").read_bytes()).hexdigest()}
    for kind in SOURCES:
        m, d, qa, va, _ = build(kind)
        assert m.nu == {"panda": 8, "go2": 12, "h1": 19, "allegro": 16}[kind]
        c = Controller(kind, m, d, qa, va)
        q, v = d.qpos.copy(), d.qvel.copy()
        c.step()
        np.testing.assert_array_equal(q, d.qpos)
        np.testing.assert_array_equal(v, d.qvel)
        report["checks"].append(f"{kind}: controller modifies controls; physical qpos/qvel preserved")
        report["checks"].append(f"{kind}: actuator count matches model contract")
        with tempfile.TemporaryDirectory(prefix="robot-repeat-") as tmp:
            a = run(kind, Path(tmp) / "a")
            b = run(kind, Path(tmp) / "b")
            assert a["passed"] and b["passed"], (kind, a, b)
            trace_a = (Path(tmp) / "a/trajectory.csv").read_bytes()
            trace_b = (Path(tmp) / "b/trajectory.csv").read_bytes()
            assert trace_a == trace_b, f"{kind}: repeated trajectory drift"
            baseline = Path(__file__).parent / "outputs/baseline" / kind / "trajectory.csv"
            if not baseline.exists():
                raise FileNotFoundError(f"Required saved baseline missing: {baseline}")
            assert trace_a == baseline.read_bytes(), f"{kind}: saved rendered baseline differs from fresh physics run"
            report["checks"].append(f"{kind}: baseline passes and matches required saved trajectory")
            report["checks"].append(f"{kind}: repeated trajectories are byte-identical")
            report["baselines"][kind] = {"passed": True, "byte_identical_repeat": True,
                                         "trajectory_sha256": hashlib.sha256(trace_a).hexdigest(), "result": a}
        failure = run(kind, options=FAILURE[kind])
        assert not failure["passed"], (kind, "failure control unexpectedly passed", failure)
        report["failure_controls"][kind] = {"options": FAILURE[kind], "result": failure}
        report["checks"].append(f"{kind}: specified failure control fails legacy criteria")
        print(f"PASS {kind}: model / state isolation / baseline / deterministic repeat / failure control", flush=True)
    report["check_count"] = len(report["checks"])
    report["criteria_scope"] = "legacy fixed-scene checks; diagnostics and personal learning gate are evaluated separately"
    output = Path(output) if output else Path(__file__).parent / "outputs/validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Use a new output path to preserve historical reports")
    args = parser.parse_args()
    verify(args.output)
