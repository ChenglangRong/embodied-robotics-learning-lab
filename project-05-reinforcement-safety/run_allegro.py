"""Run, render or verify fixed-wrist Allegro experiments."""
import argparse
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SEEDS = list(range(5000000, 5000005))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(folder):
    return {path.relative_to(folder).as_posix(): sha(path) for path in sorted(folder.rglob("*"))
            if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts}


def combinations(core, seeds):
    for task in core.TASKS:
        for mode in (core.ROLL_MODES if task == "reorient" else core.SLIP_MODES):
            for seed in seeds:
                yield task, mode, seed


def summary(cases):
    groups = {}
    for case in cases:
        stats = groups.setdefault(f"{case['task']}/{case['mode']}", {"runs": 0, "passed": 0})
        stats["runs"] += 1
        stats["passed"] += int(case["success"])
    return {"runs": len(cases), "failures": sum(not c["success"] for c in cases), "groups": groups}


def assets(core):
    directory = core.lab.CACHE / "wonik_allegro"
    return {p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def record(folder, core, gain, seeds):
    revision = subprocess.check_output(["git", "-C", str(core.lab.CACHE), "rev-parse", "HEAD"], text=True).strip()
    if revision != core.lab.PIN:
        raise ValueError("Model cache must be at the pinned revision")
    folder.mkdir(parents=True, exist_ok=False)
    source = folder / "source"
    source.mkdir()
    for path in (Path(core.__file__), core.LAB_FILE, Path(__file__)):
        shutil.copyfile(path, source / path.name)
    protocol = {"model_commit": revision, "model_assets_sha256": assets(core),
                "source_sha256": {p.name: sha(p) for p in sorted(source.iterdir())},
                "dependencies": {"mujoco": core.mujoco.__version__, "numpy": core.np.__version__,
                                 "python": sys.version.split()[0]},
                "seeds": seeds, "development_seeds": [0, 11, 12], "gain_rad_per_m": gain,
                "load_n": core.TASKS, "target_axis_palm": core.ROLL_AXIS, "target_angle_deg": core.ROLL_TARGET_DEG,
                "physics_dt_s": .002, "control_dt_s": core.DT, "episode_steps": core.HORIZON,
                "preparation_duration_s": 9., "pull_interval_s": [1., 1.3],
                "reset": "object XY uniform +/-1 mm, finger joints uniform +/-4 mrad, common 9 s preparation",
                "scope": "palm-assisted caging, one shaped target, no learning updates; stress17 is a harder matched load"}
    save(folder / "protocol.json", protocol)
    hand = core.Hand()
    cases = []
    try:
        with gzip.open(folder / "trajectories.jsonl.gz", "wt", encoding="utf-8", newline="\n") as stream:
            for key in combinations(core, seeds):
                case, rows = core.episode(hand, *key, gain)
                cases.append(case)
                stream.write(json.dumps({"case": case, "rows": rows}, allow_nan=False) + "\n")
                print(*key, case["success"], flush=True)
    finally:
        hand.close()
    save(folder / "cases.json", cases)
    save(folder / "summary.json", summary(cases))
    save(folder / "manifest.json", manifest(folder))


def compare(actual, saved, np, stats, label):
    if isinstance(saved, dict):
        if actual.keys() != saved.keys():
            raise ValueError(f"field mismatch: {label}")
        for k in saved:
            compare(actual[k], saved[k], np, stats, f"{label}.{k}")
    elif isinstance(saved, list):
        if len(actual) != len(saved):
            raise ValueError(f"length mismatch: {label}")
        for i, (a, b) in enumerate(zip(actual, saved)):
            compare(a, b, np, stats, f"{label}[{i}]")
    elif isinstance(saved, float):
        difference = abs(actual - saved)
        limit = 3e-6 + 1e-6 * abs(saved)
        if not np.isfinite(difference) or difference > limit:
            raise ValueError(f"numeric mismatch: {label}, {actual} vs {saved}, limit {limit}")
        stats["float_values"] += 1
        stats["max_difference"] = max(stats["max_difference"], difference)
    elif type(actual) is not type(saved) or actual != saved:
        raise ValueError(f"discrete mismatch: {label}, {actual} vs {saved}")


def verify(folder):
    if manifest(folder) != read(folder / "manifest.json"):
        raise ValueError("artifact hashes differ")
    protocol = read(folder / "protocol.json")
    for name, digest in protocol["source_sha256"].items():
        if sha(folder / "source" / name) != digest:
            raise ValueError(f"source hash differs: {name}")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(folder / "source"))
    core = importlib.import_module("allegro_lab")
    if protocol["model_commit"] != core.lab.PIN or assets(core) != protocol["model_assets_sha256"]:
        raise ValueError("pinned model assets differ")
    if [core.mujoco.__version__, core.np.__version__] != [protocol["dependencies"][k] for k in ("mujoco", "numpy")]:
        raise ValueError("physical runtime differs from frozen dependencies")
    cases = read(folder / "cases.json")
    if summary(cases) != read(folder / "summary.json"):
        raise ValueError("summary differs from cases")
    stats = {"float_values": 0, "max_difference": 0.}
    hand = core.Hand()
    try:
        with gzip.open(folder / "trajectories.jsonl.gz", "rt", encoding="utf-8") as stream:
            for case, key, stored in zip(cases, combinations(core, protocol["seeds"]), map(json.loads, stream), strict=True):
                if case != stored["case"] or key != (case["task"], case["mode"], case["seed"]):
                    raise ValueError("recorded case key or metrics differ")
                if case != core.assess(*key, stored["rows"]):
                    raise ValueError("stored case metrics do not follow the acceptance criteria")
                actual, rows = core.episode(hand, *key, protocol["gain_rad_per_m"])
                compare(actual, case, core.np, stats, str(key))
                compare(rows, stored["rows"], core.np, stats, str(key))
    finally:
        hand.close()
    return {"status": "PASS", "replayed_cases": len(cases), "retained_failures": summary(cases)["failures"],
            "atol": 3e-6, "rtol": 1e-6, **stats}


def make_demo(folder, core, seed, evidence):
    protocol = read(evidence / "protocol.json")
    gain = protocol["gain_rad_per_m"]
    keys = [(task, mode, seed) for task, mode in (("pull13", "hold"), ("pull13", "feedback"),
                                                  ("reorient", "hold"), ("reorient", "roll"))]
    with gzip.open(evidence / "trajectories.jsonl.gz", "rt", encoding="utf-8") as stream:
        references = {(a["task"], a["mode"], a["seed"]): entry for entry in map(json.loads, stream)
                      for a in (entry["case"],) if (a["task"], a["mode"], a["seed"]) in keys}
    folder.mkdir(parents=True, exist_ok=False)
    font = ImageFont.load_default(size=17)
    hand = core.Hand()
    try:
        for left_key, right_key in zip(keys[::2], keys[1::2], strict=True):
            task = left_key[0]
            clips = []
            for key in (left_key, right_key):
                frames = []

                def capture(env, row):
                    if row["step"] % 5 == 0:
                        frame = Image.fromarray(env.render())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0, 0, 640, 63), fill="#152530")
                        draw.text((12, 5), f"{task} / {key[1]}   t={row['time_s']:.2f}s   seed={seed}", font=font)
                        draw.text((12, 26), f"Fy={row['force_world_n'][1]:.0f} N   drift={1000*row['drift_y_m']:+.1f} mm   goal error={row['target_error_deg']:.1f} deg", font=font)
                        draw.text((12, 45), f"Loaded fingers={row['finger_count']}   thumb offset={row['thumb_correction_rad']:+.2f} rad", font=font)
                        frames.append(frame)
                    return True

                case, rows = core.episode(hand, *key, gain, on_step=capture)
                expected = references[key]
                stats = {"float_values": 0, "max_difference": 0.}
                compare(case, expected["case"], core.np, stats, str(key))
                compare(rows, expected["rows"], core.np, stats, str(key))
                clips.append(frames)
            paired = []
            for left, right in zip(*clips, strict=True):
                frame = Image.new("RGB", (1280, 480))
                frame.paste(left, (0, 0))
                frame.paste(right, (640, 0))
                paired.append(frame)
            paired[0].save(folder / f"{task}.gif", save_all=True, append_images=paired[1:], duration=100, loop=0)
            paired[12 if task == "pull13" else 29].save(folder / f"{task}.png")
    finally:
        hand.close()


def live(core, task, mode, seed, gain):
    from mujoco import viewer

    hand = core.Hand()
    try:
        with viewer.launch_passive(hand.m, hand.d) as handle:
            handle.cam.lookat[:] = [.025, 0, .33]
            handle.cam.distance = .48
            handle.cam.azimuth, handle.cam.elevation = 145, -30

            def show(env, row):
                if not handle.is_running():
                    return False
                with handle.lock():
                    handle.user_scn.ngeom = 0
                    core.mark_pose(handle.user_scn, env.d.xpos[env.obj], env.d.xmat[env.obj])
                handle.sync()
                time.sleep(core.DT)
                return True

            result, _ = core.episode(hand, task, mode, seed, gain, on_step=show)
            print(json.dumps(result, indent=2), flush=True)
            while handle.is_running():
                time.sleep(.1)
    finally:
        hand.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser("run")
    run.add_argument("output", type=Path)
    run.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    run.add_argument("--gain", type=float, default=28.)
    check = subs.add_parser("verify")
    check.add_argument("evidence", type=Path)
    check.add_argument("--report", type=Path, required=True)
    demo = subs.add_parser("demo")
    demo.add_argument("evidence", type=Path)
    demo.add_argument("output", type=Path)
    demo.add_argument("--seed", type=int, default=SEEDS[0])
    view = subs.add_parser("live")
    view.add_argument("task", default="reorient", nargs="?")
    view.add_argument("mode", default="roll", nargs="?")
    view.add_argument("--seed", type=int, default=SEEDS[0])
    view.add_argument("--gain", type=float, default=28.)
    args = parser.parse_args()
    if args.command == "verify":
        if args.report.resolve().is_relative_to(args.evidence.resolve()):
            parser.error("keep the verification report outside the frozen directory")
        result = verify(args.evidence.resolve())
        args.report.parent.mkdir(parents=True, exist_ok=True)
        save(args.report, result)
        print(json.dumps(result, indent=2))
        return
    import allegro_lab as core
    if args.command == "run":
        if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds) or not core.np.isfinite(args.gain) or args.gain < 0:
            parser.error("provide unique nonnegative seeds and a finite nonnegative gain")
        record(args.output, core, args.gain, args.seeds)
    elif args.command == "demo":
        make_demo(args.output, core, args.seed, args.evidence)
    else:
        live(core, args.task, args.mode, args.seed, args.gain)


if __name__ == "__main__":
    main()
