"""Compare matched CoM gain runs and audit delivered push and settling from raw traces."""

import argparse
import gzip
import json
from collections import defaultdict
from math import hypot
from pathlib import Path
from statistics import mean


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def indexed(records, key):
    by_key = {key(record): record for record in records}
    if len(by_key) != len(records):
        raise ValueError("duplicate case")
    return by_key


def audit_trace(rows, push, config):
    dt = config["dt"]
    onset = config["push_onset_s"]
    force = [push["force_x_n"], push["force_y_n"], 0.0]
    end = onset + push["duration_s"]
    delivered = [row for row in rows if any(row["push_force_n"])]
    for step, row in enumerate(rows, start=1):
        start = (step - 1) * dt
        expected = force if onset <= start < end else [0.0] * 3
        if row["step"] != step or abs(row["time_s"] - step * dt) > 1e-9 or row["push_force_n"] != expected:
            raise ValueError(f"unexpected time or force at step {step}")
    push_steps = sum(onset <= step * dt < end for step in range(config["horizon"])) if any(force) else 0
    full_push = len(delivered) == push_steps

    # A teaching metric. Start at actual force removal (or the nominal time for none).
    # Report the start of the final in-band interval after at least 0.4 s of evidence.
    stable_times = []
    for row in reversed(rows):
        if row["time_s"] < end - 1e-9:
            break
        if (row["fallen"] or not all(row["foot_contacts"])
                or abs(row["com_error_x_m"]) > 0.035 or abs(row["com_error_y_m"]) > 0.03
                or hypot(row["com_rate_x_m_s"], row["com_rate_y_m_s"]) > 0.12):
            break
        stable_times.append(row["time_s"])
    settled = None
    if full_push and len(rows) == config["horizon"] and stable_times and stable_times[0] - stable_times[-1] >= .4 - 1e-9:
        settled = round(stable_times[-1] - end, 9)
    return {
        "full_scheduled_push": full_push,
        "push_steps": len(delivered),
        "impulse_xy_n_s": [round(sum(row["push_force_n"][axis] for row in delivered) * dt, 9) for axis in range(2)],
        "post_release_settle_s": settled,
    }


def compare(reference, variant):
    protocols = [read(folder / "protocol.json") for folder in (reference, variant)]
    configs = [protocol["config"] for protocol in protocols]
    changed = {key for key in configs[0].keys() | configs[1].keys() if configs[0].get(key) != configs[1].get(key)}
    matched = all(protocols[0][key] == protocols[1][key] for key in (
        "model_commit", "model_assets_sha256", "source_sha256", "dependencies", "controllers", "scenarios",
        "seeds", "scenario_push", "acceptance"))
    if changed != {"com_gain"} or not matched or protocols[0]["controllers"] != ["com"]:
        raise ValueError("runs must match except for CoM proportional gain")

    def key(case):
        return case["scenario"], case["seed"]

    compared = []
    expected = {(scenario, seed) for scenario in protocols[0]["scenarios"] for seed in protocols[0]["seeds"]}
    for folder, protocol, config in zip((reference, variant), protocols, configs):
        cases = indexed(read(folder / "cases.json"), key)
        with gzip.open(folder / "trajectories.jsonl.gz", "rt", encoding="utf-8") as stream:
            traces = indexed([json.loads(line) for line in stream], lambda record: key(record["case"]))
        if cases.keys() != expected or traces.keys() != expected:
            raise ValueError("incomplete matched coverage")
        audited = {}
        for k, case in cases.items():
            rows = traces[k]["trajectory"]
            if case != traces[k]["case"] or len(rows) != case["steps"] or case["controller"] != "com":
                raise ValueError(f"case and trajectory differ for {k}")
            audited[k] = {**case, **audit_trace(rows, protocol["scenario_push"][case["scenario"]], config)}
        compared.append(audited)

    pairs = []
    grouped = defaultdict(list)
    for scenario in protocols[0]["scenarios"]:
        for seed in protocols[0]["seeds"]:
            pair = {"scenario": scenario, "seed": seed, "reference": compared[0][scenario, seed],
                    "variant": compared[1][scenario, seed]}
            pairs.append(pair)
            grouped[scenario].append(pair)

    def summarize(cases):
        result = {"cases": len(cases), "successes": sum(case["success"] for case in cases),
                  "full_scheduled_pushes": sum(case["full_scheduled_push"] for case in cases)}
        for field in ("recovery_time_s", "post_release_settle_s", "push_peak_com_error_m", "final_com_error_m",
                      "mean_contact_slip_m_s", "saturation_fraction", "max_abs_ankle_correction_rad"):
            values = [case[field] for case in cases if case[field] is not None]
            result[field] = {"mean": mean(values) if values else None, "measured": len(values)}
        return result

    return {
        "reference_gain": configs[0]["com_gain"], "variant_gain": configs[1]["com_gain"],
        "fixed_rate_gain": configs[0]["com_rate_gain"], "matched_pairs": len(pairs),
        "technical_metric": "Time from scheduled force removal to start of final in-band interval with at least 0.4 s attendance through episode end. In band: both feet at sampled steps, |ex| <= .035 m, |ey| <= .03 m, pelvis xy speed <= .12 m/s. For none the time origin is nominal force removal.",
        "by_scenario": {scenario: {arm: summarize([pair[arm] for pair in selected]) for arm in ("reference", "variant")}
                        for scenario, selected in grouped.items()},
        "pairs": pairs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(args.output.resolve().is_relative_to(folder.resolve()) for folder in (args.reference, args.variant)):
        parser.error("write comparison outside both experiment directories")
    result = compare(args.reference, args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Compared {result['matched_pairs']} matched pairs; report: {args.output.name}")
