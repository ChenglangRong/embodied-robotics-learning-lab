"""Matched visible success/failure examples from immutable stored weights."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tabular_lab import Config, draw_arm, rollout, save_csv, save_json


def make(folder, output):
    if output.exists():
        raise FileExistsError("Use a new showcase directory")
    output.mkdir(parents=True)
    cfg = Config(**json.loads((folder/"config.json").read_text(encoding="utf-8")))
    learned = np.load(folder/"q-seed-0.npz")["q"]
    oracle = np.load(folder/"oracle.npz")["q"]
    cases = [("learned-success", learned, 210, None), ("learned-failure", learned, 0, None),
             ("untrained-matched", np.zeros_like(learned), 210, None),
             ("random-matched", np.zeros_like(learned), 210, 10210),
             ("oracle-matched", oracle, 210, None)]
    records = []
    for name, q, start, rng in cases:
        rows, result = rollout(cfg, q, start, random_seed=rng)
        save_csv(output/f"{name}.csv", rows)
        images = [draw_arm(cfg, r, name) for r in rows]
        images[0].save(output/f"{name}.gif", save_all=True, append_images=images[1:], duration=140, loop=0)
        records.append({"name": name, "random_seed": rng, **result})
    save_json(output/"selection.json", {"cases": records, "selection_rule": "Teaching starts 210 and 0 were selected after inspecting baseline outcomes; performance uses exhaustive evaluation, and these demos provide illustration only.",
               "weights_sha256": hashlib.sha256((folder/"q-seed-0.npz").read_bytes()).hexdigest(),
               "showcase_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    make(args.folder, args.output)
