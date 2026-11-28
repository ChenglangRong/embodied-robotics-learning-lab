"""Animate recorded MuJoCo states and goal errors on the same simulation clock."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from closed_loop_lab import tips


def render(evidence, output, split, seed, names):
    records = {}
    with gzip.open(evidence / "trajectories.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            case = record["case"]
            if case["split"] == split and case["seed"] == seed and case["method"] in names:
                records[case["method"]] = record
    if set(records) != set(names):
        raise ValueError("No complete matched comparison for the requested reset")
    width, height = 388, 395
    white, muted, grid = "#edf5ff", "#9fb3c8", "#32485e"
    colors = ("#59c8ff", "#f6b978", "#68e6aa")
    title = ImageFont.load_default(size=18)
    text = ImageFont.load_default(size=15)
    longest = max(len(r["steps"]) for r in records.values())
    largest_error = max(row["goal_error_m"] for r in records.values() for row in r["steps"])
    timeline = list(range(0, longest, 5)) + [longest, longest]

    def pixel(xy, offset):
        return offset + 165 + float(xy[0])*113, 186-float(xy[1])*113

    def plot_point(step, error, offset):
        return offset + 48 + 310*step/longest, 353 - 55*error/max(largest_error, .045)

    frames = []
    for tick in timeline:
        image = Image.new("RGB", (width*len(names), height), "#101c29")
        draw = ImageDraw.Draw(image)
        for i, name in enumerate(names):
            offset = i*width
            record = records[name]
            case, rows = record["case"], record["steps"]
            step = min(tick, len(rows))
            q = rows[step-1]["x_next"][:2] if step else case["initial"][:2]
            elbow = .6*np.array([np.cos(q[0]), np.sin(q[0])])
            tip = tips(np.asarray(q))
            position = [pixel(p, offset) for p in ([0, 0], elbow, tip)]
            center = pixel(case["goal"], offset)
            draw.text((offset+14, 10), name, fill=colors[i], font=title)
            draw.text((offset+14, 36), f"{split} {seed}  |  sim {tick*.02:.2f}s", fill=muted, font=text)
            draw.line((offset+10, 186, offset+378, 186), fill=grid)
            draw.line((offset+165, 64, offset+165, 298), fill=grid)
            x, y = center
            draw.ellipse((x-8, y-8, x+8, y+8), outline=white, width=2)
            draw.line(position, fill=colors[i], width=10)
            for x, y in position:
                draw.ellipse((x-5, y-5, x+5, y+5), fill=white)
            elapsed = step*.02
            done = case["termination"] if step == len(rows) else "running"
            error = rows[step-1]["goal_error_m"] if step else float(np.linalg.norm(tip-case["goal"]))
            draw.text((offset+14, 266), f"{elapsed:.2f}s  |  {1000*error:.1f} mm  |  {done}", fill=white, font=text)
            draw.line((plot_point(0, .045, offset), plot_point(longest, .045, offset)), fill=muted, width=1)
            xs = [plot_point(r["step"], r["goal_error_m"], offset) for r in rows[:step]]
            if len(xs) >= 2:
                draw.line(xs, fill=colors[i], width=2)
            draw.text((offset+14, 365), "Goal error  |  gray line = 45 mm", fill=muted, font=text)
        frames.append(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=115, loop=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("test", "ood"), default="test")
    parser.add_argument("--seed", type=int, default=4500)
    parser.add_argument("--methods", nargs=3, default=["mpc_masked", "mpc_action", "dls"])
    args = parser.parse_args()
    if len(set(args.methods)) != 3 or args.output.resolve().is_relative_to(args.evidence.resolve()):
        parser.error("use three distinct methods and keep rendered media outside the frozen directory")
    render(args.evidence, args.output, args.split, args.seed, args.methods)
