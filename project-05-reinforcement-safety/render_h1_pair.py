"""Render a synchronized H1 gain comparison validated against saved physics traces."""

import argparse
import gzip
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from h1_recovery import Config, H1Stand, LAB_FILE, ROOT
from verify_h1 import equal_trajectory, read, sha


def replay_frames(folder, scenario, seed):
    protocol = read(folder / "protocol.json")
    for path in (ROOT / "h1_recovery.py", LAB_FILE, ROOT / "tabular_lab.py"):
        if sha(path) != protocol["source_sha256"][path.name]:
            raise ValueError(f"Current {path.name} differs from the recorded source")

    with gzip.open(folder / "trajectories.jsonl.gz", "rt", encoding="utf-8") as stream:
        record = next((record for record in map(json.loads, stream)
                       if (record["case"]["controller"], record["case"]["scenario"], record["case"]["seed"])
                       == ("com", scenario, seed)), None)
    if record is None:
        raise ValueError(f"No saved com episode for {scenario}/{seed}")

    frames = []
    env = H1Stand(Config(**protocol["config"]))
    try:
        if len(record["trajectory"]) != env.cfg.horizon:
            raise ValueError("This synchronized demonstration requires a complete recorded episode")
        env.reset(seed)
        for observed in record["trajectory"]:
            row = env.step("com", scenario)
            equal_trajectory(row, observed, f"step{row['step']}")
            if row["step"] % 5 == 0:
                frames.append((row, env.render()))
    finally:
        env.close()
    return protocol, frames


def render_pair(reference, variant, output, still, scenario, seed):
    # Select a pair by recorded scenario and seed and validate each trajectory before encoding.
    left, left_frames = replay_frames(reference, scenario, seed)
    right, right_frames = replay_frames(variant, scenario, seed)
    config_difference = {key for key, value in left["config"].items() if value != right["config"][key]}
    matching_fields = ("model_commit", "model_assets_sha256", "scenario_push", "source_sha256", "dependencies")
    if config_difference != {"com_gain"} or any(left[key] != right[key] for key in matching_fields):
        raise ValueError("The recordings must differ only in proportional gain")

    frames = []
    title = ImageFont.load_default(size=19)
    detail = ImageFont.load_default(size=16)
    heading_color = "#e5e7eb"
    for (lrow, lpic), (rrow, rpic) in zip(left_frames, right_frames, strict=True):
        if lrow["time_s"] != rrow["time_s"] or lrow["push_force_n"] != rrow["push_force_n"]:
            raise ValueError("The frames must share their time and force")
        frame = Image.new("RGB", (1280, 586), "#17242e")
        frame.paste(Image.fromarray(lpic), (0, 66))
        frame.paste(Image.fromarray(rpic), (640, 66))
        draw = ImageDraw.Draw(frame)
        force = lrow["push_force_n"]
        draw.text((16, 8), f"H1  |  {scenario}  |  reset {seed}  |  t = {lrow['time_s']:.2f} s  |  "
                  f"last step: Fx = {force[0]:+.0f} N, Fy = {force[1]:+.0f} N", fill=heading_color, font=title)
        for offset, protocol, row in ((0, left, lrow), (640, right, rrow)):
            cfg = protocol["config"]
            draw.text((offset + 16, 36), f"Kc = {cfg['com_gain']:g}  |  Kr = {cfg['com_rate_gain']:g}",
                      fill=heading_color, font=title)
            draw.text((offset + 16, 556), f"CoM x error {100*row['com_error_x_m']:+.1f} cm  |  "
                      f"ankle correction {row['ankle_correction_rad']:+.2f} rad  |  feet {sum(row['foot_contacts'])}/2",
                      fill=heading_color, font=detail)
        draw.line((639, 33, 639, 585), fill="#5f7383")
        frames.append(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=100, loop=0)
    if still is not None:
        frames[min(28, len(frames) - 1)].save(still)
    print(f"Rendered {len(frames)} paired frames after validating {5*len(frames)} state samples per episode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--still", type=Path)
    parser.add_argument("--scenario", default="strong_front")
    parser.add_argument("--seed", type=int, default=3000014)
    args = parser.parse_args()
    for destination in (args.output, args.still):
        if destination is not None and (destination.exists() or any(destination.resolve().is_relative_to(path.resolve())
                                                                   for path in (args.reference, args.variant))):
            parser.error("use new media paths outside the recorded experiment folders")
    render_pair(args.reference, args.variant, args.output, args.still, args.scenario, args.seed)
