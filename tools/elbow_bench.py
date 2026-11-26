"""Prepare and summarize a manual, quasi-static elbow bench measurement.

Force is the signed external tangential force on the moving arm. Flexion is
positive, including during extension. Run each condition in its own CSV.
"""

import argparse
import csv
import math
from pathlib import Path
from statistics import median


ANGLES = (45, 60, 75, 90, 105)
FORCE_FIELDS = [f"force_{angle}_deg_n" for angle in ANGLES]
FIELDS = ["cycle", "direction", *FORCE_FIELDS]
DIRECTIONS = {"flex": 1, "extend": -1}


def make_template(path):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for cycle in range(1, 6):
            for direction in DIRECTIONS:
                writer.writerow([cycle, direction, *([""] * len(ANGLES))])


def read_passes(path, radius):
    passes = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"expected columns: {', '.join(FIELDS)}")
        for row in reader:
            cycle = int(row["cycle"])
            direction = row["direction"]
            if cycle < 1 or direction not in DIRECTIONS or (cycle, direction) in passes:
                raise ValueError(f"row {reader.line_num}: invalid or repeated cycle/direction")
            if None in row or any(not row[field] for field in FORCE_FIELDS):
                raise ValueError(f"row {reader.line_num}: fill every force cell")
            torques = [float(row[field]) * radius for field in FORCE_FIELDS]
            if not all(math.isfinite(torque) for torque in torques):
                raise ValueError(f"row {reader.line_num}: expected finite forces")
            passes[cycle, direction] = torques
    if not passes:
        raise ValueError("no measurements")
    for cycle, _ in passes:
        if any((cycle, direction) not in passes for direction in DIRECTIONS):
            raise ValueError(f"cycle {cycle}: both directions are required")
    return passes


def summarize(passes):
    print("direction,cycles,median_motion_opposing_torque_nm,median_signed_input_work_j,"
          "angular_torque_span_nm")
    for direction, sign in DIRECTIONS.items():
        strokes = []
        work = []
        for (_, label), torques in passes.items():
            if label != direction:
                continue
            opposing = [sign * torque for torque in torques]
            strokes.append(opposing)
            stroke_work = 0.0
            for index in range(1, len(ANGLES)):
                mean_torque = (opposing[index - 1] + opposing[index]) / 2
                angle_change = math.radians(ANGLES[index] - ANGLES[index - 1])
                stroke_work += mean_torque * angle_change
            work.append(stroke_work)
        angle_medians = [median(values) for values in zip(*strokes)]
        torque_span = max(angle_medians) - min(angle_medians)
        midpoint = median(torque for stroke in strokes for torque in stroke)
        print(f"{direction},{len(work)},{midpoint:.4f},{median(work):.4f},{torque_span:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="one measured condition or a new template path")
    parser.add_argument("--template", action="store_true", help="write a blank five-cycle CSV")
    parser.add_argument("--radius-m", type=float, default=0.20, help="perpendicular lever arm")
    args = parser.parse_args()
    try:
        if args.template:
            make_template(args.csv)
        else:
            if not math.isfinite(args.radius_m) or args.radius_m <= 0:
                raise ValueError("radius must be finite and positive")
            summarize(read_passes(args.csv, args.radius_m))
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
