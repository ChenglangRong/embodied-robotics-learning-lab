"""Summarize a recorded ROS 2 episode, including statuses and simulated time."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from std_msgs.msg import Float64, String


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="directory containing metadata.yaml and the MCAP file")
    args = parser.parse_args()
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(args.bag.resolve()), storage_id="mcap"),
                ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"))

    topic_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    sim_times: list[float] = []
    reset_transitions = 0
    previous_sim_time: float | None = None
    while reader.has_next():
        topic, serialized, _ = reader.read_next()
        topic_counts[topic] += 1
        if topic == "/robot_lab/status":
            status_counts[deserialize_message(serialized, String).data] += 1
        elif topic == "/robot_lab/sim_time_s":
            sim_time = float(deserialize_message(serialized, Float64).data)
            if previous_sim_time is not None and sim_time < previous_sim_time - 0.01:
                reset_transitions += 1
            sim_times.append(sim_time)
            previous_sim_time = sim_time

    result = {
        "bag": str(args.bag.resolve()),
        "messages": sum(topic_counts.values()),
        "topic_counts": dict(sorted(topic_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "simulation_time_min_s": min(sim_times) if sim_times else None,
        "simulation_time_max_s": max(sim_times) if sim_times else None,
        "reset_transitions": reset_transitions,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if (status_counts["running"] == 0 or status_counts["success"] == 0
            or topic_counts["/tf"] == 0 or topic_counts["/tf_static"] == 0
            or reset_transitions == 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
