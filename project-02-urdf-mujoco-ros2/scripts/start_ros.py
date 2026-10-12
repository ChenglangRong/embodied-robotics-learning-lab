"""Start robot_state_publisher and the MuJoCo bridge with one shared URDF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--goal-x", type=float, default=0.55)
    parser.add_argument("--goal-y", type=float, default=0.55)
    args = parser.parse_args()
    urdf = (ROOT / "model" / "arm2d.urdf").read_text(encoding="utf-8")
    description = LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": urdf, "publish_frequency": 50.0}],
             output="screen"),
        ExecuteProcess(
            cmd=[sys.executable, str(ROOT / "scripts" / "ros_sim_node.py"),
                 "--seed", str(args.seed), "--goal-x", str(args.goal_x),
                 "--goal-y", str(args.goal_y)],
            output="screen",
        ),
    ])
    service = LaunchService()
    service.include_launch_description(description)
    service.run()


if __name__ == "__main__":
    main()
