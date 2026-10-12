"""Independently check the live MuJoCo -> ROS 2 -> TF geometry chain."""

from __future__ import annotations

import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from tf2_ros import Buffer, TransformException, TransformListener


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--geometry-tolerance", type=float, default=1e-6)
    parser.add_argument("--goal-tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    rclpy.init()
    node = Node("robot_lab_bridge_verifier")
    latest: dict[str, object] = {}
    node.create_subscription(JointState, "/joint_states", lambda m: latest.__setitem__("joint", m), 10)
    node.create_subscription(PointStamped, "/robot_lab/tool_point", lambda m: latest.__setitem__("tool", m), 10)
    node.create_subscription(PointStamped, "/robot_lab/goal_point", lambda m: latest.__setitem__("goal", m), 10)
    node.create_subscription(Float64, "/robot_lab/error_m", lambda m: latest.__setitem__("error", m), 10)
    node.create_subscription(String, "/robot_lab/status", lambda m: latest.__setitem__("status", m), 10)
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)
    deadline = time.monotonic() + args.timeout
    terminal_since: float | None = None

    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if not all(k in latest for k in ("joint", "tool", "goal", "error", "status")):
                continue
            if latest["status"].data not in ("success", "max_steps"):
                continue
            if terminal_since is None:
                terminal_since = time.monotonic()
            # Allow TF and direct topics to receive the same stationary final pose.
            if time.monotonic() - terminal_since < 0.3:
                continue
            try:
                transform = tf_buffer.lookup_transform("base_link", "tool_link", Time())
            except TransformException:
                continue
            break
        else:
            raise RuntimeError(f"Timed out waiting for complete terminal state and TF; received={list(latest)}")

        joint = latest["joint"]
        tool = latest["tool"]
        goal = latest["goal"]
        reported_error = float(latest["error"].data)
        if joint.name != ["shoulder", "elbow"] or len(joint.position) != 2:
            raise RuntimeError(f"Unexpected joint order: {joint.name}")
        if tool.header.frame_id != "base_link" or goal.header.frame_id != "base_link":
            raise RuntimeError("Tool and goal points must be expressed in base_link")

        q1, q2 = joint.position
        # These are the independent, hand-readable link lengths in arm2d.urdf.
        fk_x = 0.6 * math.cos(q1) + 0.4 * math.cos(q1 + q2)
        fk_y = 0.6 * math.sin(q1) + 0.4 * math.sin(q1 + q2)
        tf_point = transform.transform.translation
        tf_gap = math.dist((tool.point.x, tool.point.y, tool.point.z),
                           (tf_point.x, tf_point.y, tf_point.z))
        fk_gap = math.dist((tool.point.x, tool.point.y, tool.point.z), (fk_x, fk_y, 0.0))
        goal_error = math.dist((tool.point.x, tool.point.y), (goal.point.x, goal.point.y))
        error_gap = abs(goal_error - reported_error)
        result = {
            "status": latest["status"].data,
            "joint_rad": {"shoulder": q1, "elbow": q2},
            "tool_m": [tool.point.x, tool.point.y, tool.point.z],
            "tf_tool_m": [tf_point.x, tf_point.y, tf_point.z],
            "goal_m": [goal.point.x, goal.point.y],
            "goal_error_m": goal_error,
            "reported_error_m": reported_error,
            "tf_gap_m": tf_gap,
            "forward_kinematics_gap_m": fk_gap,
            "reported_error_gap_m": error_gap,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if (result["status"] != "success" or goal_error > args.goal_tolerance
                or max(tf_gap, fk_gap, error_gap) > args.geometry_tolerance):
            raise SystemExit(1)
    finally:
        del tf_listener
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
