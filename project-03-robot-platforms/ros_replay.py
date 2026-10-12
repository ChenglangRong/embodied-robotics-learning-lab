"""ROS 2 telemetry replay. Requires Ubuntu ROS 2; sends no robot commands."""
import argparse
import csv
import json
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped


def replay(folder, validate=False):
    config = json.loads((folder / "config.json").read_text())
    with (folder / "trajectory.csv").open() as f:
        rows = list(csv.DictReader(f))
    kind = config["task"]
    names = [n.replace(" (tendon)", "") for n in config["joint_names"]]
    node = Node(f"robotlab_{kind}_telemetry")
    pub = node.create_publisher(JointState, f"/robotlab/{kind}/joint_states", 10)
    point_pub = node.create_publisher(PointStamped, f"/robotlab/{kind}/tracked_point", 10)
    received = {"joint_states": 0, "tracked_point": 0}
    def on_joint(msg):
        assert len(msg.name) == len(msg.position) == len(msg.velocity)
        received["joint_states"] += 1
    def on_point(msg):
        received["tracked_point"] += 1
    # Validation checks serialized ROS publications returning through subscriptions.
    subscribers = []
    if validate:
        subscribers = [node.create_subscription(JointState, f"/robotlab/{kind}/joint_states", on_joint, 10),
                       node.create_subscription(PointStamped, f"/robotlab/{kind}/tracked_point", on_point, 10)]
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.02)
    count = min(50, len(rows)) if validate else len(rows)
    for row in rows[:count]:
        msg = JointState()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(row[f"joint_{i}_q"]) for i in range(len(names))]
        msg.velocity = [float(row[f"joint_{i}_dq"]) for i in range(len(names))]
        if kind == "panda":
            msg.name.append("finger_joint2")
            msg.position.append(msg.position[-1])
            msg.velocity.append(msg.velocity[-1])
        pub.publish(msg)
        point = PointStamped()
        point.header.stamp = msg.header.stamp
        point.header.frame_id = "world"
        point.point.x, point.point.y, point.point.z = (float(row[k]) for k in ["x_m", "y_m", "z_m"])
        point_pub.publish(point)
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=.003 if validate else .006)
    deadline = time.monotonic() + .2
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.01)
    result = {"robot": kind, "published_per_topic": count, "received": received, "joint_count": len(msg.name),
              "passed": all(v == count for v in received.values()) if validate else None,
              "scope": "recorded-state telemetry replay; reception check; no TF/model or command bridge"}
    node.destroy_node()
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, help="Experiment output directory; use outputs/baseline for --all")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    rclpy.init()
    try:
        folders = [args.folder / k for k in ["panda", "go2", "h1", "allegro"]] if args.all else [args.folder]
        results = [replay(f, args.validate) for f in folders]
        print(json.dumps(results, indent=2))
        if args.report:
            args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
        if args.validate and not all(r["passed"] for r in results):
            raise SystemExit(1)
    finally:
        rclpy.shutdown()
