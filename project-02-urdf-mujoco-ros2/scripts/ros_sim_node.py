"""Publish a MuJoCo two-link simulation into ROS 2.

Run through start_ros.py after sourcing ROS 2 Jazzy in Ubuntu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, String
from std_srvs.srv import Trigger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "project-01-simulation-software" / "src"))
from robot_lab.planar_arm import Config, TwoLinkReachEnv, dls_action
from run_mujoco import build_sim_model, tip_from_mujoco


class MuJoCoBridge(Node):
    def __init__(self, seed: int, goal: tuple[float, float]):
        super().__init__("robot_lab_mujoco_bridge")
        self.cfg = Config()
        self.seed = seed
        self.goal = goal
        self.model = build_sim_model(ROOT / "outputs" / "ros-run" / "derived_scene.xml", goal)
        self.data = mujoco.MjData(self.model)
        self.controller = TwoLinkReachEnv(self.cfg)
        self.physics_dt = float(self.model.opt.timestep)
        self.substeps = round(self.cfg.dt / self.physics_dt)
        if self.substeps < 1 or not np.isclose(self.substeps * self.physics_dt, self.cfg.dt):
            raise RuntimeError("control period must be an exact multiple of physics timestep")
        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.tool_pub = self.create_publisher(PointStamped, "robot_lab/tool_point", 10)
        self.goal_pub = self.create_publisher(PointStamped, "robot_lab/goal_point", 10)
        self.error_pub = self.create_publisher(Float64, "robot_lab/error_m", 10)
        self.ctrl_pub = self.create_publisher(Float64MultiArray, "robot_lab/ctrl_nm", 10)
        self.sim_time_pub = self.create_publisher(Float64, "robot_lab/sim_time_s", 10)
        self.status_pub = self.create_publisher(String, "robot_lab/status", 10)
        self.create_service(Trigger, "robot_lab/reset", self.reset_service)
        self.reset_simulation()
        self.create_timer(self.cfg.dt, self.on_tick)

    def reset_simulation(self) -> None:
        self.controller.reset(seed=self.seed, goal=self.goal)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.controller.q
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.status = "running"
        self.get_logger().info(f"reset seed={self.seed}, goal={self.goal}")

    def reset_service(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self.reset_simulation()
        response.success = True
        response.message = "episode reset with the same seed and goal"
        return response

    def on_tick(self) -> None:
        if self.status == "running":
            self.controller.q = self.data.qpos.copy()
            desired_qdot = np.clip(dls_action(self.controller),
                                   -self.cfg.velocity_limit, self.cfg.velocity_limit)
            for _ in range(self.substeps):
                self.data.ctrl[:] = np.clip(
                    1.5 * (desired_qdot - self.data.qvel) - 0.2 * self.data.qvel,
                    -3.0, 3.0,
                )
                mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.steps += 1
            error = float(np.linalg.norm(np.asarray(self.goal) - tip_from_mujoco(self.data)[:2]))
            if error <= self.cfg.tolerance:
                self.status = "success"
                self.get_logger().info(f"success after {self.steps} steps, error={error:.6f} m")
            elif self.steps >= self.cfg.max_steps:
                self.status = "max_steps"
                self.get_logger().info(f"max_steps after {self.steps} steps, error={error:.6f} m")
        self.publish_state()

    def publish_state(self) -> None:
        stamp = self.get_clock().now().to_msg()
        joint = JointState()
        joint.header.stamp = stamp
        joint.name = ["shoulder", "elbow"]
        joint.position = [float(x) for x in self.data.qpos]
        joint.velocity = [float(x) for x in self.data.qvel]
        self.joint_pub.publish(joint)

        tool = PointStamped()
        tool.header.stamp = stamp
        tool.header.frame_id = "base_link"
        tool.point.x, tool.point.y, tool.point.z = [float(x) for x in tip_from_mujoco(self.data)]
        self.tool_pub.publish(tool)

        goal = PointStamped()
        goal.header.stamp = stamp
        goal.header.frame_id = "base_link"
        goal.point.x, goal.point.y, goal.point.z = self.goal[0], self.goal[1], 0.0
        self.goal_pub.publish(goal)

        error = Float64()
        error.data = float(np.linalg.norm(np.asarray(self.goal) - tip_from_mujoco(self.data)[:2]))
        self.error_pub.publish(error)

        ctrl = Float64MultiArray()
        ctrl.data = [float(x) for x in self.data.ctrl]
        self.ctrl_pub.publish(ctrl)

        sim_time = Float64()
        sim_time.data = float(self.data.time)
        self.sim_time_pub.publish(sim_time)

        status = String()
        status.data = self.status
        self.status_pub.publish(status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--goal-x", type=float, default=0.55)
    parser.add_argument("--goal-y", type=float, default=0.55)
    args = parser.parse_args()
    rclpy.init()
    node = MuJoCoBridge(args.seed, (args.goal_x, args.goal_y))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
