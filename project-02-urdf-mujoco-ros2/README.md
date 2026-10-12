# 项目 02：一份 URDF，连接 MuJoCo 与 ROS 2

目标：让一台二连杆机械臂在物理仿真中运动，并把同一运动发布到 ROS 2。项目 01 给出理想速度积分；本项目加入惯量、关节速度、力矩命令与仿真步进。

[MuJoCo 实测动画](outputs/example/animation.gif)与[最后一帧](outputs/example/final_frame.png)展示当前模型的到达过程。

```text
model/arm2d.urdf（几何、关节、惯量、坐标树）
     ├── MuJoCo 导入 → 加力矩电机 → 物理步进 → qpos/qvel/ctrl → 末端轨迹
     └── robot_state_publisher ← /joint_states ← MuJoCo bridge
                                      ↓
                             /tf、/tf_static、rosbag、RViz
```

## URDF 读法

`link` 是刚体，`joint` 连接父子 link。`shoulder` 的轴为 `+z`，`elbow` 的父 link 是 `link1`，其原点相对 `link1` 位于 `(0.6,0,0) m`。`tool_fixed` 把末端坐标系固定在 `link2` 的 `(0.4,0,0) m`。当 `q1=90°、q2=90°`，末端坐标应为 `(-0.4,0.6,0) m`。

- `visual` 决定外观，`collision` 决定碰撞几何，`inertial` 提供质量与转动惯量。
- `limit` 给出关节范围、额定力矩和速度；执行器命令在生成的 MJCF 中显式添加。
- `origin` 表达子关节或几何体相对于父 link 的位置和姿态；`axis` 表达关节转轴。
- MuJoCo 导入后会合并部分固定 link；脚本依据 URDF 的 `tool_fixed` 原点在 MJCF 中增加 `tool_tip` site，确保日志中的末端与 ROS 2 的 `tool_link` 对齐。

## A. Windows 本机：先完成真实物理仿真

在 PowerShell 中执行：

```powershell
cd 'D:\RCL\深圳理工大学-具身智能\project-02-urdf-mujoco-ros2'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import mujoco; print(mujoco.__version__)"
.\.venv\Scripts\python.exe scripts\run_mujoco.py --output outputs\my-first-run --render-gif
Invoke-Item .\outputs\my-first-run\animation.gif
Get-Content .\outputs\my-first-run\summary.json
```

已有 `.venv` 时，直接从安装依赖一行开始。`--viewer` 会打开交互窗口，运行结束后关闭：

```powershell
.\.venv\Scripts\python.exe scripts\run_mujoco.py --output outputs\live-view --viewer
```

固定 `seed=7`、目标 `(0.55,0.55) m` 的本机实测：100 个控制步成功，最终误差 `0.000927 m`；物理步长 `0.002 s`，控制周期 `0.02 s`，每次控制决策包含 10 个物理步；解析运动学与 MuJoCo 末端位置最大差 `2.5e-16 m`。`animation.gif` 是 MuJoCo 渲染的实物理仿真画面，红点是目标。

输出文件：`derived_scene.xml` 是由 URDF 生成、再加力矩电机和目标/末端 site 的 MJCF；`trajectory.csv` 逐步记录关节位置、速度、力矩命令、期望速度、末端位置和误差；`summary.json` 给出结果；`run_config.json` 给出参数与源码哈希。打开 `trajectory.csv` 第一行和最后一行，观察 `qpos`、`qvel`、`ctrl` 如何变化。

本机使用 MuJoCo `3.2.5`。较新版本在这台 Windows 11 上触发 DLL 初始化错误；独立虚拟环境复现了同一问题。这个版本选择以本机验证为依据，后续可在更新 VC++ 运行库后重新核查。MuJoCo 的 Python 安装与模型导入见[官方文档](https://mujoco.readthedocs.io/en/stable/python.html)；[上游 Windows DLL 问题记录](https://github.com/google-deepmind/mujoco/issues/2275)提供排查线索。

URDF 的质量、惯量、阻尼和电机力矩上限采用教学取值；本实验的比较限定于这份模型与当前控制器。

### 亲手做的两个对照

```powershell
.\.venv\Scripts\python.exe scripts\run_mujoco.py --goal-x 1.05 --goal-y 0 --output outputs\my-unreachable
.\.venv\Scripts\python.exe scripts\run_mujoco.py --seed 11 --output outputs\my-seed11
```

先写预测：不可达目标距离原点 `1.05 m`，机械臂最大伸展为 `1.0 m`，理论末端误差下界为 `0.05 m`；更换种子会改变初始关节角。运行后记录实际 `termination`、最终误差与初始关节角。若结果与预测不同，打开 CSV 寻找最早出现偏差的步骤。

本机不可达目标实测运行到 600 个控制步上限，最终误差为 `0.050000 m`；这个值与手算的几何下界一致。

## B. Ubuntu + ROS 2 Jazzy：接通消息与 TF

当前电脑已经完成 Windows MuJoCo 阶段。ROS 2 阶段需要 Ubuntu 环境；Windows 11 可通过 WSL 2 安装 Ubuntu 24.04。管理员 PowerShell 使用 `wsl --list --online` 查询发行版名称，再执行 `wsl --install -d Ubuntu-24.04`，按照提示完成发行版初始化与可能的重启。安装方式见[微软 WSL 官方说明](https://learn.microsoft.com/windows/wsl/install)。

在 Ubuntu 24.04 中，按 [ROS 2 Jazzy 官方安装说明](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)配置仓库并安装 `ros-jazzy-desktop`。之后将本仓库克隆到 Ubuntu 用户目录，在 Ubuntu 终端执行：

```bash
source /opt/ros/jazzy/setup.bash
git clone https://github.com/ChenglangRong/embodied-robotics-learning-lab.git ~/embodied-robotics-learning-lab
cd ~/embodied-robotics-learning-lab/project-02-urdf-mujoco-ros2
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'import mujoco, rclpy; print(mujoco.__version__)'
python scripts/start_ros.py
```

`--system-site-packages` 让虚拟环境读到 Ubuntu 已安装的 `rclpy` 与 `launch_ros`。`start_ros.py` 启动 `robot_state_publisher` 和 MuJoCo bridge。前者读取同一份 URDF，后者按 `0.02 s` 周期发布 `/joint_states`、末端点、目标点、误差、力矩命令、模拟时间与状态。ROS 消息时间戳采用 ROS 节点时钟；`/robot_lab/sim_time_s` 单独记录仿真时间。

新开 Ubuntu 终端，先执行 `source /opt/ros/jazzy/setup.bash`，再观察：

```bash
ros2 topic list
ros2 topic echo /joint_states --once
ros2 topic echo /robot_lab/status --once
ros2 run tf2_ros tf2_echo base_link tool_link
```

`tf2_echo` 的平移应与 `/robot_lab/tool_point` 的坐标一致。RViz 中选择 `Fixed Frame=base_link`，添加 `RobotModel` 和 `TF` 显示，便于核对 link 与关节轴。

记录一次完整运行时，先开始 rosbag，再调用重置服务：

```bash
ros2 bag record -o ~/robot_lab_demo /joint_states /tf /tf_static /robot_lab/tool_point /robot_lab/goal_point /robot_lab/error_m /robot_lab/ctrl_nm /robot_lab/sim_time_s /robot_lab/status
```

另一个终端调用：

```bash
ros2 service call /robot_lab/reset std_srvs/srv/Trigger '{}'
```

观察状态到 `success` 后，用 `Ctrl+C` 结束录制；`ros2 bag info ~/robot_lab_demo` 检查消息数，再用 `ros2 bag play ~/robot_lab_demo` 回放。`robot_state_publisher` 根据 `/joint_states` 发布活动关节的 `/tf`，固定关节发布到 `/tf_static`；接口语义见[上游 README](https://github.com/ros/robot_state_publisher)。

ROS 2 的运行验收等待 Ubuntu 环境就绪；目前已完成代码编译检查与 Windows MuJoCo 实测。验收时应记录 `ros2 topic hz /joint_states`、TF 末端位置与仿真日志误差、bag 消息数和一段 RViz 演示。

## 项目验收

1. 在纸上画出 `base_link → link1 → link2 → tool_link`，标出两个转轴与两段长度。
2. 用 Windows 命令产出一段 MuJoCo 动画和 CSV；能指出第 1、最后 1 个控制步的 `qpos/qvel/ctrl` 单位。
3. 改一个目标坐标，重复运行，写出几何预测与实际误差。
4. 在 ROS 2 环境用同一 URDF 发布 `/joint_states` 和 TF，录制并回放 bag；检验 TF 末端位置和 MuJoCo `tool_point` 对齐。

每个结论都引用对应配置、命令、CSV 行或 ROS bag。仿真结果的适用范围是当前 URDF、执行器参数、控制律和软件版本。
