# Project 01 — Reproducible 2-Link Simulation Software

一个可复现、可演示的二连杆机器人仿真软件项目。它用于训练机器人软件接口、实验日志、控制器替换、失败注入和结果审计能力。

从第一条命令到实验记录的指导见[从零动手.md](从零动手.md)。[示例动画](outputs/example/reachable/animation.gif)展示可达目标的完整过程。

## 环境

Python 3.10+，NumPy，Matplotlib。MuJoCo 和 ROS 2 进入项目 02；项目 01 保持纯 Python 路径，便于在 Windows 上稳定复现。

## 运行

```powershell
python -m pip install -r requirements.txt
python scripts/run_demo.py --output outputs/demo
python scripts/run_workshop.py --output-root outputs/my-workshop
python -m unittest discover -s tests -v
```

输出包括轨迹 CSV、摘要 JSON、运行配置和轨迹图。固定 `--seed` 后，重复运行的摘要应一致。

## 任务

控制二连杆末端到达目标。实验覆盖可达目标、不可达目标和速度限幅。控制器使用阻尼最小二乘，环境接口固定为 `reset(seed)` 与 `step(action)`。

## 当前验收

- 解释 `qpos`、`qvel`、action、时间步和终止原因。
- 通过运动学、确定性重置、日志字段和任务判据测试。
- 展示一次正常轨迹和一次受控失败。
- 修改一个控制器参数，提交前后结果和解释。

## 结论边界

项目验证软件闭环和低维控制逻辑。它不提供真实机器人动力学、碰撞安全或人机实验结论。

