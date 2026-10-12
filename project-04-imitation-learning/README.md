# 项目04：从专家控制器到学习策略

进入顺序：完成项目03个人修改与解释 → 观测/动作契约 → 专家示范与分组划分 → 二连杆BC/DAgger → Panda → Go2 → H1 → Allegro → 五平台正式评测 → 动作块策略。

当前状态：模仿学习覆盖二连杆、Panda、Go2、H1、Allegro。二连杆BC/DAgger五种子已生成；四平台相位条件参考BC、相位插值BC、固定状态消融与保持初态对照已实现并完成快速实验。四平台正式五种子、自主阶段切换与动作块按后续闸门推进；负结果完整保留。

第1课：[01-专家数据与动作接口.md](01-专家数据与动作接口.md)。

当前实操：[03-五平台模仿学习实操课.md](03-五平台模仿学习实操课.md)。双击 `open_lab.cmd` 打开五平台入口，可训练、回放、启动MuJoCo实时3D。二连杆课程：[02-行为克隆实操课.md](02-行为克隆实操课.md)，直接入口 `open_planar_lab.cmd`，保留单步与个人CSV保存。

四平台样例：[report.html](outputs/platform-baseline-20261012-r3/report.html)。quick为1种子、每域2案例；Panda/Allegro相位BC的ID各2/2，MLP BC各平台ID均0/2。相位BC属于时钟参考模仿，反馈泛化独立验证；PD/位置伺服保留，Allegro腕部夹具外部规定，H1当前双足支撑。

## 二连杆五种子测量

正式样例目录：`outputs/baseline-20261012-r2/`；初次运行保留在 `outputs/baseline-20261012/`。每种子24个ID、24个OOD案例；5种子使用同一数据和测试集，离散度衡量训练随机性。

| 控制器 | ID成功率均值 | OOD成功率均值 | ID平均最终误差 |
| --- | ---: | ---: | ---: |
| DLS专家 | 100% | 100% | 约0.96 mm |
| 零动作 | 0% | 0% | 见原始结果 |
| BC | 9.17% | 0.83% | 19.447 mm |
| BC + 追加更新 | 5.00% | 1.67% | 25.559 mm |
| BC + 新专家轨迹 | 5.00% | 0.83% | 19.619 mm |
| BC + DAgger标签 | 5.00% | 0% | 19.607 mm |

精确值与逐种子SD：[summary.json](outputs/baseline-20261012-r2/summary.json)；图形报告：[report.html](outputs/baseline-20261012-r2/report.html)；ID回放：[demo-test.gif](outputs/baseline-20261012-r2/demo-test.gif)。本次DAgger变体的改进证据为负；模型拟合、数据覆盖与精细到达控制需要进一步分离诊断。

## 安装与复现

本机入口优先使用本项目 `.venv`，同时支持复用项目02已有的Python环境。新机器安装Python3.12，保留项目01/03，先按项目03的setup下载固定提交模型，再在本目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv\Scripts\python.exe -m unittest test_learning test_panel -v
.\.venv\Scripts\python.exe learning.py --mode full
.\.venv\Scripts\python.exe verify_run.py outputs/baseline-20261012-r2
.\open_lab.cmd
```

快速学习用 `--mode quick`（1种子、600+400次更新）；正式协议用 `--mode full`（5种子、1600+1200次更新）。每次运行创建独立目录，完整训练产生 `report.html`、损失CSV、权重、数据、查询账本和评测JSON。壁钟时间随电脑负载变化，模拟时长固定为每步0.02 s，最长600步。

`MLP.save()`保存推理权重与归一化；本次追加分支通过内存克隆保留Adam状态。磁盘权重用于推理，继续训练时需单独保存优化器状态。`selected_stage_update`表示当前阶段的检查点更新次数；`selected_optimizer_update`表示从基础训练累计的Adam步数。`latency`只测已编码观测的归一化与网络前向，观测构造、渲染、通信和执行耗时单独评测。

本机r2完整实验实测47.69 s，包含数据采集、五种子训练、闭环评测和GIF生成；耗时随机器负载变化。最新运行指针、个人记录与控制台日志留在本机；精选实验样例可随代码审查后公开。

代码阅读顺序：`encode/collect` → `MLP.forward/loss_grad/update` → `fit` → `rollout/evaluate` → `run_experiment`。原始教学实现依赖NumPy，绘图与GIF依赖Pillow。`verify_run.py`核对数据/源码哈希、划分、归一化、专家标签、查询预算和存储策略的重新执行结果；20项单元测试包含5项无界面回调测试。

证据与验收范围：[审查与验证.md](审查与验证.md)。精选源码、数据、权重和完整失败样例随仓库公开；阶段同步、个人验收和实际CI状态统一登记在[学习进度与能力证据](../学习进度与能力证据.md)。私人会话与运行临时目录留在本机。

## 预先固定的对照协议

- 复用项目01的 `TwoLinkReachEnv` 与DLS专家；状态观测6维，动作2维(rad/s)。这是运动学到达环境，动力学接触训练在Panda阶段追加。
- 以完整episode划分训练、验证、测试，固定目标与初态；归一化只用训练集拟合。保存所有失败专家episode及其终止原因，训练子集的筛选规则单独记录。
- 学生策略、专家与零动作基线使用相同成功阈值和步数上限；先保持项目01的1 mm到达容差，评估困难时按失败报告处理。
- 记录teacher-forced动作MSE(rad²/s²)、闭环成功率、最终位置误差(m)、动作限幅率、失败原因和实际测得的推理延迟。
- 最终比较至少5个训练种子，使用同一冻结测试集；保留每个种子的结果，报告均值与离散度。OOD测试目标域在训练前另行冻结。
- DAgger在学习策略访问的状态查询DLS动作，保存专家查询数量与数据新增量；测试集保持冻结。加入算法时报告相同标注/训练预算下的对照。

## 迁移到Panda的独立闸门

观测需覆盖任务状态、物体/目标相对位姿、关节状态和夹爪状态；时间阶段或历史观测需解决抓取/释放动作的阶段歧义。当前Panda动作为7关节角参考(rad)+夹爪比例，接入既有位置伺服；关节角增量与末端增量属于后续动作表示对照。

先收集多初态、多目标示范，复用项目03搬运与放置判据并补充过程接触/相对位姿条件。加入相机后评估观测误差、坐标标定、数据规模和计算延迟。更复杂的动作块策略在BC闭环基线稳定后开展。
