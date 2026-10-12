# Codex + Claude Code 自动协作

在工作区 PowerShell 中提交一次任务，协调器负责调用两个 CLI 并传递消息。无需提前手动进入 `codex` 或 `claude`。

当前协作分工：Codex + Claude Code（DeepSeek Flash）。协调器给 Claude Code 显式传入 `--model deepseek-flash`，保留全局 Claude 配置；用量记录包含 CLI 报告的实际模型名。

## 使用

检查路径与 Codex 登录：

```powershell
.\Start-Collaboration.ps1 -Doctor
```

通信测试（三次短模型调用，使用当前登录/接口额度）：

```powershell
.\Start-Collaboration.ps1 -SmokeTest
```

只读协作审查：

```powershell
.\Start-Collaboration.ps1 -Task "审查 project-03-robot-platforms 的成功判据，列出证据与改进建议" -Mode audit
```

实现、审查、修正与复审：

```powershell
.\Start-Collaboration.ps1 -Task "为 project-03-robot-platforms 增加只读日志分析脚本，统计四足滑动并编写离线单元测试" -Mode implement -MaxRounds 2
```

提供你确定的独立测试命令：

```powershell
.\Start-Collaboration.ps1 -Task "审查自动协作脚本" -Mode audit -CheckCommand 'python -m unittest discover -s tools/ai-collab -p test_cooperate.py'
```

较长任务写入 UTF-8 文本，然后使用 `-TaskFile .\你的任务.txt`。PowerShell 策略限制脚本执行时，可以直接用 Python 入口，保持系统策略原状：

```powershell
python .\tools\ai-collab\cooperate.py --doctor
python .\tools\ai-collab\cooperate.py --task "审查项目03" --mode audit
```

## 自动消息流

`Codex 分析/实现 → Claude Code 独立审查 → Codex 核查/修正 → Claude Code 复审`

审查接受后，最终报告还会转交 Codex 做只读交接确认。Claude Code 判定需要修正时，implement 模式继续下一轮；audit 模式保留建议并结束。遇到权限缺失、超时、认证错误或无效回复格式时停止，保留日志。默认最多 2 轮，每次 CLI 调用最多 900 秒；2 轮最多包含 5 次模型调用。

各阶段新建独立 CLI 会话，通过协调器转发记录保持上下文。现有 Claude 交互会话与 Codex 桌面聊天分别保留原来的历史；这套脚本的自动消息流和完整结果保存在工作区。

## 消息和结果在哪里

每次生成新目录 `.ai-collab/runs/时间-随机ID/`：

- `task.txt`：原始任务。
- `messages.md` / `messages.jsonl`：双方全部交换及协调器测试结果。
- `*.prompt.txt`：实际发给 CLI 的提示。
- `*.reply.md`：回复原文。
- `*.stdout.log` / `*.stderr.log`：CLI 原始日志。
- `summary.json`：最终状态、版本、路径、完成轮数和测试状态。
- `initial-files.json` / `initial-git.txt`：开始时的文件哈希和工作区状态。

`.ai-collab/latest.txt` 保存最近一次已取得协调锁的运行目录。运行资料已加入 `.gitignore`，保留在本机。命令结束后看终端的 Status 和 messages.md 路径即可。

## 权限、状态与成本

默认 audit 模式：Codex 使用 read-only 命令沙箱，Claude Code 的内置工具仅开放 Read/Glob/Grep。implement 模式：Codex 使用 workspace-write 命令沙箱，Claude Code 继续只读。最终 Codex 交接阶段始终只读。Claude Code 使用 safe-mode，独立 MCP 配置设为空，并禁用 MCP 工具，降低个人扩展干扰。

协调器使用独占锁，防止另一份协调脚本同时运行。Claude 审查前后核对 Git 可见文件内容；发现变化就停止并报告，保留原文件。其他桌面/CLI 会话需要自行遵守单写者约定，文件哈希检查只能检测冲突。Codex 的命令沙箱和这些工具限制覆盖各自声明的边界；独立操作系统账户或容器可以进一步隔离网络、文件和扩展权限。

提示中要求保留用户改动及 Git 提交/推送状态。脚本直接管理当前任务的子进程，超时或 Ctrl+C 时终止自己启动的进程树。测试命令仅由用户通过 `-CheckCommand` 提供；对方回复作为待核查证据传递。

最终状态明确区分：`completed`（审查接受且用户指定测试通过）、`review_passed_tests_unconfigured`（审查接受，独立测试未指定）、`tests_failed`、`changes_requested`、`round_limit_changes_remaining`、`blocked`、`agent_error`。审查通过与测试通过分别记录。

每次调用消耗相应登录账户或配置接口的额度。CLI 用量估计与实际账单可能存在差异，第三方接口还受路由和价格配置影响。

## 本机已确认的环境与测试

2026-10-12：Claude Code 2.1.296 的实际程序位于 `D:\RCL\claude-cli\node_modules\@anthropic-ai\claude-code\bin\claude.exe`。协调器按本工作区的相邻 claude-cli 目录自动寻找它。

npm 的 Codex CLI 0.125.0 读取当前桌面配置时，对 `service_tier=default` 报兼容错误。协调器优先选择桌面附带的 Codex CLI，当前为 `0.162.0-alpha.17.2`，已确认 ChatGPT 登录有效。脚本保留全局配置和 npm 安装状态；桌面升级后按安装目录重新发现程序。

Claude Code 的本机全局配置包含 `ANTHROPIC_BASE_URL` 的 DeepSeek 主机及 `ANTHROPIC_MODEL=deepseek-v4-pro`。本协调器根据用户选择，在每次调用中显式使用 `deepseek-flash`，保留既有接口和认证。客户端、接口提供方、实际模型分别记录和判断。DeepSeek 当前官方模型标识为 `deepseek-flash`，见 [DeepSeek 官方接口文档](https://api-docs.deepseek.com/)与 [Claude Code 接入说明](https://api-docs.deepseek.com/quick_start/agent_integrations/claude_code/)。

真实双向通信测试已通过，完整证据位于 `.ai-collab/runs/20261012-171044-1eb68950/messages.md`：Codex 唯一标识被 Claude Code 确认，Claude Code 的唯一回复又被 Codex 确认。

第一次通信测试沿用全局 Pro 模型。收到 Flash 选择后，中止自己启动的后续审查进程树，保留运行日志，并开始 Flash 专项验证。其他用户交互会话保留运行状态。

Flash 专项通信测试已通过，证据位于 `.ai-collab/runs/20261012-171832-64eed52f/messages.md` 与 `summary.json`，CLI 用量报告的模型名为 `deepseek-flash`。15 项离线测试全部通过，覆盖消息回传、修正复审流程、轮次限制、只读模式、UTF-8、超时、错误退出、测试失败与锁释放。实际代码修改任务仍需按各任务提供独立测试命令。

单独在你常用的 PowerShell 中进入 Flash 会话，可使用：

```powershell
claude --model deepseek-flash
```

离线测试：

```powershell
python -m unittest discover -s tools/ai-collab -p test_cooperate.py -v
```

## 官方接口依据

Codex 使用 `exec --json --output-last-message` 与显式沙箱配置，见 [OpenAI Docs：非交互模式](https://learn.chatgpt.com/docs/non-interactive-mode)。Claude Code 使用 `-p --output-format json`、工具限制及独立会话，见 [Claude Code：程序化运行](https://code.claude.com/docs/en/headless)与 [CLI 参数](https://code.claude.com/docs/en/cli-reference)。
