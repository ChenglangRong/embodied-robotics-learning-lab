param(
    [string]$Task = "",
    [string]$TaskFile = "",
    [ValidateSet("audit", "implement")][string]$Mode = "audit",
    [ValidateRange(1, 5)][int]$MaxRounds = 2,
    [int]$TimeoutSeconds = 900,
    [string]$CodexCli = "",
    [string]$ClaudeCli = "",
    [string]$CodexModel = "",
    [string]$ClaudeModel = "deepseek-flash",
    [string]$CheckCommand = "",
    [switch]$Doctor,
    [switch]$SmokeTest
)
$ErrorActionPreference = "Stop"
$PythonCommand = Get-Command python -ErrorAction Stop
$RelayArgs = @((Join-Path $PSScriptRoot "tools\ai-collab\cooperate.py"), "--workspace", $PSScriptRoot, "--mode", $Mode, "--max-rounds", "$MaxRounds", "--timeout", "$TimeoutSeconds")
if ($Task) { $RelayArgs += @("--task", $Task) }
if ($TaskFile) { $RelayArgs += @("--task-file", $TaskFile) }
if ($CodexCli) { $RelayArgs += @("--codex-cli", $CodexCli) }
if ($ClaudeCli) { $RelayArgs += @("--claude-cli", $ClaudeCli) }
if ($CodexModel) { $RelayArgs += @("--codex-model", $CodexModel) }
if ($ClaudeModel) { $RelayArgs += @("--claude-model", $ClaudeModel) }
if ($CheckCommand) { $RelayArgs += @("--check-command", $CheckCommand) }
if ($Doctor) { $RelayArgs += "--doctor" }
if ($SmokeTest) { $RelayArgs += "--smoke-test" }
& $PythonCommand.Source @RelayArgs
exit $LASTEXITCODE
