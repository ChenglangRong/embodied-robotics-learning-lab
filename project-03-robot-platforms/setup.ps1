param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$LabRoot = $PSScriptRoot
$ModelCache = Join-Path ([Environment]::GetFolderPath("UserProfile")) ".cache\robot-learning\menagerie"
$ModelCommit = "0059d4335f8156206f63a35662313385f7ad6d74"
function CheckExit { if ($LASTEXITCODE -ne 0) { throw "Previous command failed: $LASTEXITCODE" } }
& $Python -c "import sys; assert sys.version_info[:2] == (3,12), 'Use Python 3.12'"
CheckExit
& $Python -m venv (Join-Path $LabRoot ".venv")
CheckExit
$LabPython = Join-Path $LabRoot ".venv\Scripts\python.exe"
& $LabPython -m pip install -r (Join-Path $LabRoot "requirements.txt")
CheckExit
if (!(Test-Path -LiteralPath $ModelCache)) {
    git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git $ModelCache
    CheckExit
}
$CacheRemote = git -C $ModelCache remote get-url origin
CheckExit
if ($CacheRemote -ne "https://github.com/google-deepmind/mujoco_menagerie.git") { throw "Model cache belongs to a different repository" }
$CacheChanges = git -C $ModelCache status --porcelain
if ($CacheChanges) { throw "Preserve model-cache edits before changing its revision" }
git -C $ModelCache fetch origin $ModelCommit
CheckExit
git -C $ModelCache sparse-checkout set franka_emika_panda unitree_go2 unitree_h1 wonik_allegro
CheckExit
git -C $ModelCache checkout --detach $ModelCommit
CheckExit
& $LabPython (Join-Path $LabRoot "verify.py")
CheckExit
Write-Host "Ready. Double-click open_lab.cmd."
