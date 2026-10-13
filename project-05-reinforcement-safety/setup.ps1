$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
python -m venv "$ProjectRoot\.venv"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& "$ProjectRoot\.venv\Scripts\python.exe" -m pip install -r "$ProjectRoot\requirements.txt"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
Write-Host 'Setup complete. Double-click open_lab.cmd.'
