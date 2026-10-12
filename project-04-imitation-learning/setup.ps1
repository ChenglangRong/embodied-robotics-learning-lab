$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Creating the virtual environment failed" }
}
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Installing requirements failed" }
& ".\.venv\Scripts\python.exe" -m unittest test_learning test_panel -v
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
Write-Host "For physical robot tests, first run project-03 setup to prepare the pinned models, then run: .venv\Scripts\python.exe -m unittest test_platform_learning -v"
