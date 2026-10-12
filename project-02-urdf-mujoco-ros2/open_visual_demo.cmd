@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. Follow README.md to install it.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" "scripts\run_mujoco.py" --seed 7 --goal-x 0.55 --goal-y 0.55 --viewer --viewer-speed 0.15 --hold-final 15 --render-gif --output "outputs\guided-demo"
if errorlevel 1 (
  pause
  exit /b 1
)
start "" "outputs\guided-demo\animation.gif"
echo Replay and CSV are saved in outputs\guided-demo.
pause
