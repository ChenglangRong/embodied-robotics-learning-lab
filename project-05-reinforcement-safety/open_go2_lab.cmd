@echo off
setlocal
cd /d "%~dp0"
set "TASK_PY=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%TASK_PY%" set "TASK_PY=%~dp0..\.venv\Scripts\pythonw.exe"
if not exist "%TASK_PY%" set "TASK_PY=%~dp0..\project-02-urdf-mujoco-ros2\.venv\Scripts\pythonw.exe"
if not exist "%TASK_PY%" (
  echo Run project03 model setup and project05 environment setup first.
  pause
  exit /b 1
)
start "" "%TASK_PY%" "%~dp0go2_panel.py"
