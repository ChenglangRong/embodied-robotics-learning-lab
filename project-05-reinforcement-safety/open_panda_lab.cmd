@echo off
setlocal
cd /d "%~dp0"
set "TASK_PY=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%TASK_PY%" set "TASK_PY=%~dp0..\.venv\Scripts\pythonw.exe"
if not exist "%TASK_PY%" set "TASK_PY=%~dp0..\project-02-urdf-mujoco-ros2\.venv\Scripts\pythonw.exe"
if not exist "%TASK_PY%" (
  echo Run setup.ps1 and project-03-robot-platforms/setup.ps1 first.
  pause
  exit /b 1
)
start "" "%TASK_PY%" "%~dp0panda_panel.py"
