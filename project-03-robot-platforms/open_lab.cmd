@echo off
setlocal
set "PYW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PYW%" set "PYW=%~dp0..\project-02-urdf-mujoco-ros2\.venv\Scripts\pythonw.exe"
if not exist "%PYW%" (
  echo Run setup.ps1 first. Python 3.12 and Git are required.
  pause
  exit /b 1
)
start "" "%PYW%" "%~dp0panel.py"
