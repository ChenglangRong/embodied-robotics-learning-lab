@echo off
setlocal
cd /d "%~dp0"
set "ALLEGRO_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%ALLEGRO_PY%" set "ALLEGRO_PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%ALLEGRO_PY%" set "ALLEGRO_PY=%~dp0..\project-02-urdf-mujoco-ros2\.venv\Scripts\python.exe"
if not exist "%ALLEGRO_PY%" (
  echo Prepare the project's Python environment and the pinned Menagerie model first.
  pause
  exit /b 1
)
"%ALLEGRO_PY%" "%~dp0run_allegro.py" live %*
if errorlevel 1 pause
