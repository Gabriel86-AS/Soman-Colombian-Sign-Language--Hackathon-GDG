@echo off
rem Live webcam demo of the letter model. Optional arg: camera index (default 0).
cd /d "%~dp0"
if not exist ".venv-demo\Scripts\python.exe" (
  echo Missing ml\.venv-demo. Create it with Python 3.9-3.12 and: pip install mediapipe opencv-python numpy
  pause
  exit /b 1
)
.venv-demo\Scripts\python demo_letters.py %*
pause
