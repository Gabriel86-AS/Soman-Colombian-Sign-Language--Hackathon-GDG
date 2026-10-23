@echo off
rem Live webcam demo of the dynamic word model (8 words). Optional args: camera index, --check, --model words_mlp.
cd /d "%~dp0"
if not exist ".venv-demo\Scripts\python.exe" (
  echo Missing ml\.venv-demo. Create it with Python 3.9-3.12 and: pip install mediapipe opencv-python numpy pandas
  pause
  exit /b 1
)
if not exist "out\words_mlp_swap.json" (
  echo Missing ml\out\words_mlp_swap.json. Run git pull.
  pause
  exit /b 1
)
.venv-demo\Scripts\python Demo_dynamic_words\demo.py %*
pause
