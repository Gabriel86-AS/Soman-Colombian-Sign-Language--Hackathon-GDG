@echo off
cd /d "%~dp0.."
.venv-demo\Scripts\python demo_live\demo.py %*
pause
