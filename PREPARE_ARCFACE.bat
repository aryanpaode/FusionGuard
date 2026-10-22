@echo off
python -m pip install deepface==0.0.101 tf-keras
if errorlevel 1 exit /b %errorlevel%
python prepare_arcface.py
pause
