@echo off
cd /d "%~dp0"
echo Starting Deepfake Detector Web App...
call "venv\Scripts\activate.bat"
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause
