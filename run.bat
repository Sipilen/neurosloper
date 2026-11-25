@echo off
cd /d "%~dp0"
echo Starting Neurosloper...
echo Open in browser: http://127.0.0.1:8000
start "" http://127.0.0.1:8000
".venv\Scripts\python.exe" "app.py"
pause
