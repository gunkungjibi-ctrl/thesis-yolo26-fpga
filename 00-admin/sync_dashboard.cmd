@echo off
REM Double-click to rebuild dashboard.html from timeline.md
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync_dashboard.ps1"
echo.
pause
