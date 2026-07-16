@echo off
setlocal
title Buka AI Trading Cockpit
cd /d "%~dp0"

if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Python virtual environment project tidak ditemukan.
    echo.
    echo Coba jalankan:
    echo .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Membuka AI Trading Cockpit ringan...
echo URL: http://localhost:8500/cockpit
echo.
echo Jendela ini adalah server cockpit. Tutup jendela ini jika ingin menghentikan cockpit.
echo.

start "" "http://localhost:8500/cockpit"
".venv\Scripts\python.exe" -m uvicorn fastapi_app:app --host 127.0.0.1 --port 8500

pause
endlocal
