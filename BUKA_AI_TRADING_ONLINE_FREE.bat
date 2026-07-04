@echo off
setlocal
title AI Trading Online Gratis
cd /d "%~dp0"

if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

if not exist ".venv\Scripts\streamlit.exe" (
    echo Streamlit tidak ditemukan di virtual environment project.
    echo Jalankan dulu:
    echo .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

set "CLOUDFLARED_CMD="
if exist "tools\cloudflared-386.exe" set "CLOUDFLARED_CMD=tools\cloudflared-386.exe"
if "%CLOUDFLARED_CMD%"=="" if exist "tools\cloudflared.exe" set "CLOUDFLARED_CMD=tools\cloudflared.exe"
if "%CLOUDFLARED_CMD%"=="" (
    where cloudflared >nul 2>nul
    if not errorlevel 1 set "CLOUDFLARED_CMD=cloudflared"
)

if "%CLOUDFLARED_CMD%"=="" (
    echo cloudflared belum ditemukan di PATH.
    echo.
    echo Install Cloudflare Tunnel terlebih dahulu, lalu jalankan file ini lagi.
    echo Alternatif sementara:
    echo   ngrok http 8502
    echo.
    echo Dashboard lokal tetap akan dibuka di http://localhost:8502
    echo.
) else (
    echo Cloudflare Tunnel ditemukan.
)

echo Membuka AI Trading Dashboard lokal...
start "AI Trading Dashboard" cmd /k ".venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8502 --server.address localhost --browser.gatherUsageStats false"

timeout /t 8 /nobreak >nul

if not "%CLOUDFLARED_CMD%"=="" (
    echo Membuka Cloudflare Tunnel untuk http://localhost:8502 ...
    start "AI Trading Cloudflare Tunnel" cmd /k "%CLOUDFLARED_CMD% tunnel --url http://localhost:8502"
)

echo.
echo Dashboard lokal : http://localhost:8502
echo URL online akan muncul di jendela Cloudflare Tunnel.
echo.
pause
endlocal
