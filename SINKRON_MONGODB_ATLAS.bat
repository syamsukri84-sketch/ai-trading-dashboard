@echo off
setlocal
title Sinkron MongoDB Atlas
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment project tidak ditemukan.
    pause
    exit /b 1
)

if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

echo %MONGODB_URI% | findstr /C:"USER:PASSWORD" /C:"CLUSTER.mongodb.net" /C:"<password>" >nul
if not errorlevel 1 set "MONGODB_URI="

if "%MONGODB_URI%"=="" (
    echo MONGODB_URI belum diisi.
    echo Buat file .env dari .env.example lalu isi koneksi MongoDB Atlas.
    pause
    exit /b 1
)

set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1

".venv\Scripts\python.exe" scripts\sync_mongodb_cli.py
pause
endlocal
