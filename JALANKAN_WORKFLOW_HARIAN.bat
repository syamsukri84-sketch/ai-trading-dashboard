@echo off
setlocal
title Workflow Harian Global Model
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Python virtual environment project tidak ditemukan.
    echo.
    echo Coba jalankan instalasi dependency terlebih dahulu.
    pause
    exit /b 1
)

echo Workflow Harian Global Model
echo ================================================================
echo Urutan:
echo 1. Bangun dataset sentimen lokal
echo 2. Update data harga
echo 3. Evaluasi prediksi pending
echo 4. Prediksi baru dengan Global Model
echo.
echo Prediksi lama tidak ditimpa karena memakai duplicate_policy=skip.
echo ================================================================
echo.

".venv\Scripts\python.exe" scripts\daily_global_workflow_cli.py --config config\stocks.yaml

echo.
echo Selesai. Tekan tombol apa saja untuk menutup jendela ini.
pause
endlocal
