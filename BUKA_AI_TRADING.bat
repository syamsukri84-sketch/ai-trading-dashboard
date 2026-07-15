@echo off
setlocal
title Buka AI Trading
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

echo Membuka AI Trading Dashboard tanpa VS Code...
echo URL: http://localhost:8501
echo.
if exist "data\models\GLOBAL\Global-Direction-LIGHTGBM_H1_NEXT_DAY_DIRECTION.joblib" (
    echo Status model: Global Model tersedia dan siap digunakan.
) else (
    echo Status model: Global Model belum ditemukan.
    echo Jalankan dulu:
    echo .venv\Scripts\python.exe scripts\train_global_models_cli.py --config config\stocks.yaml --run-type FINAL
)
echo.
echo Jendela ini adalah server dashboard. Tutup jendela ini jika ingin menghentikan dashboard.
echo.

".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false

pause
endlocal
