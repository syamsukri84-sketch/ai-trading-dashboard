@echo off
cd /d "D:\PROYEK_PYTHON\08_AI TRADING"
if not exist logs mkdir logs
echo === PYTEST MULAI === > logs\verifikasi_gerbang.log
.venv\Scripts\python.exe -m pytest -q >> logs\verifikasi_gerbang.log 2>&1
echo === GERBANG UI MULAI === >> logs\verifikasi_gerbang.log
.venv\Scripts\python.exe scripts\verifikasi_gerbang_ui.py >> logs\verifikasi_gerbang.log 2>&1
echo === SELESAI kode=%errorlevel% === >> logs\verifikasi_gerbang.log
