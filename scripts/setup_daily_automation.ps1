<#
.SYNOPSIS
Mendaftarkan tugas terjadwal Windows (Task Scheduler) untuk menjalankan
workflow AI Trading secara otomatis, tanpa perlu membuka dashboard atau
menjalankan skrip manual tiap hari.

.DESCRIPTION
Mendaftarkan DUA task:
1. "AITrading_DailyWorkflow" -- update data harga, evaluasi akurasi prediksi
   lama, dan buat prediksi Global Model baru. Jalan tiap Senin-Jumat jam
   -DailyTime (default 16:30, setelah bursa IDX tutup).
2. "AITrading_WeeklyEdgeScreening" -- screening walk-forward genuine edge
   untuk SEMUA ticker (proses berat, ~1 jam). Jalan tiap Sabtu jam
   -WeeklyTime (default 06:00) supaya tidak mengganggu jam kerja.

Skrip ini TIDAK mengubah apa pun secara otomatis saat file-nya dibuat --
baru berjalan setelah Anda menjalankannya sendiri lewat PowerShell.

.PARAMETER DailyTime
Jam eksekusi workflow harian, format HH:mm (default 16:30).

.PARAMETER WeeklyTime
Jam eksekusi screening mingguan, format HH:mm (default 06:00).

.EXAMPLE
.\scripts\setup_daily_automation.ps1
Mendaftarkan kedua task dengan jadwal default.

.EXAMPLE
.\scripts\setup_daily_automation.ps1 -DailyTime "17:00"
Workflow harian jalan jam 17:00 alih-alih 16:30.

.NOTES
Untuk membatalkan otomatisasi:
  Unregister-ScheduledTask -TaskName "AITrading_DailyWorkflow" -Confirm:$false
  Unregister-ScheduledTask -TaskName "AITrading_WeeklyEdgeScreening" -Confirm:$false

Untuk cek status/riwayat run:
  Get-ScheduledTaskInfo -TaskName "AITrading_DailyWorkflow"
#>
param(
    [string]$DailyTime = "16:30",
    [string]$WeeklyTime = "06:00"
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python venv tidak ditemukan di $PythonExe. Jalankan skrip ini dari folder project AI Trading yang sudah punya .venv."
    exit 1
}

# Task Scheduler menjalankan proses tanpa konsol -- tanpa redirect eksplisit,
# print()/traceback dari Python lenyap total kalau terjadi crash tak terduga.
# Semua output (stdout+stderr) di-append ke file log ini lewat wrapper cmd.exe,
# supaya kegagalan tengah malam/pagi tetap punya jejak yang bisa dibaca besoknya.
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# --- Task 1: Workflow harian (update data + evaluasi + prediksi) ---
$DailyScript = Join-Path $ProjectRoot "scripts\daily_global_workflow_cli.py"
$DailyLog = Join-Path $LogDir "daily_workflow.log"
$DailyArgument = "/c `"`"$PythonExe`" `"$DailyScript`" >> `"$DailyLog`" 2>&1`""
$DailyAction = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $DailyArgument -WorkingDirectory $ProjectRoot
$DailyTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $DailyTime
$DailySettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Hours 2)

Register-ScheduledTask -TaskName "AITrading_DailyWorkflow" `
    -Action $DailyAction -Trigger $DailyTrigger -Settings $DailySettings `
    -Description "AI Trading: update data harga, evaluasi akurasi, prediksi Global Model. Senin-Jumat setelah bursa tutup." `
    -Force | Out-Null
Write-Host "Terdaftar: AITrading_DailyWorkflow (Senin-Jumat jam $DailyTime) -- log: $DailyLog"

# --- Task 2: Screening genuine edge mingguan (proses berat, ~1 jam) ---
$ScreeningScript = Join-Path $ProjectRoot "scripts\screen_genuine_edge.py"
$ScreeningLog = Join-Path $LogDir "weekly_edge_screening.log"
$ScreeningArgument = "/c `"`"$PythonExe`" `"$ScreeningScript`" >> `"$ScreeningLog`" 2>&1`""
$ScreeningAction = New-ScheduledTaskAction -Execute "cmd.exe" -Argument $ScreeningArgument -WorkingDirectory $ProjectRoot
$ScreeningTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At $WeeklyTime
$ScreeningSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Hours 3)

Register-ScheduledTask -TaskName "AITrading_WeeklyEdgeScreening" `
    -Action $ScreeningAction -Trigger $ScreeningTrigger -Settings $ScreeningSettings `
    -Description "AI Trading: screening walk-forward genuine edge untuk semua ticker di config/stocks.yaml. Sabtu pagi." `
    -Force | Out-Null
Write-Host "Terdaftar: AITrading_WeeklyEdgeScreening (Sabtu jam $WeeklyTime) -- log: $ScreeningLog"

Write-Host ""
Write-Host "Selesai. Dashboard akan otomatis punya data terbaru tanpa perlu dibuka manual."
Write-Host "Cek status kapan saja: Get-ScheduledTaskInfo -TaskName 'AITrading_DailyWorkflow'"
Write-Host "Cek log kalau ada kecurigaan gagal: Get-Content '$DailyLog' -Tail 100"
Write-Host "Batalkan: Unregister-ScheduledTask -TaskName 'AITrading_DailyWorkflow' -Confirm:`$false"
