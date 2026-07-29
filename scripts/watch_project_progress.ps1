param(
    [int]$Interval = 5,
    [int]$Tail = 8,
    [ValidateSet(0, 1, 2)]
    [int]$Seed = 0,
    [ValidateSet(1, 2, 4)]
    [int]$Shot = 1
)

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "outputs\logs\promptad\visa\seed_${Seed}_shot_${Shot}"

while ($true) {
    Clear-Host
    Write-Host "Project progress - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "GPU" -ForegroundColor Yellow
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
    Write-Host ""
    Write-Host "Active training processes" -ForegroundColor Yellow
    $processes = Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -like "*train_cls.py*" -or
            $_.CommandLine -like "*train_seg.py*" -or
            $_.CommandLine -like "*run_promptad_visa_matrix.ps1*"
        } |
        Select-Object ProcessId, ParentProcessId, CommandLine
    if ($processes) { $processes | Format-Table -Wrap } else { Write-Host "none" }
    Write-Host ""
    Write-Host "Completed task markers" -ForegroundColor Yellow
    $markers = @(Get-ChildItem $logDir -Filter "*.complete" | Sort-Object Name)
    if ($markers) { $markers | Select-Object -ExpandProperty Name } else { Write-Host "none" }
    Write-Host ""
    Write-Host "Prediction NPZ files" -ForegroundColor Yellow
    $predDir = Join-Path $root "outputs\promptad\visa\seed_${Seed}_shot_${Shot}\predictions"
    @(Get-ChildItem $predDir -Filter "*.npz" | Sort-Object Name) |
        Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
    Write-Host ""
    Write-Host "Latest log lines" -ForegroundColor Yellow
    $latestLog = Get-ChildItem $logDir -Filter "*.log" |
        Where-Object { $_.Name -notlike "*.error.log" } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latestLog) { Get-Content $latestLog.FullName -Tail $Tail }
    Write-Host ""
    Write-Host "Press Ctrl+C to stop watching. Refreshing in $Interval seconds..." -ForegroundColor DarkGray
    Start-Sleep -Seconds $Interval
}
