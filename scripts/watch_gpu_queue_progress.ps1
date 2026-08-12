param(
    [string]$QueueId = "20260731_full_gpu_queue_v2",
    [int]$Interval = 5,
    [int]$Tail = 6
)

$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$queueDir = Join-Path $root "outputs\logs\orchestration\$QueueId"
$statusPath = Join-Path $queueDir "status.json"
$schedulerLog = Join-Path $queueDir "scheduler.log"

while ($true) {
    Clear-Host
    Write-Host "GPU training queue - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host ""

    $status = $null
    if (Test-Path -LiteralPath $statusPath) {
        try { $status = Get-Content -LiteralPath $statusPath -Raw -Encoding utf8 | ConvertFrom-Json } catch {}
    }
    if ($null -eq $status) {
        Write-Host "Cannot read scheduler status: $statusPath" -ForegroundColor Red
    } else {
        $total = @($status.completed_jobs).Count + @($status.remaining_jobs).Count
        $done = @($status.completed_jobs).Count
        Write-Host "Queue: $done / $total jobs completed  |  State: $($status.state)" -ForegroundColor Yellow
        Write-Host "Current job: $($status.current_job)"
        Write-Host "Detail: $($status.detail)"
        Write-Host "Next: $(@($status.remaining_jobs) -join ' -> ')" -ForegroundColor DarkGray
        Write-Host ""
    }

    Write-Host "GPU" -ForegroundColor Yellow
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
    Write-Host ""

    $training = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -match 'train_cls.py|train_seg.py'
    } | Select-Object ProcessId, CommandLine
    Write-Host "Actual active stage" -ForegroundColor Yellow
    if ($training) { $training | Format-Table -Wrap } else { Write-Host "No active train_cls.py/train_seg.py process." }
    Write-Host ""

    if ($status -and $status.current_job -match '^promptad_visa_seed_(\d+)_shot_(\d+)$') {
        $seed, $shot = $Matches[1], $Matches[2]
        $logDir = Join-Path $root "outputs\logs\promptad\visa\seed_${seed}_shot_${shot}"
        $predictionDir = Join-Path $root "outputs\promptad\visa\seed_${seed}_shot_${shot}\predictions"
        $markers = @(Get-ChildItem -LiteralPath $logDir -Filter '*.complete')
        $predictions = @(Get-ChildItem -LiteralPath $predictionDir -Filter '*.npz')
        Write-Host "Current configuration progress: seed $seed, $shot-shot" -ForegroundColor Yellow
        Write-Host "Training stages: $($markers.Count) / 24  |  Final category predictions: $($predictions.Count) / 12"
        if ($predictions) {
            $latest = $predictions | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            Write-Host "Latest completed prediction: $($latest.Name) at $($latest.LastWriteTime)"
        }
        $latestLog = Get-ChildItem -LiteralPath $logDir -Filter '*.log' |
            Where-Object { $_.Name -notlike '*.error.log' } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latestLog) {
            Write-Host ""
            Write-Host "Latest training log: $($latestLog.Name)" -ForegroundColor Yellow
            Get-Content -LiteralPath $latestLog.FullName -Tail $Tail
        }
    }

    if (Test-Path -LiteralPath $schedulerLog) {
        Write-Host ""
        Write-Host "Latest scheduler event" -ForegroundColor Yellow
        Get-Content -LiteralPath $schedulerLog -Tail 1
    }
    Write-Host ""
    Write-Host "Press Ctrl+C to stop watching. Refreshing in $Interval seconds..." -ForegroundColor DarkGray
    Start-Sleep -Seconds $Interval
}
