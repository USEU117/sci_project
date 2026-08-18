param(
    [int]$PollSeconds = 120,
    [int]$MaxHours = 96,
    [int]$StallMinutes = 45
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$StatusFile = Join-Path $ProjectRoot 'outputs\logs\promptad_mvtec_resumable_queue\status.json'
$LogFile = Join-Path $ProjectRoot 'outputs\logs\baselines_after_promptad.log'
$QueueScript = Join-Path $ProjectRoot 'scripts\run_promptad_mvtec_resumable_queue.py'
$QueuePython = Join-Path $ProjectRoot '.venv-anomalyclip\Scripts\python.exe'
$MarkerRoot = Join-Path $ProjectRoot 'outputs\logs\promptad\mvtec'

function Write-Log($Msg) {
    $Line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    Write-Host $Line
    Add-Content -LiteralPath $LogFile -Value $Line
}

function Get-QueueProcess {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match 'python' -and $_.CommandLine -match 'run_promptad_mvtec_resumable_queue\.py'
    }
}

function Invoke-Baseline($Name, $ScriptPath) {
    Write-Log "=== Starting $Name ==="
    try {
        & $ScriptPath
        Write-Log "=== $Name completed (exit $LASTEXITCODE) ==="
    }
    catch {
        Write-Log "=== $Name FAILED: $_ ==="
    }
}

Write-Log "Watchdog started. poll=${PollSeconds}s max=${MaxHours}h stall=${StallMinutes}min."
$Deadline = (Get-Date).AddHours($MaxHours)

while ((Get-Date) -lt $Deadline) {
    $State = 'unknown'
    $Failures = ''
    if (Test-Path -LiteralPath $StatusFile) {
        try {
            $Status = Get-Content -LiteralPath $StatusFile -Raw | ConvertFrom-Json
            $State = $Status.state
            $Failures = ($Status.failures -join ', ')
        } catch {
            Write-Log "WARN: failed to parse status.json: $_"
        }
    }

    if ($State -eq 'completed') {
        Write-Log "PromptAD queue completed. Starting baselines..."
        Invoke-Baseline 'AdaptCLIP MVTec Gate A' (Join-Path $PSScriptRoot 'start_adaptclip_mvtec_gate_a.ps1')
        Invoke-Baseline 'ReMP-AD MVTec baseline' (Join-Path $PSScriptRoot 'start_remp_ad_mvtec.ps1')
        Write-Log "Baseline relay finished."
        exit 0
    }
    if ($State -eq 'blocked') {
        Write-Log "PromptAD queue blocked (failures: $Failures). Aborting."
        exit 1
    }

    $queueProc = Get-QueueProcess
    if ($null -eq $queueProc) {
        Write-Log "Queue process not running (state=$State). Restarting from checkpoint..."
        Start-Process -FilePath $QueuePython -ArgumentList $QueueScript -WorkingDirectory $ProjectRoot -WindowStyle Hidden
        Write-Log "Queue restart dispatched."
    }

    # Stall warning only (no auto-kill): detect "process alive but no progress" edge case.
    $latest = Get-ChildItem -Path $MarkerRoot -Recurse -Filter '*.complete' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest -and $queueProc) {
        $mins = [int]((Get-Date) - $latest.LastWriteTime).TotalMinutes
        if ($mins -gt $StallMinutes) {
            Write-Log "WARN: no new marker for ${mins} min (last: $($latest.Name) @ $($latest.LastWriteTime)). Queue process alive but possibly stalled."
        }
    }

    Start-Sleep -Seconds $PollSeconds
}

Write-Log "Watchdog timed out (${MaxHours}h)."
exit 2
