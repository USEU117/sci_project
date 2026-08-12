param(
    [string]$QueueId = "20260731_promptad_overnight_tail",
    [int]$PredecessorPid = 24720,
    [string]$PredecessorStatus = "outputs/logs/orchestration/20260731_promptad_s1k1_stage2_s1k2/status.json",
    [int]$PollSeconds = 60,
    [int]$MaxAttempts = 2
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$queueRoot = Join-Path $projectRoot "outputs/logs/orchestration/$QueueId"
$queueLog = Join-Path $queueRoot "queue.log"
$statusJson = Join-Path $queueRoot "status.json"
$predecessorStatusPath = Join-Path $projectRoot $PredecessorStatus
$promptadScript = Join-Path $PSScriptRoot "run_promptad_visa_matrix.ps1"
$sequence = @(
    [pscustomobject]@{ Seed = 1; Shot = 4 },
    [pscustomobject]@{ Seed = 2; Shot = 1 },
    [pscustomobject]@{ Seed = 2; Shot = 2 },
    [pscustomobject]@{ Seed = 2; Shot = 4 }
)

if ($PollSeconds -lt 10) {
    throw "PollSeconds must be at least 10."
}
if ($MaxAttempts -lt 1 -or $MaxAttempts -gt 3) {
    throw "MaxAttempts must be in [1, 3]."
}
if (Test-Path -LiteralPath $queueRoot) {
    throw "Queue directory already exists; refusing to overwrite: $queueRoot"
}
New-Item -ItemType Directory -Path $queueRoot -Force | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $line = "[$([DateTime]::UtcNow.ToString('o'))] $Message"
    Add-Content -LiteralPath $queueLog -Value $line -Encoding utf8
    Write-Output $line
}

function Write-QueueStatus {
    param(
        [string]$State,
        [string]$Detail,
        [int]$ExitCode = 0
    )
    [ordered]@{
        schema_version = 1
        queue_id = $QueueId
        state = $State
        detail = $Detail
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
        exit_code = $ExitCode
        predecessor_pid = $PredecessorPid
        predecessor_status = $predecessorStatusPath
        remaining_sequence = @(
            $sequence | ForEach-Object { "promptad_visa_seed_$($_.Seed)_shot_$($_.Shot)" }
        )
        max_attempts_per_configuration = $MaxAttempts
    } | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $statusJson -Encoding utf8
}

function Get-ActivePromptAD {
    return @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.ProcessId -ne $PID -and (
                    $_.CommandLine -match 'run_promptad_visa_matrix.ps1' -or
                    $_.CommandLine -match 'train_cls.py' -or
                    $_.CommandLine -match 'train_seg.py'
                )
            }
    )
}

function Assert-ConfigurationComplete {
    param([int]$Seed, [int]$Shot)
    $markerDir = Join-Path $projectRoot "outputs/logs/promptad/visa/seed_${Seed}_shot_${Shot}"
    $predictionDir = Join-Path $projectRoot "outputs/promptad/visa/seed_${Seed}_shot_${Shot}/predictions"
    $evaluation = Join-Path $projectRoot "outputs/unified/promptad_visa_seed_${Seed}_shot_${Shot}/evaluation_report.json"
    $markers = @(Get-ChildItem -LiteralPath $markerDir -Filter "*.complete" -ErrorAction SilentlyContinue)
    $predictions = @(Get-ChildItem -LiteralPath $predictionDir -Filter "*.npz" -ErrorAction SilentlyContinue)
    if ($markers.Count -ne 24) {
        throw "seed=$Seed shot=$Shot has $($markers.Count)/24 markers."
    }
    if ($predictions.Count -ne 12) {
        throw "seed=$Seed shot=$Shot has $($predictions.Count)/12 predictions."
    }
    if (-not (Test-Path -LiteralPath $evaluation -PathType Leaf)) {
        throw "seed=$Seed shot=$Shot evaluation report is missing."
    }
    $report = Get-Content -LiteralPath $evaluation -Encoding utf8 -Raw |
        ConvertFrom-Json
    if ($report.category_count -ne 12 -or
        $report.sample_count -ne 2162 -or
        $report.validation_errors -ne 0) {
        throw "seed=$Seed shot=$Shot evaluation validation failed."
    }
}

try {
    Write-QueueStatus "waiting_predecessor" "Waiting for PID $PredecessorPid."
    Write-QueueLog "Overnight tail queue started. Waiting for predecessor PID $PredecessorPid."
    while (Get-Process -Id $PredecessorPid -ErrorAction SilentlyContinue) {
        Write-QueueStatus "waiting_predecessor" "The first serial queue is still active."
        Start-Sleep -Seconds $PollSeconds
    }

    if (-not (Test-Path -LiteralPath $predecessorStatusPath -PathType Leaf)) {
        throw "Predecessor status file is missing."
    }
    $predecessor = Get-Content -LiteralPath $predecessorStatusPath -Encoding utf8 -Raw |
        ConvertFrom-Json
    if ($predecessor.state -ne "completed" -or $predecessor.exit_code -ne 0) {
        throw "Predecessor queue did not complete successfully: $($predecessor.state)"
    }
    Assert-ConfigurationComplete -Seed 1 -Shot 2
    Write-QueueLog "Predecessor and PromptAD seed 1, 2-shot validation passed."

    foreach ($configuration in $sequence) {
        $seed = [int]$configuration.Seed
        $shot = [int]$configuration.Shot
        $name = "promptad_visa_seed_${seed}_shot_${shot}"
        $passed = $false
        for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
            if (@(Get-ActivePromptAD).Count -gt 0) {
                throw "Another PromptAD process is active before $name."
            }
            Write-QueueStatus "running" "$name attempt $attempt/$MaxAttempts"
            Write-QueueLog "Starting $name attempt $attempt/$MaxAttempts."
            & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $promptadScript `
                -Seed $seed -Shot $shot -Epoch 100 2>&1 |
                Tee-Object -LiteralPath $queueLog -Append
            $runExitCode = $LASTEXITCODE
            if ($runExitCode -eq 0) {
                try {
                    Assert-ConfigurationComplete -Seed $seed -Shot $shot
                    $passed = $true
                    Write-QueueLog "$name passed full validation."
                    break
                }
                catch {
                    Write-QueueLog "$name validation failed: $($_.Exception.Message)"
                }
            }
            else {
                Write-QueueLog "$name exited with code $runExitCode."
            }
            if ($attempt -lt $MaxAttempts) {
                Write-QueueLog "Retrying $name through its resumable markers and NPZ files."
            }
        }
        if (-not $passed) {
            throw "$name failed after $MaxAttempts attempts."
        }
    }

    Write-QueueStatus "completed" "All overnight tail configurations completed." 0
    Write-QueueLog "Overnight tail queue completed."
    exit 0
}
catch {
    $message = $_.Exception.Message
    Write-QueueStatus "failed" $message 1
    Write-QueueLog "FAILED: $message"
    exit 1
}
