param(
    [string]$QueueId = "20260731_promptad_s1k1_stage2_s1k2",
    [string]$ReferenceRunId = "20260731_visa_s0_k1_real_reference_v1",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$queueRoot = Join-Path $projectRoot "outputs/logs/orchestration/$QueueId"
$queueLog = Join-Path $queueRoot "queue.log"
$statusJson = Join-Path $queueRoot "status.json"
$currentMarkerDir = Join-Path $projectRoot "outputs/logs/promptad/visa/seed_1_shot_1"
$currentPredictionDir = Join-Path $projectRoot "outputs/promptad/visa/seed_1_shot_1/predictions"
$currentEvaluation = Join-Path $projectRoot "outputs/unified/promptad_visa_seed_1_shot_1/evaluation_report.json"
$referenceScript = Join-Path $PSScriptRoot "run_dynamic_fusion_reference_pipeline.ps1"
$promptadScript = Join-Path $PSScriptRoot "run_promptad_visa_matrix.ps1"

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
        current_configuration = "promptad_visa_seed_1_shot_1"
        reference_run_id = $ReferenceRunId
        next_configuration = "promptad_visa_seed_1_shot_2"
    } | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath $statusJson -Encoding utf8
}

function Get-PromptADProcesses {
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

function Assert-CurrentConfigurationComplete {
    $markers = @(Get-ChildItem -LiteralPath $currentMarkerDir -Filter "*.complete")
    $predictions = @(Get-ChildItem -LiteralPath $currentPredictionDir -Filter "*.npz")
    if ($markers.Count -ne 24) {
        throw "Current configuration has $($markers.Count)/24 completion markers."
    }
    if ($predictions.Count -ne 12) {
        throw "Current configuration has $($predictions.Count)/12 prediction NPZ files."
    }
    if (-not (Test-Path -LiteralPath $currentEvaluation -PathType Leaf)) {
        throw "Unified evaluation report is missing: $currentEvaluation"
    }
    $report = Get-Content -LiteralPath $currentEvaluation -Encoding utf8 -Raw |
        ConvertFrom-Json
    if ($report.category_count -ne 12 -or
        $report.sample_count -ne 2162 -or
        $report.validation_errors -ne 0) {
        throw "Unified evaluation report failed category/sample/schema checks."
    }
}

try {
    Write-QueueStatus "waiting_current_promptad" "Waiting for seed 1, 1-shot to finish."
    Write-QueueLog "Queue started. Waiting for active PromptAD processes."
    while (@(Get-PromptADProcesses).Count -gt 0) {
        $markerCount = @(
            Get-ChildItem -LiteralPath $currentMarkerDir -Filter "*.complete" -ErrorAction SilentlyContinue
        ).Count
        $predictionCount = @(
            Get-ChildItem -LiteralPath $currentPredictionDir -Filter "*.npz" -ErrorAction SilentlyContinue
        ).Count
        Write-QueueStatus "waiting_current_promptad" "markers=$markerCount/24 predictions=$predictionCount/12"
        Start-Sleep -Seconds $PollSeconds
    }

    Write-QueueLog "No PromptAD process remains. Validating seed 1, 1-shot."
    Write-QueueStatus "validating_current_promptad" "Checking markers, NPZ files and unified evaluation."
    Assert-CurrentConfigurationComplete
    Write-QueueLog "Seed 1, 1-shot validation passed."

    Write-QueueStatus "running_reference_pipeline" "Running real normal-reference export and calibration."
    Write-QueueLog "Starting the real normal-reference pipeline: $ReferenceRunId"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $referenceScript -RunId $ReferenceRunId 2>&1 |
        Tee-Object -LiteralPath $queueLog -Append
    $referenceExitCode = $LASTEXITCODE
    if ($referenceExitCode -eq 0) {
        Write-QueueLog "Real normal-reference pipeline passed."
    }
    else {
        Write-QueueLog "Real normal-reference pipeline failed with exit code $referenceExitCode. Evidence is preserved; baseline queue will continue."
    }

    if (@(Get-PromptADProcesses).Count -gt 0) {
        throw "Another PromptAD job started before seed 1, 2-shot; refusing GPU overlap."
    }
    Write-QueueStatus "running_next_promptad" "Starting VisA seed 1, 2-shot."
    Write-QueueLog "Starting PromptAD VisA seed 1, 2-shot."
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $promptadScript -Seed 1 -Shot 2 -Epoch 100 2>&1 |
        Tee-Object -LiteralPath $queueLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "PromptAD seed 1, 2-shot failed with exit code $LASTEXITCODE"
    }

    Write-QueueStatus "completed" "Reference pipeline attempted and PromptAD seed 1, 2-shot completed." 0
    Write-QueueLog "Serial queue completed."
    exit 0
}
catch {
    $message = $_.Exception.Message
    Write-QueueStatus "failed" $message 1
    Write-QueueLog "FAILED: $message"
    exit 1
}
