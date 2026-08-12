param(
    [string]$Config = "configs/gpu_job_queue.json",
    [switch]$ValidateOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$configPath = Join-Path $projectRoot $Config
$promptadScript = Join-Path $PSScriptRoot "run_promptad_visa_matrix.ps1"
$referenceScript = Join-Path $PSScriptRoot "run_dynamic_fusion_reference_pipeline.ps1"

function Assert-File {
    param([string]$Path, [string]$Name)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Name is missing: $Path"
    }
}

Assert-File $configPath "queue config"
Assert-File $promptadScript "PromptAD matrix runner"
Assert-File $referenceScript "reference pipeline"
$queue = Get-Content -LiteralPath $configPath -Encoding utf8 -Raw | ConvertFrom-Json
if ($queue.schema_version -ne 1) {
    throw "Unsupported queue schema version."
}
if ([string]::IsNullOrWhiteSpace($queue.queue_id)) {
    throw "queue_id is required."
}
$jobs = @($queue.jobs)
if ($jobs.Count -eq 0) {
    throw "At least one job is required."
}
$pollSeconds = [int]$queue.poll_seconds
$maxAttempts = [int]$queue.max_attempts_per_job
if ($pollSeconds -lt 10) {
    throw "poll_seconds must be at least 10."
}
if ($maxAttempts -lt 1 -or $maxAttempts -gt 3) {
    throw "max_attempts_per_job must be in [1, 3]."
}
$allowedTypes = @("promptad_visa", "dynamic_reference")
$seen = @{}
foreach ($job in $jobs) {
    if ([string]::IsNullOrWhiteSpace($job.id) -or $seen.ContainsKey($job.id)) {
        throw "Job IDs must be present and unique."
    }
    $seen[$job.id] = $true
    if ($allowedTypes -notcontains $job.type) {
        throw "Unsupported job type: $($job.type)"
    }
    if ($job.type -eq "promptad_visa") {
        if (@(0, 1, 2) -notcontains [int]$job.seed) {
            throw "Invalid PromptAD seed for $($job.id)"
        }
        if (@(1, 2, 4) -notcontains [int]$job.shot) {
            throw "Invalid PromptAD shot for $($job.id)"
        }
    }
}

$stateRoot = Join-Path $projectRoot "outputs/logs/orchestration/$($queue.queue_id)"
$statusPath = Join-Path $stateRoot "status.json"
$logPath = Join-Path $stateRoot "scheduler.log"
$historyPath = Join-Path $stateRoot "history.jsonl"

function Get-GpuJobProcesses {
    return @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.ProcessId -ne $PID -and (
                    $_.CommandLine -match 'run_promptad_visa_matrix.ps1' -or
                    $_.CommandLine -match 'train_cls.py' -or
                    $_.CommandLine -match 'train_seg.py' -or
                    $_.CommandLine -match 'export_anomalyclip_normal_references.py' -or
                    $_.CommandLine -match 'export_anomalydino_normal_references.py' -or
                    $_.CommandLine -match 'run_dynamic_fusion_reference_pipeline.ps1'
                )
            }
    )
}

function Write-Log {
    param([string]$Message)
    $line = "[$([DateTime]::UtcNow.ToString('o'))] $Message"
    Add-Content -LiteralPath $logPath -Value $line -Encoding utf8
    Write-Output $line
}

function Write-Status {
    param(
        [string]$State,
        [int]$JobIndex,
        [string]$Detail,
        [int]$ExitCode = 0
    )
    $currentJob = if ($JobIndex -lt $jobs.Count) { $jobs[$JobIndex].id } else { $null }
    $payload = [ordered]@{
        schema_version = 1
        queue_id = $queue.queue_id
        scheduler_pid = $PID
        state = $State
        next_job_index = $JobIndex
        current_job = $currentJob
        detail = $Detail
        completed_jobs = @($jobs | Select-Object -First $JobIndex | ForEach-Object { $_.id })
        remaining_jobs = @($jobs | Select-Object -Skip $JobIndex | ForEach-Object { $_.id })
        updated_at_utc = [DateTime]::UtcNow.ToString("o")
        exit_code = $ExitCode
    }
    $payload | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Add-History {
    param(
        [string]$Event,
        [string]$JobId,
        [string]$Detail
    )
    [ordered]@{
        timestamp_utc = [DateTime]::UtcNow.ToString("o")
        event = $Event
        job_id = $JobId
        detail = $Detail
    } | ConvertTo-Json -Compress |
        Add-Content -LiteralPath $historyPath -Encoding utf8
}

function Invoke-ChildPowerShell {
    param(
        [string]$ScriptPath,
        [string[]]$Arguments,
        [string]$Label
    )
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $stdoutPath = Join-Path $stateRoot "child_${stamp}_stdout.log"
    $stderrPath = Join-Path $stateRoot "child_${stamp}_stderr.log"
    $child = Start-Process -FilePath "powershell.exe" `
        -ArgumentList (@("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $ScriptPath) + $Arguments) `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    Add-Content -LiteralPath $logPath -Value "[$([DateTime]::UtcNow.ToString('o'))] $Label stdout:" -Encoding utf8
    if (Test-Path -LiteralPath $stdoutPath) {
        Get-Content -LiteralPath $stdoutPath | Add-Content -LiteralPath $logPath -Encoding utf8
    }
    if (Test-Path -LiteralPath $stderrPath) {
        Add-Content -LiteralPath $logPath -Value "[$([DateTime]::UtcNow.ToString('o'))] $Label stderr:" -Encoding utf8
        Get-Content -LiteralPath $stderrPath | Add-Content -LiteralPath $logPath -Encoding utf8
    }
    return [int]$child.ExitCode
}

function Assert-PromptADComplete {
    param([int]$Seed, [int]$Shot)
    $markerDir = Join-Path $projectRoot "outputs/logs/promptad/visa/seed_${Seed}_shot_${Shot}"
    $predictionDir = Join-Path $projectRoot "outputs/promptad/visa/seed_${Seed}_shot_${Shot}/predictions"
    $evaluation = Join-Path $projectRoot "outputs/unified/promptad_visa_seed_${Seed}_shot_${Shot}/evaluation_report.json"
    $markers = @(Get-ChildItem -LiteralPath $markerDir -Filter "*.complete" -ErrorAction SilentlyContinue)
    $predictions = @(Get-ChildItem -LiteralPath $predictionDir -Filter "*.npz" -ErrorAction SilentlyContinue)
    if ($markers.Count -ne 24) {
        throw "markers=$($markers.Count)/24"
    }
    if ($predictions.Count -ne 12) {
        throw "predictions=$($predictions.Count)/12"
    }
    Assert-File $evaluation "unified evaluation"
    $report = Get-Content -LiteralPath $evaluation -Encoding utf8 -Raw | ConvertFrom-Json
    if ($report.category_count -ne 12 -or
        $report.sample_count -ne 2162 -or
        $report.validation_errors -ne 0) {
        throw "unified evaluation failed category/sample/schema checks"
    }
}

function Test-PromptADComplete {
    param([int]$Seed, [int]$Shot)
    try {
        Assert-PromptADComplete -Seed $Seed -Shot $Shot
        return $true
    }
    catch {
        return $false
    }
}

function Find-PassedReferenceRun {
    param([string]$Prefix)
    $candidates = @(
        Get-ChildItem -LiteralPath (Join-Path $projectRoot "outputs/dynamic_fusion/normal_reference_predictions") `
            -Directory -Filter "${Prefix}*" -ErrorAction SilentlyContinue
    )
    foreach ($candidate in $candidates) {
        $calibration = Join-Path $candidate.FullName "calibration.json"
        if (Test-Path -LiteralPath $calibration -PathType Leaf) {
            try {
                $report = Get-Content -LiteralPath $calibration -Encoding utf8 -Raw |
                    ConvertFrom-Json
                if ($report.status -eq "passed" -and
                    $report.test_predictions_used -eq $false -and
                    $report.test_labels_used -eq $false -and
                    @($report.categories.PSObject.Properties).Count -eq 12) {
                    return $candidate.Name
                }
            }
            catch {
            }
        }
    }
    return $null
}

if ($ValidateOnly) {
    [ordered]@{
        status = "validation_passed"
        queue_id = $queue.queue_id
        jobs = @($jobs | ForEach-Object { $_.id })
        job_count = $jobs.Count
        poll_seconds = $pollSeconds
        max_attempts_per_job = $maxAttempts
        active_gpu_job_processes = @(Get-GpuJobProcesses).Count
        state_root_exists = Test-Path -LiteralPath $stateRoot
    } | ConvertTo-Json -Depth 5
    exit 0
}

$startIndex = 0
if ($Resume) {
    Assert-File $statusPath "scheduler status for resume"
    $previous = Get-Content -LiteralPath $statusPath -Encoding utf8 -Raw |
        ConvertFrom-Json
    $oldPid = [int]$previous.scheduler_pid
    # A deliberately paused queue has no scheduler PID.  Do not treat its
    # deserialized null value (0 when cast to int) as a live scheduler.
    if ($oldPid -gt 0 -and $oldPid -ne $PID -and (Get-Process -Id $oldPid -ErrorAction SilentlyContinue)) {
        throw "Another scheduler instance is still running with PID $oldPid."
    }
    $startIndex = [int]$previous.next_job_index
}
else {
    if (Test-Path -LiteralPath $stateRoot) {
        throw "State directory exists. Use -Resume or a new queue_id."
    }
    New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null
}

try {
    Write-Status "starting" $startIndex "Scheduler initialized."
    Write-Log "Scheduler started at job index $startIndex."
    for ($index = $startIndex; $index -lt $jobs.Count; $index++) {
        $job = $jobs[$index]
        $jobId = [string]$job.id
        Write-Status "checking" $index "Checking $jobId."
        Add-History "checking" $jobId "Checking existing evidence."

        if ($job.type -eq "promptad_visa") {
            $seed = [int]$job.seed
            $shot = [int]$job.shot
            if (Test-PromptADComplete -Seed $seed -Shot $shot) {
                Write-Log "$jobId already passed validation; skipping."
                Add-History "skipped_complete" $jobId "Existing result passed validation."
                Write-Status "advanced" ($index + 1) "$jobId already complete."
                continue
            }

            $active = @(Get-GpuJobProcesses)
            if ($active.Count -gt 0) {
                Write-Log "$jobId has an active GPU process; waiting for it to exit."
                while (@(Get-GpuJobProcesses).Count -gt 0) {
                    Write-Status "waiting_active_job" $index "$jobId is already running outside the scheduler."
                    Start-Sleep -Seconds $pollSeconds
                }
                if (Test-PromptADComplete -Seed $seed -Shot $shot) {
                    Write-Log "$jobId external run passed validation."
                    Add-History "external_complete" $jobId "External run passed validation."
                    Write-Status "advanced" ($index + 1) "$jobId complete."
                    continue
                }
                Write-Log "$jobId external run ended incomplete; scheduler will resume it."
            }

            $passed = $false
            for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
                if (@(Get-GpuJobProcesses).Count -gt 0) {
                    throw "GPU job appeared before starting $jobId."
                }
                Write-Status "running" $index "$jobId attempt $attempt/$maxAttempts"
                Write-Log "Starting $jobId attempt $attempt/$maxAttempts."
                Add-History "started" $jobId "Attempt $attempt/$maxAttempts"
                $runExitCode = Invoke-ChildPowerShell `
                    -ScriptPath $promptadScript `
                    -Arguments @("-Seed", "$seed", "-Shot", "$shot", "-Epoch", "$([int]$job.epoch)") `
                    -Label $jobId
                if ($runExitCode -eq 0 -and
                    (Test-PromptADComplete -Seed $seed -Shot $shot)) {
                    $passed = $true
                    Write-Log "$jobId passed full validation."
                    Add-History "passed" $jobId "Attempt $attempt"
                    break
                }
                Write-Log "$jobId attempt $attempt failed or did not validate."
                Add-History "attempt_failed" $jobId "Attempt $attempt exit=$runExitCode"
            }
            if (-not $passed) {
                throw "$jobId failed after $maxAttempts attempts."
            }
        }
        else {
            $prefix = [string]$job.run_id_prefix
            $existing = Find-PassedReferenceRun -Prefix $prefix
            if ($null -ne $existing) {
                Write-Log "$jobId already has passed run $existing; skipping."
                Add-History "skipped_complete" $jobId $existing
                Write-Status "advanced" ($index + 1) "$jobId already complete."
                continue
            }
            while (@(Get-GpuJobProcesses).Count -gt 0) {
                Write-Status "waiting_gpu_idle" $index "Waiting to start $jobId."
                Start-Sleep -Seconds $pollSeconds
            }
            $passed = $false
            for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
                $runId = if ($attempt -eq 1) { $prefix } else { "${prefix}_retry$($attempt - 1)" }
                Write-Status "running" $index "$jobId run_id=$runId"
                Write-Log "Starting $jobId with run ID $runId."
                Add-History "started" $jobId $runId
                $runExitCode = Invoke-ChildPowerShell `
                    -ScriptPath $referenceScript `
                    -Arguments @("-RunId", $runId) `
                    -Label $jobId
                $existing = Find-PassedReferenceRun -Prefix $prefix
                if ($runExitCode -eq 0 -and $null -ne $existing) {
                    $passed = $true
                    Write-Log "$jobId passed as $existing."
                    Add-History "passed" $jobId $existing
                    break
                }
                Write-Log "$jobId run $runId failed."
                Add-History "attempt_failed" $jobId "$runId exit=$runExitCode"
            }
            if (-not $passed) {
                throw "$jobId failed after $maxAttempts attempts."
            }
        }
        Write-Status "advanced" ($index + 1) "$jobId passed."
    }

    Write-Status "completed" $jobs.Count "All configured GPU jobs completed." 0
    Write-Log "All configured GPU jobs completed."
    exit 0
}
catch {
    $message = $_.Exception.Message
    $failedIndex = if (Test-Path -LiteralPath $statusPath) {
        [int](Get-Content -LiteralPath $statusPath -Encoding utf8 -Raw |
            ConvertFrom-Json).next_job_index
    }
    else {
        $startIndex
    }
    Write-Status "failed" $failedIndex $message 1
    Write-Log "FAILED: $message"
    exit 1
}
