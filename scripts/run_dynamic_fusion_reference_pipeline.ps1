param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$RunId,
    [string]$ViewsJson = "outputs/dynamic_fusion/reference_views/20260730_visa_s0_k1_v1/reference_views.json",
    [string]$Manifest = "data/splits/visa/manifest.json",
    [string]$DataRoot = "methods/winclip/datasets/VisA_pytorch/1cls",
    [string]$AnomalyClipCheckpoint = "methods/AnomalyCLIP-main/checkpoints/9_12_4_multiscale_visa/epoch_15.pth",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($RunId)) {
    throw "RunId is required."
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$anomalyClipPython = Join-Path $projectRoot ".venv-anomalyclip/Scripts/python.exe"
$patchCorePython = Join-Path $projectRoot ".venv-patchcore/Scripts/python.exe"
$viewsPath = Join-Path $projectRoot $ViewsJson
$manifestPath = Join-Path $projectRoot $Manifest
$dataRootPath = Join-Path $projectRoot $DataRoot
$checkpointPath = Join-Path $projectRoot $AnomalyClipCheckpoint
$outputRoot = Join-Path $projectRoot "outputs/dynamic_fusion/normal_reference_predictions/$RunId"
$experimentRoot = Join-Path $projectRoot "experiments/dynamic_fusion/$RunId"
$visualDir = Join-Path $outputRoot "anomalydino_visual"
$textDir = Join-Path $outputRoot "anomalyclip_text"
$calibrationPath = Join-Path $outputRoot "calibration.json"
$logPath = Join-Path $experimentRoot "stdout.log"

function Assert-InputFile {
    param([string]$Path, [string]$Name)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Name not found: $Path"
    }
}

function Assert-InputDirectory {
    param([string]$Path, [string]$Name)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Name not found: $Path"
    }
}

function Format-Command {
    param([string]$Executable, [string[]]$Arguments)
    $quoted = $Arguments | ForEach-Object {
        if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }
    return ('"{0}" {1}' -f $Executable, ($quoted -join " "))
}

function Invoke-RecordedStep {
    param(
        [string]$Name,
        [string]$Executable,
        [string[]]$Arguments
    )
    $header = "[$([DateTime]::UtcNow.ToString('o'))] START $Name"
    Add-Content -LiteralPath $logPath -Value $header -Encoding utf8
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
    $stdoutPath = Join-Path $experimentRoot "step_${stamp}_stdout.log"
    $stderrPath = Join-Path $experimentRoot "step_${stamp}_stderr.log"
    $child = Start-Process -FilePath $Executable `
        -ArgumentList $Arguments `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -WindowStyle Hidden `
        -Wait `
        -PassThru
    if (Test-Path -LiteralPath $stdoutPath) {
        Get-Content -LiteralPath $stdoutPath | Add-Content -LiteralPath $logPath -Encoding utf8
    }
    if (Test-Path -LiteralPath $stderrPath) {
        Add-Content -LiteralPath $logPath -Value "[$([DateTime]::UtcNow.ToString('o'))] $Name stderr:" -Encoding utf8
        Get-Content -LiteralPath $stderrPath | Add-Content -LiteralPath $logPath -Encoding utf8
    }
    if ($child.ExitCode -ne 0) {
        throw "$Name failed with exit code $($child.ExitCode)"
    }
    Add-Content -LiteralPath $logPath `
        -Value "[$([DateTime]::UtcNow.ToString('o'))] END $Name" -Encoding utf8
}

Assert-InputFile $anomalyClipPython "AnomalyCLIP Python"
Assert-InputFile $patchCorePython "AnomalyDINO Python"
Assert-InputFile $viewsPath "reference view manifest"
Assert-InputFile $manifestPath "few-shot manifest"
Assert-InputDirectory $dataRootPath "AnomalyDINO VisA data root"
Assert-InputFile $checkpointPath "AnomalyCLIP checkpoint"

$views = Get-Content -LiteralPath $viewsPath -Encoding utf8 -Raw | ConvertFrom-Json
if ($views.status -ne "passed") {
    throw "Reference view manifest status is not passed."
}
if ($views.test_images_used -ne $false -or $views.test_labels_used -ne $false) {
    throw "Reference view manifest used forbidden test data."
}
if ($views.dataset -ne "visa" -or $views.seed -ne 0 -or $views.shot -notin @(1, 2, 4)) {
    throw "This pipeline is locked to VisA seed 0 development data with shot in {1,2,4}."
}
$shot = [int]$views.shot

$baselineProcesses = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -match 'run_promptad_visa_matrix|train_cls.py|train_seg.py' -and
            $_.ProcessId -ne $PID
        }
)
if ($baselineProcesses.Count -gt 0 -and -not $ValidateOnly) {
    throw "PromptAD baseline is still running; reference export will not take the GPU."
}
if ((Test-Path -LiteralPath $outputRoot) -or (Test-Path -LiteralPath $experimentRoot)) {
    throw "RunId already exists; choose a new RunId to avoid overwriting evidence: $RunId"
}

$visualExportArgs = @(
    (Join-Path $projectRoot "scripts/export_anomalydino_normal_references.py"),
    "--views-json", $viewsPath,
    "--manifest", $manifestPath,
    "--data-root", $dataRootPath,
    "--output-dir", $visualDir,
    "--dataset", "VisA",
    "--model-name", "dinov2_vits14",
    "--resolution", "448",
    "--map-max-edge", "448",
    "--device", "cuda:0"
)
$visualAuditArgs = @(
    (Join-Path $projectRoot "scripts/audit_normal_reference_cache.py"),
    "--cache-dir", $visualDir,
    "--manifest", $manifestPath,
    "--dataset", "visa",
    "--branch", "anomalydino_visual",
    "--seed", "0",
    "--shot", ([string]$shot),
    "--min-views-per-source", "5",
    "--expected-categories", "12",
    "--report-json", (Join-Path $outputRoot "anomalydino_audit.json"),
    "--report-csv", (Join-Path $outputRoot "anomalydino_audit.csv")
)
$textExportArgs = @(
    (Join-Path $projectRoot "scripts/export_anomalyclip_normal_references.py"),
    "--views-json", $viewsPath,
    "--checkpoint", $checkpointPath,
    "--output-dir", $textDir,
    "--device", "cuda",
    "--image-size", "518",
    "--features-list", "6", "12", "18", "24",
    "--feature-map-layer", "0"
)
$textAuditArgs = @(
    (Join-Path $projectRoot "scripts/audit_normal_reference_cache.py"),
    "--cache-dir", $textDir,
    "--manifest", $manifestPath,
    "--dataset", "visa",
    "--branch", "anomalyclip_text",
    "--seed", "0",
    "--shot", ([string]$shot),
    "--min-views-per-source", "5",
    "--expected-categories", "12",
    "--report-json", (Join-Path $outputRoot "anomalyclip_audit.json"),
    "--report-csv", (Join-Path $outputRoot "anomalyclip_audit.csv")
)
$fitArgs = @(
    (Join-Path $projectRoot "scripts/fit_dynamic_fusion_calibration.py"),
    "--visual-dir", $visualDir,
    "--text-dir", $textDir,
    "--visual-branch", "anomalydino_visual",
    "--text-branch", "anomalyclip_text",
    "--dataset", "visa",
    "--seed", "0",
    "--shot", ([string]$shot),
    "--temperature", "1.0",
    "--output", $calibrationPath
)

$commands = @(
    (Format-Command $patchCorePython $visualExportArgs),
    (Format-Command $anomalyClipPython $visualAuditArgs),
    (Format-Command $anomalyClipPython $textExportArgs),
    (Format-Command $anomalyClipPython $textAuditArgs),
    (Format-Command $anomalyClipPython $fitArgs)
)

if ($ValidateOnly) {
    [pscustomobject]@{
        status = "validation_passed"
        run_id = $RunId
        baseline_processes = $baselineProcesses.Count
        would_execute = $commands
        output_root = $outputRoot
        experiment_root = $experimentRoot
    } | ConvertTo-Json -Depth 4
    exit 0
}

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
New-Item -ItemType Directory -Path $experimentRoot -Force | Out-Null
$commands | Set-Content -LiteralPath (Join-Path $experimentRoot "command.txt") -Encoding utf8
@"
dataset: visa
seed: 0
shot: $shot
visual_branch: anomalydino_visual
text_branch: anomalyclip_text
normal_views_per_source: 5
temperature: 1.0
gpu_used: true
test_predictions_used: false
test_labels_used: false
"@ | Set-Content -LiteralPath (Join-Path $experimentRoot "config.yaml") -Encoding utf8

$start = [DateTime]::UtcNow
$status = "failed"
$exitCode = 1
$failure = $null
try {
    Invoke-RecordedStep "AnomalyDINO normal-reference export" $patchCorePython $visualExportArgs
    Invoke-RecordedStep "AnomalyDINO reference audit" $anomalyClipPython $visualAuditArgs
    Invoke-RecordedStep "AnomalyCLIP normal-reference export" $anomalyClipPython $textExportArgs
    Invoke-RecordedStep "AnomalyCLIP reference audit" $anomalyClipPython $textAuditArgs
    Invoke-RecordedStep "Normal-reference calibration fit" $anomalyClipPython $fitArgs
    $calibration = Get-Content -LiteralPath $calibrationPath -Encoding utf8 -Raw |
        ConvertFrom-Json
    if ($calibration.status -ne "passed" -or
        $calibration.test_predictions_used -ne $false -or
        $calibration.test_labels_used -ne $false) {
        throw "Final calibration report failed the no-test-data contract."
    }
    $status = "passed"
    $exitCode = 0
}
catch {
    $failure = $_.Exception.Message
        "FAILED: $failure" | Tee-Object -FilePath $logPath -Append
}
finally {
    $end = [DateTime]::UtcNow
    $run = [ordered]@{
        schema_version = 1
        run_id = $RunId
        purpose = "Real VisA seed-0 $shot-shot normal-reference export, audit and calibration"
        status = $status
        dataset = "visa"
        seed = 0
        shot = $shot
        gpu_used = $true
        test_predictions_used = $false
        test_labels_used = $false
        started_at_utc = $start.ToString("o")
        ended_at_utc = $end.ToString("o")
        exit_code = $exitCode
        failure = $failure
        output_root = $outputRoot
    }
    $run | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (Join-Path $experimentRoot 'run.json') -Encoding utf8
    $run | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (Join-Path $experimentRoot 'report.json') -Encoding utf8
    $safeFailure = ([string]$failure) -replace '[\r\n,]', ' '
    $csvLines = @(
        'status,exit_code,failure'
        ('{0},{1},{2}' -f $status, $exitCode, $safeFailure)
    )
    $csvLines |
        Set-Content -LiteralPath (Join-Path $experimentRoot 'report.csv') -Encoding utf8
    if ($status -eq "passed") {
        $decision = '# Decision' + [Environment]::NewLine +
            [Environment]::NewLine +
            'Real normal-reference export, two branch audits, and calibration passed.'
    }
    else {
        $decision = '# Decision' + [Environment]::NewLine +
            [Environment]::NewLine +
            'The run failed. Preserve this directory and retry with a new RunId.' +
            [Environment]::NewLine + 'Failure: ' + $failure
    }
    $decision |
        Set-Content -LiteralPath (Join-Path $experimentRoot 'decision.md') -Encoding utf8
}

exit $exitCode
