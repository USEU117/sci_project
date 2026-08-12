param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$Workspace = (Resolve-Path -LiteralPath $Workspace).Path
Set-Location -LiteralPath $Workspace
$python = Join-Path $Workspace ".venv-anomalyclip\Scripts\python.exe"
$statusPath = Join-Path $Workspace "experiments\dynamic_fusion\v2\data_preparation\automation_status.json"
$logPath = Join-Path $Workspace "outputs\logs\v2_data_preparation.log"
$dataEvidence = Join-Path $Workspace "experiments\dynamic_fusion\v2\data_preparation"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $statusPath), (Split-Path -Parent $logPath) | Out-Null

function Write-Status([string]$Status, [string]$Step, [string]$Message) {
    $payload = [ordered]@{
        schema_version = 1
        updated_at = (Get-Date).ToString("o")
        status = $Status
        step = $Step
        message = $Message
        gpu_used = $false
    }
    $payload | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
    "$(Get-Date -Format o) [$Status] [$Step] $Message" | Add-Content -LiteralPath $logPath -Encoding UTF8
}

function Run-Python([string[]]$Arguments) {
    & $python @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed ($LASTEXITCODE): $($Arguments -join ' ')"
    }
}

try {
    $archives = @(
        @{Name="mpdd"; Path=(Join-Path $Workspace "data\downloads\MPDD.zip"); Size=1825041283},
        @{Name="btad"; Path=(Join-Path $Workspace "data\downloads\btad.zip"); Size=1229193337}
    )
    Write-Status "waiting" "download" "Waiting for both resumable downloads to reach their expected sizes."
    while ($true) {
        $complete = $true
        foreach ($item in $archives) {
            $size = if (Test-Path -LiteralPath $item.Path) { (Get-Item -LiteralPath $item.Path).Length } else { 0 }
            if ($size -ne $item.Size) { $complete = $false }
        }
        if ($complete) { break }
        Start-Sleep -Seconds $PollSeconds
    }

    Write-Status "running" "mpdd_extract" "Verifying and extracting MPDD mirror archive."
    Run-Python @(
        "scripts\prepare_v2_dataset_archive.py", "--dataset", "mpdd",
        "--archive", "data\downloads\MPDD.zip", "--destination", "data\mpdd_raw",
        "--source-url", "https://huggingface.co/datasets/meksamiao/mpdd",
        "--source-kind", "mirror",
        "--expected-sha256", "69f8da73eea4a31451a50251e5c261e83e0c53f2d1a39a7d4dfc78b5c434ddd6",
        "--output", "experiments\dynamic_fusion\v2\data_preparation\mpdd_archive.json"
    )
    Write-Status "running" "btad_extract" "Verifying and extracting official BTAD archive."
    Run-Python @(
        "scripts\prepare_v2_dataset_archive.py", "--dataset", "btad",
        "--archive", "data\downloads\btad.zip", "--destination", "data\btad_raw",
        "--source-url", "https://avires.dimi.uniud.it/papers/btad/btad.zip",
        "--source-kind", "official",
        "--output", "experiments\dynamic_fusion\v2\data_preparation\btad_archive.json"
    )

    foreach ($dataset in @("mpdd", "btad")) {
        $archiveReport = Get-Content -LiteralPath (Join-Path $dataEvidence "${dataset}_archive.json") -Raw | ConvertFrom-Json
        $datasetRoot = $archiveReport.dataset_root
        Write-Status "running" "${dataset}_audit" "Validating dataset and generating nested manifests."
        Run-Python @("scripts\validate_dataset.py", "--dataset", $dataset, "--root", $datasetRoot, "--output", "experiments\dynamic_fusion\v2\data_preparation")
        Run-Python @("scripts\prepare_splits.py", "--dataset", $dataset, "--root", $datasetRoot, "--output", "data\splits")
        Run-Python @("scripts\validate_splits.py", "data\splits\$dataset\manifest.json", "--output", "experiments\dynamic_fusion\v2\data_preparation\${dataset}_split_validation.json")
    }
    Write-Status "running" "freeze" "Freezing verified data protocol evidence."
    Run-Python @("scripts\audit_v2_data_readiness.py", "--output", "experiments\dynamic_fusion\v2\data_preparation\readiness.json", "--strict")
    Run-Python @("scripts\freeze_v2_data_protocol.py", "--output", "experiments\dynamic_fusion\v2\data_protocol_freeze\manifest.json")
    Run-Python @("scripts\freeze_v2_data_protocol.py", "--output", "experiments\dynamic_fusion\v2\data_protocol_freeze\manifest.json", "--verify")
    Write-Status "complete" "complete" "MPDD and BTAD data preparation and data-protocol freeze passed."
} catch {
    Write-Status "failed" "failed" $_.Exception.Message
    throw
}
