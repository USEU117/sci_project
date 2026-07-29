param(
    [int]$ImageSize = 518,
    [string]$Python = "",
    [switch]$Background
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$MethodRoot = Join-Path $ProjectRoot "methods\anomalyclip-main"
if (-not $Python) {
    $Python = Join-Path $ProjectRoot ".venv-anomalyclip\Scripts\python.exe"
}

$dataRoot = Join-Path $ProjectRoot "data\mvtec"
$saveRoot = Join-Path $ProjectRoot "outputs\anomalyclip\mvtec_visa_checkpoint"
$dumpRoot = Join-Path $ProjectRoot "outputs\anomalyclip\mvtec_npz"
$logRoot = Join-Path $ProjectRoot "outputs\logs"
$logPath = Join-Path $logRoot "anomalyclip_mvtec.log"
$errorLogPath = Join-Path $logRoot "anomalyclip_mvtec.error.log"

foreach ($required in @($Python, $MethodRoot, $dataRoot)) {
    if (-not (Test-Path $required)) {
        throw "Required path not found: $required"
    }
}
New-Item -ItemType Directory -Force $saveRoot, $dumpRoot, $logRoot | Out-Null

$arguments = @(
    "test.py",
    "--dataset", "mvtec",
    "--data_path", $dataRoot,
    "--save_path", $saveRoot,
    "--checkpoint_path", "checkpoints\9_12_4_multiscale_visa\epoch_15.pth",
    "--features_list", "24",
    "--image_size", [string]$ImageSize,
    "--depth", "9",
    "--n_ctx", "12",
    "--t_n_ctx", "4",
    "--dump_predictions", $dumpRoot
)

if ($Background) {
    $quoted = ($arguments | ForEach-Object {
        if ($_ -match '[\\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join " "
    Start-Process -FilePath $Python -ArgumentList $quoted -WorkingDirectory $MethodRoot `
        -WindowStyle Hidden -RedirectStandardOutput $logPath -RedirectStandardError $errorLogPath
    Write-Output "Started background AnomalyCLIP MVTec run."
    Write-Output "Log: $logPath"
    exit 0
}

Push-Location $MethodRoot
try {
    & $Python @arguments 2>&1 | Tee-Object -FilePath $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "AnomalyCLIP MVTec failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
