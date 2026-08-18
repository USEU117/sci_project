param(
    [switch]$ValidateOnly,
    [int]$BatchSize = 1
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$MethodRoot = Join-Path $ProjectRoot 'methods\remp_ad'
$Python = Join-Path $ProjectRoot '.venv-remp_ad\Scripts\python.exe'
$TrainData = Join-Path $ProjectRoot 'data\visa'
$TestData = Join-Path $ProjectRoot 'data\mvtec'
$ConfigPath = Join-Path $MethodRoot 'config\mvtec.yaml'

# ---- Prerequisite checks ----
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'ReMP-AD Python environment missing.'
}
if (-not (Test-Path -LiteralPath (Join-Path $TrainData 'meta.json') -PathType Leaf)) {
    throw "VisA training data missing meta.json: $TrainData"
}
if (-not (Test-Path -LiteralPath (Join-Path $TrainData 'split_csv\1cls.csv') -PathType Leaf)) {
    throw "VisA split_csv missing: $(Join-Path $TrainData 'split_csv\1cls.csv')"
}
if (-not (Test-Path -LiteralPath (Join-Path $TestData 'meta.json') -PathType Leaf)) {
    throw "MVTec test data missing meta.json: $TestData"
}
if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "ReMP-AD config missing: $ConfigPath"
}

if ($ValidateOnly) {
    Write-Host "ReMP-AD MVTec preflight OK: batch_size=$BatchSize"
    Write-Host "Train data: $TrainData"
    Write-Host "Test data : $TestData"
    exit 0
}

if (Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'remp_ad'
}) {
    throw 'A ReMP-AD worker is already running; refusing to duplicate it.'
}

Push-Location $MethodRoot
try {
    # 1. Train linear layer on VisA (cross-domain: train VisA -> test MVTec).
    Write-Host "ReMP-AD: training on VisA (batch_size=$BatchSize)..."
    & $Python train.py --config_path $ConfigPath --train_data_path $TrainData --batch_size $BatchSize
    if ($LASTEXITCODE -ne 0) { throw "ReMP-AD train exited with code $LASTEXITCODE" }

    # 2. Test on MVTec for k_shot in 4/2/1.
    foreach ($KShot in 4, 2, 1) {
        Write-Host "ReMP-AD: testing on MVTec (k_shot=$KShot)..."
        & $Python test.py --config_path $ConfigPath --test_data_path $TestData --k_shot $KShot
        if ($LASTEXITCODE -ne 0) { throw "ReMP-AD test (k_shot=$KShot) exited with code $LASTEXITCODE" }
    }
    Write-Host "ReMP-AD MVTec baseline completed."
}
finally {
    Pop-Location
}
