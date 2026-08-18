param(
    [switch]$ValidateOnly,
    [int]$KShots = 1,
    [int]$Seed = 10,
    [string]$Category = 'bottle'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Checkpoint = Join-Path $ProjectRoot 'methods\adaptclip\adaptclip_checkpoints\12_4_128_train_on_visa_3adapters_batch8\epoch_15.pth'
$ExpectedHash = '777821DA141EB57D159ACEF46868440FAF773A2DD0ACF5C276EC3F258C27EDEE'
$SourceData = Join-Path $ProjectRoot 'data\mvtec'
$RunRoot = Join-Path $ProjectRoot 'experiments\dynamic_fusion\v3\baselines\adaptclip_mvtec'
$StageRoot = Join-Path $RunRoot "staged_mvtec_${Category}"
$MethodRoot = Join-Path $ProjectRoot 'methods\adaptclip'
$Python = Join-Path $ProjectRoot '.venv-adaptclip\Scripts\python.exe'
$CudaOverlay = Join-Path $ProjectRoot '.venv-anomalyclip\Lib\site-packages'

# ---- Prerequisite checks ----
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "Official AdaptCLIP checkpoint missing: $Checkpoint"
}
$ActualHash = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash
if ($ActualHash -ne $ExpectedHash) {
    throw "Checkpoint SHA256 mismatch. Expected $ExpectedHash, got $ActualHash"
}
$SourceMetaPath = Join-Path $SourceData 'meta.json'
if (-not (Test-Path -LiteralPath $SourceMetaPath -PathType Leaf)) {
    throw "MVTec meta.json missing: $SourceMetaPath"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'AdaptCLIP Python environment missing.'
}

# ---- Build a single-category staged root (Gate A smoke test) ----
$Meta = Get-Content -LiteralPath $SourceMetaPath -Raw | ConvertFrom-Json
if (-not $Meta.train.$Category -or -not $Meta.test.$Category) {
    throw "Category '$Category' not found in MVTec meta.json (train/test)."
}
New-Item -ItemType Directory -Force -Path $StageRoot | Out-Null
$Staged = @{ train = @{}; test = @{} }
$Staged.train.$Category = $Meta.train.$Category
$Staged.test.$Category = $Meta.test.$Category
$StagedJson = ConvertTo-Json -InputObject $Staged -Depth 12
[System.IO.File]::WriteAllText(
    (Join-Path $StageRoot 'meta.json'),
    $StagedJson,
    [System.Text.UTF8Encoding]::new($false)
)
$Junction = Join-Path $StageRoot $Category
if (-not (Test-Path -LiteralPath $Junction)) {
    New-Item -ItemType Junction -Path $Junction -Target (Join-Path $SourceData $Category) | Out-Null
}

if ($ValidateOnly) {
    Write-Host "AdaptCLIP MVTec Gate A preflight OK: category=$Category k_shots=$KShots seed=$Seed batch_size=1"
    Write-Host "Staged root: $StageRoot"
    exit 0
}

if (Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'adaptclip.*test.py'
}) {
    throw 'An AdaptCLIP worker is already running; refusing to duplicate it.'
}

$TestArgs = @(
    (Join-Path $MethodRoot 'test.py'),
    '--dataset', 'mvtec',
    '--test_data_path', $StageRoot,
    '--save_path', (Join-Path $RunRoot 'logs'),
    '--checkpoint_path', $Checkpoint,
    '--seed', "$Seed", '--k_shots', "$KShots", '--batch_size', '1',
    '--features_list', '6', '12', '18', '24', '--image_size', '518',
    '--n_ctx', '12', '--vl_reduction', '4', '--pq_mid_dim', '128',
    '--visual_learner', '--textual_learner', '--pq_learner', '--pq_context'
)

Push-Location $MethodRoot
try {
    Write-Host "Starting AdaptCLIP MVTec Gate A (category=$Category, batch_size=1)..."
    & $Python @TestArgs
    if ($LASTEXITCODE -ne 0) { throw "AdaptCLIP Gate A exited with code $LASTEXITCODE" }
    Write-Host "AdaptCLIP MVTec Gate A completed."
}
finally {
    Pop-Location
}
