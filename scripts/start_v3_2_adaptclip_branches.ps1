param(
    [switch]$ValidateOnly,
    [string]$RunId = 'v3_2_mpdd_s0_k1_branches'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Checkpoint = Join-Path $ProjectRoot 'methods\adaptclip\adaptclip_checkpoints\12_4_128_train_on_visa_3adapters_batch8\epoch_15.pth'
$ExpectedHash = '777821DA141EB57D159ACEF46868440FAF773A2DD0ACF5C276EC3F258C27EDEE'
$MetadataRoot = Join-Path $ProjectRoot 'experiments\dynamic_fusion\v3\stronger_text_branch_audit\mpdd_s0_k1'
$SourceData = Join-Path $ProjectRoot 'data\mpdd_raw\MPDD'
$RunRoot = Join-Path $ProjectRoot (Join-Path 'experiments\dynamic_fusion\v3' $RunId)
$StageRoot = Join-Path $RunRoot 'staged_mpdd_s0_k1'
$OutputRoot = Join-Path $ProjectRoot 'outputs\dynamic_fusion\v3_2_branches\v3_2_mpdd_s0_k1'
$MethodRoot = Join-Path $ProjectRoot 'methods\adaptclip'
$Python = Join-Path $ProjectRoot '.venv-adaptclip\Scripts\python.exe'
$CudaOverlay = Join-Path $ProjectRoot '.venv-anomalyclip\Lib\site-packages'

if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "Official AdaptCLIP checkpoint missing: $Checkpoint"
}
$ActualHash = (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash
if ($ActualHash -ne $ExpectedHash) {
    throw "Checkpoint SHA256 mismatch. Expected $ExpectedHash, got $ActualHash"
}
if (-not (Test-Path -LiteralPath (Join-Path $MetadataRoot 'meta.json') -PathType Leaf)) {
    throw 'Audited MPDD seed0/K1 metadata is missing.'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'The isolated AdaptCLIP Python environment is missing.'
}

if (Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'adaptclip.*test.py'
}) {
    throw 'An AdaptCLIP worker is already running; refusing to duplicate it.'
}

New-Item -ItemType Directory -Force -Path $StageRoot, $OutputRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $MetadataRoot 'meta.json') -Destination (Join-Path $StageRoot 'meta.json') -Force
$Metadata = Get-Content -LiteralPath (Join-Path $MetadataRoot 'meta.json') -Raw | ConvertFrom-Json

foreach ($Category in $Metadata.test.PSObject.Properties.Name) {
    $Target = Join-Path $SourceData $Category
    $Junction = Join-Path $StageRoot $Category
    if (-not (Test-Path -LiteralPath $Junction)) {
        New-Item -ItemType Junction -Path $Junction -Target $Target | Out-Null
    }
}

$env:PYTHONPATH = $CudaOverlay
$TestArgs = @(
    (Join-Path $MethodRoot 'test.py'),
    '--dataset', 'mpdd',
    '--test_data_path', $StageRoot,
    '--save_path', (Join-Path $RunRoot 'logs'),
    '--checkpoint_path', $Checkpoint,
    '--seed', '0', '--k_shots', '1', '--batch_size', '1',
    '--features_list', '6', '12', '18', '24', '--image_size', '518',
    '--n_ctx', '12', '--vl_reduction', '4', '--pq_mid_dim', '128',
    '--visual_learner', '--textual_learner', '--pq_learner', '--pq_context',
    '--prediction_cache_dir', $OutputRoot, '--sample_id_root', $SourceData,
    '--skip_metrics', '--export_branches'
)

Push-Location $MethodRoot
try {
    Write-Host "Starting V3.2 AdaptCLIP branch decomposition inference..."
    Write-Host "Output: $OutputRoot"
    & $Python @TestArgs
    if ($LASTEXITCODE -ne 0) { throw "V3.2 branch export exited with code $LASTEXITCODE" }
    Write-Host "V3.2 branch export completed successfully."
}
finally {
    Pop-Location
}
