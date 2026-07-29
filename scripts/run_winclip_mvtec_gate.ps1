param(
    [ValidateSet(0, 1, 2)]
    [int]$Seed = 0,
    [ValidateSet(1, 2, 4)]
    [int]$Shot = 1,
    [string]$Category = "bottle"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$winclipRoot = Join-Path $projectRoot "methods\winclip\WinClip-master"
$python = Join-Path $projectRoot ".venv-winclip\Scripts\python.exe"
$metricPython = Join-Path $projectRoot ".venv-anomalyclip\Scripts\python.exe"
$manifest = Join-Path $projectRoot "data\splits\mvtec\manifest.json"
$dataRoot = Join-Path $projectRoot "data\mvtec"
$runRoot = Join-Path $projectRoot "outputs\winclip\mvtec_gate\seed_${Seed}_shot_${Shot}"
$predictionDir = Join-Path $runRoot "mvtec-k-$Shot\seed-$Seed\predictions"
$unifiedOutput = Join-Path $projectRoot "outputs\unified\winclip_mvtec_${Category}_seed_${Seed}_shot_${Shot}"
$featureCacheDir = Join-Path $projectRoot "outputs\winclip\mvtec_test_feature_cache"

$oldManifest = $env:WINCLIP_UNIFIED_MANIFEST
$oldData = $env:WINCLIP_MVTEC_DIR
try {
    $env:WINCLIP_UNIFIED_MANIFEST = (Resolve-Path $manifest).Path
    $env:WINCLIP_MVTEC_DIR = (Resolve-Path $dataRoot).Path
    Push-Location $winclipRoot
    & $python "eval_WinCLIP_matrix.py" `
        --dataset mvtec --class-name $Category --k-shot $Shot `
        --experiment_indx $Seed --batch-size 16 `
        --img-resize 240 --img-cropsize 240 --resolution 240 `
        --vis false --cal-pro false --dump-predictions true `
        --feature-cache-dir $featureCacheDir --root-dir $runRoot --use-cpu 0
    if ($LASTEXITCODE -ne 0) { throw "WinCLIP MVTec Gate A failed" }
}
finally {
    Pop-Location
    $env:WINCLIP_UNIFIED_MANIFEST = $oldManifest
    $env:WINCLIP_MVTEC_DIR = $oldData
}

& $metricPython (Join-Path $projectRoot "scripts\evaluate_unified.py") `
    --cache-dir $predictionDir --output-dir $unifiedOutput --apro-steps 200
if ($LASTEXITCODE -ne 0) { throw "WinCLIP MVTec unified evaluation failed" }
Get-Content (Join-Path $unifiedOutput "summary.csv")
