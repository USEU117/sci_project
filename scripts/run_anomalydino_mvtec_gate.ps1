param(
    [ValidateSet(0, 1, 2)]
    [int]$Seed = 0,
    [ValidateSet(1, 2, 4)]
    [int]$Shot = 1,
    [string[]]$Objects = @("bottle")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$methodRoot = Join-Path $projectRoot "methods\anomalydino"
$python = Join-Path $projectRoot ".venv-patchcore\Scripts\python.exe"
$metricPython = Join-Path $projectRoot ".venv-anomalyclip\Scripts\python.exe"
$dataRoot = Join-Path $projectRoot "data\mvtec"
$manifest = Join-Path $projectRoot "data\splits\mvtec\manifest.json"
$tag = "mvtec_gate_s${Seed}_k${Shot}"
$methodOutput = Join-Path $methodRoot "results_MVTec\dinov2_vits14_448\${Shot}-shot_preprocess=agnostic_$tag"
$anomalyDir = Join-Path $methodOutput "anomaly_maps\seed=$Seed"
$predictionDir = Join-Path $projectRoot "outputs\anomalydino\mvtec_gate\seed_${Seed}_shot_${Shot}\predictions"
$unifiedOutput = Join-Path $projectRoot "outputs\unified\anomalydino_mvtec_gate_s${Seed}_k${Shot}"

Push-Location $methodRoot
try {
    & $python "run_anomalydino.py" --dataset MVTec --objects $Objects --shots $Shot `
        --just_seed $Seed --preprocess agnostic --data_root $dataRoot --faiss_on_cpu `
        --no-save_examples --warmup_iters 1 --split_manifest $manifest `
        --feature_cache_dir (Join-Path $projectRoot "outputs\anomalydino\mvtec_test_feature_cache") `
        --map_max_edge 448 --save_tiffs --tag $tag
    if ($LASTEXITCODE -ne 0) { throw "AnomalyDINO MVTec failed" }
} finally { Pop-Location }

foreach ($category in $Objects) {
    $prediction = Join-Path $predictionDir "$category.npz"
    if (!(Test-Path $prediction)) {
        & $python (Join-Path $projectRoot "scripts\convert_anomalydino_predictions.py") `
            --data-root $dataRoot --anomaly-dir $anomalyDir --category $category --output $prediction
        if ($LASTEXITCODE -ne 0) { throw "Conversion failed: $category" }
    }
}
& $metricPython (Join-Path $projectRoot "scripts\evaluate_unified.py") `
    --cache-dir $predictionDir --output-dir $unifiedOutput --apro-steps 200 --workers 4
if ($LASTEXITCODE -ne 0) { throw "Unified evaluation failed" }
Get-Content (Join-Path $unifiedOutput "summary.csv")
