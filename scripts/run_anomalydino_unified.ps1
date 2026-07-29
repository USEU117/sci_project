param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(0, 1, 2)]
    [int]$Seed,

    [Parameter(Mandatory = $true)]
    [ValidateSet(1, 2, 4)]
    [int]$Shot,

    [string[]]$Objects = @(
        "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
    )
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$methodRoot = Join-Path $projectRoot "methods\anomalydino"
$python = Join-Path $projectRoot ".venv-patchcore\Scripts\python.exe"
$metricPython = Join-Path $projectRoot ".venv-anomalyclip\Scripts\python.exe"
$dataRoot = Join-Path $projectRoot "methods\winclip\datasets\VisA_pytorch\1cls"
$manifest = Join-Path $projectRoot "data\splits\visa\manifest.json"
$featureCache = Join-Path $projectRoot "outputs\anomalydino\test_feature_cache\dinov2_vits14_448"
$tag = "unified_s${Seed}_k${Shot}"
$methodOutput = Join-Path $methodRoot "results_VisA\dinov2_vits14_448\${Shot}-shot_preprocess=agnostic_$tag"
$anomalyDir = Join-Path $methodOutput "anomaly_maps\seed=$Seed"
$predictionDir = Join-Path $projectRoot "outputs\anomalydino\unified_matrix\seed_${Seed}_shot_${Shot}\predictions"
$unifiedOutput = Join-Path $projectRoot "outputs\unified\anomalydino_visa_seed_${Seed}_shot_${Shot}"
$summary = Join-Path $unifiedOutput "summary.csv"
$evaluationReport = Join-Path $unifiedOutput "evaluation_report.json"

if ((Test-Path -LiteralPath $summary) -and (Test-Path -LiteralPath $evaluationReport)) {
    $report = Get-Content -LiteralPath $evaluationReport -Raw | ConvertFrom-Json
    if ([int]$report.category_count -eq $Objects.Count) {
        Write-Host "Already complete: $summary"
        exit 0
    }
    Write-Host "Existing output is a partial gate ($($report.category_count)/$($Objects.Count)); continuing full run."
}

Push-Location $methodRoot
try {
    & $python "run_anomalydino.py" `
        --dataset VisA `
        --objects $Objects `
        --shots $Shot `
        --just_seed $Seed `
        --preprocess agnostic `
        --data_root $dataRoot `
        --faiss_on_cpu `
        --no-save_examples `
        --warmup_iters 1 `
        --split_manifest $manifest `
        --feature_cache_dir $featureCache `
        --map_max_edge 448 `
        --save_tiffs `
        --tag $tag
    if ($LASTEXITCODE -ne 0) {
        throw "AnomalyDINO failed: seed=$Seed shot=$Shot"
    }
}
finally {
    Pop-Location
}

foreach ($category in $Objects) {
    $prediction = Join-Path $predictionDir "$category.npz"
    if (Test-Path -LiteralPath $prediction) {
        Write-Host "Skip converted category: $category"
        continue
    }
    & $python (Join-Path $projectRoot "scripts\convert_anomalydino_predictions.py") `
        --data-root $dataRoot `
        --anomaly-dir $anomalyDir `
        --category $category `
        --output $prediction
    if ($LASTEXITCODE -ne 0) {
        throw "AnomalyDINO conversion failed: seed=$Seed shot=$Shot category=$category"
    }
}

$predictionCount = @(Get-ChildItem -LiteralPath $predictionDir -Filter "*.npz").Count
if ($predictionCount -ne $Objects.Count) {
    throw "Expected $($Objects.Count) predictions, found $predictionCount"
}

& $metricPython (Join-Path $projectRoot "scripts\evaluate_unified.py") `
    --cache-dir $predictionDir `
    --output-dir $unifiedOutput `
    --apro-steps 200 `
    --workers 4
if ($LASTEXITCODE -ne 0) {
    throw "Unified evaluation failed: seed=$Seed shot=$Shot"
}

Write-Host "Completed AnomalyDINO seed=$Seed shot=$Shot"
Get-Content -LiteralPath $summary
