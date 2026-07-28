param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(0, 1, 2)]
    [int]$Seed,

    [Parameter(Mandatory = $true)]
    [ValidateSet(1, 2, 4)]
    [int]$Shot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
$categories = @(
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
)
$group = "seed_${Seed}_shot_${Shot}"
$manifest = Join-Path $projectRoot "data\splits\visa\manifest.json"
$source = Join-Path $projectRoot "data\visa_patchcore_all"
$dataRoot = Join-Path $projectRoot "data\visa_patchcore_fewshot\$group"
$outputRoot = Join-Path $projectRoot "outputs\patchcore\unified_matrix\$group"
$unifiedOutput = Join-Path $projectRoot "outputs\unified\patchcore_visa_$group"
$patchcoreRoot = Join-Path $projectRoot "methods\patchcore\patchcore-inspection-main"
$python = Join-Path $projectRoot ".venv-patchcore\Scripts\python.exe"
$metricPython = Join-Path $projectRoot ".venv-anomalyclip\Scripts\python.exe"
$expectedSummary = Join-Path $unifiedOutput "summary.csv"

if (Test-Path -LiteralPath $expectedSummary) {
    Write-Host "Already complete: $expectedSummary"
    exit 0
}

foreach ($category in $categories) {
    & python (Join-Path $projectRoot "scripts\prepare_patchcore_fewshot.py") `
        --manifest $manifest `
        --source $source `
        --target $dataRoot `
        --category $category `
        --shot $Shot `
        --seed $Seed
    if ($LASTEXITCODE -ne 0) {
        throw "Few-shot adapter failed for $category"
    }
}

& python (Join-Path $projectRoot "scripts\validate_dataset.py") `
    --dataset visa `
    --root $dataRoot `
    --output (Join-Path $projectRoot "outputs\logs\patchcore_$group")
if ($LASTEXITCODE -ne 0) {
    throw "Dataset validation failed for $group"
}

$datasetArgs = @()
foreach ($category in $categories) {
    $datasetArgs += @("-d", $category)
}
$oldPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = "src"
    Push-Location $patchcoreRoot
    & $python "bin\run_patchcore.py" `
        --gpu 0 `
        --seed $Seed `
        --dump_predictions `
        --log_group $group `
        --log_project visa_unified `
        $outputRoot `
        patch_core `
        -b wideresnet50 `
        -le layer2 `
        -le layer3 `
        --pretrain_embed_dimension 1024 `
        --target_embed_dimension 256 `
        --anomaly_scorer_num_nn 1 `
        --patchsize 3 `
        --faiss_num_workers 1 `
        sampler -p 0.1 approx_greedy_coreset `
        dataset `
        --resize 144 `
        --imagesize 128 `
        --batch_size 1 `
        --num_workers 0 `
        @datasetArgs `
        mvtec $dataRoot
    if ($LASTEXITCODE -ne 0) {
        throw "PatchCore failed for $group"
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $oldPythonPath
}

$predictionDir = Join-Path $outputRoot "visa_unified\$group\predictions"
if (-not (Test-Path -LiteralPath $predictionDir)) {
    throw "Prediction directory was not created: $predictionDir"
}
& $metricPython (Join-Path $projectRoot "scripts\evaluate_unified.py") `
    --cache-dir $predictionDir `
    --output-dir $unifiedOutput `
    --apro-steps 200
if ($LASTEXITCODE -ne 0) {
    throw "Unified evaluation failed for $group"
}

Write-Host "Completed $group"
Get-Content -LiteralPath $expectedSummary
