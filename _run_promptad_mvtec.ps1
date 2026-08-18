param(
    [ValidateSet(0, 1, 2)]
    [int]$Seed = 0,
    [ValidateSet(1, 2, 4)]
    [int]$Shot = 1,
    [int]$Epoch = 100,
    [string[]]$Categories = @(
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
    )
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = $PSScriptRoot
$methodRoot = Join-Path $projectRoot "methods\promptad"
$python = Join-Path $projectRoot ".venv-promptad\Scripts\python.exe"
$metricPython = Join-Path $projectRoot ".venv-anomalyclip\Scripts\python.exe"
$mvtecDir = Join-Path $projectRoot "data\mvtec"
$manifest = Join-Path $projectRoot "data\splits\mvtec\manifest.json"
$rawDir = Join-Path $projectRoot "outputs\promptad\mvtec\seed_${Seed}_shot_${Shot}\raw"
$predictionDir = Join-Path $projectRoot "outputs\promptad\mvtec\seed_${Seed}_shot_${Shot}\predictions"
$markerDir = Join-Path $projectRoot "outputs\logs\promptad\mvtec\seed_${Seed}_shot_${Shot}"

foreach ($required in @($python, $metricPython, $methodRoot, $mvtecDir, $manifest)) {
    if (-not (Test-Path $required)) {
        throw "Required path not found: $required"
    }
}

New-Item -ItemType Directory -Force -Path $rawDir, $predictionDir, $markerDir | Out-Null
$env:PROMPTAD_MVTEC_DIR = (Resolve-Path $mvtecDir).Path
$env:PROMPTAD_SPLIT_MANIFEST = (Resolve-Path $manifest).Path
$env:PROMPTAD_SPLIT_SEED = [string]$Seed

foreach ($category in $Categories) {
    foreach ($task in @("cls", "seg")) {
        $marker = Join-Path $markerDir "${category}_${task}.complete"
        if (!(Test-Path $marker)) {
            $entry = if ($task -eq "cls") { "train_cls.py" } else { "train_seg.py" }
            Push-Location $methodRoot
            try {
                & $python $entry --dataset mvtec --class_name $category --k-shot $Shot `
                    --seed $Seed --Epoch $Epoch --gpu-id 0 --vis False
                if ($LASTEXITCODE -ne 0) { throw "PromptAD training failed: $category $task" }
            } finally { Pop-Location }
            Set-Content -LiteralPath $marker -Value (Get-Date -Format o)
        }
        $raw = Join-Path $rawDir "${category}_${task}.npz"
        if (!(Test-Path $raw)) {
            $env:PROMPTAD_EXPORT_NPZ = $raw
            Push-Location $methodRoot
            try {
                $entry = if ($task -eq "cls") { "test_cls.py" } else { "test_seg.py" }
                & $python $entry --dataset mvtec --class_name $category --k-shot $Shot `
                    --seed $Seed --gpu-id 0 --vis False
                if ($LASTEXITCODE -ne 0) { throw "PromptAD export failed: $category $task" }
            } finally { Pop-Location }
        }
    }
    & $python (Join-Path $PSScriptRoot "scripts\merge_promptad_predictions.py") `
        --classification (Join-Path $rawDir "${category}_cls.npz") `
        --segmentation (Join-Path $rawDir "${category}_seg.npz") `
        --output (Join-Path $predictionDir "$category.npz")
    if ($LASTEXITCODE -ne 0) { throw "PromptAD merge failed: $category" }
}

& $metricPython (Join-Path $PSScriptRoot "scripts\evaluate_unified.py") `
    --cache-dir $predictionDir `
    --output-dir (Join-Path $projectRoot "outputs\unified\promptad_mvtec_seed_${Seed}_shot_${Shot}") `
    --apro-steps 200 --workers 4
if ($LASTEXITCODE -ne 0) { throw "PromptAD unified evaluation failed" }
