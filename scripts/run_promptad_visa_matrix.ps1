param(
    [ValidateSet(0, 1, 2)]
    [int]$Seed = 0,
    [ValidateSet(1, 2, 4)]
    [int]$Shot = 1,
    [int]$Epoch = 100,
    [string[]]$Categories = @(
        "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
    )
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$methodRoot = Join-Path $projectRoot "methods\promptad"
$python = Join-Path $projectRoot ".venv-promptad\Scripts\python.exe"
$metricPython = Join-Path $projectRoot ".venv-anomalyclip\Scripts\python.exe"
$visaDir = Join-Path $projectRoot "methods\winclip\datasets\VisA_pytorch\1cls"
$manifest = Join-Path $projectRoot "data\splits\visa\manifest.json"
$rawDir = Join-Path $projectRoot "outputs\promptad\visa\seed_${Seed}_shot_${Shot}\raw"
$predictionDir = Join-Path $projectRoot "outputs\promptad\visa\seed_${Seed}_shot_${Shot}\predictions"
$markerDir = Join-Path $projectRoot "outputs\logs\promptad\visa\seed_${Seed}_shot_${Shot}"

New-Item -ItemType Directory -Force -Path $rawDir, $predictionDir, $markerDir | Out-Null
$env:PROMPTAD_VISA_DIR = $visaDir
$env:PROMPTAD_SPLIT_MANIFEST = $manifest
$env:PROMPTAD_SPLIT_SEED = [string]$Seed

foreach ($category in $Categories) {
    foreach ($task in @("cls", "seg")) {
        $marker = Join-Path $markerDir "${category}_${task}.complete"
        if (!(Test-Path $marker)) {
            & (Join-Path $PSScriptRoot "run_promptad_gate.ps1") -Task $task `
                -Category $category -Shot $Shot -Seed $Seed -Epoch $Epoch `
                -Python $python -VisaDir $visaDir -Manifest $manifest
            if ($LASTEXITCODE -ne 0) { throw "PromptAD training failed: $category $task" }
            Set-Content -LiteralPath $marker -Value (Get-Date -Format o)
        }
        $raw = Join-Path $rawDir "${category}_${task}.npz"
        if (!(Test-Path $raw)) {
            $env:PROMPTAD_EXPORT_NPZ = $raw
            Push-Location $methodRoot
            try {
                $entry = if ($task -eq "cls") { "test_cls.py" } else { "test_seg.py" }
                & $python $entry --dataset visa --class_name $category --k-shot $Shot `
                    --seed $Seed --gpu-id 0 --vis False
                if ($LASTEXITCODE -ne 0) { throw "PromptAD export failed: $category $task" }
            } finally { Pop-Location }
        }
    }
    & $python (Join-Path $PSScriptRoot "merge_promptad_predictions.py") `
        --classification (Join-Path $rawDir "${category}_cls.npz") `
        --segmentation (Join-Path $rawDir "${category}_seg.npz") `
        --output (Join-Path $predictionDir "$category.npz")
    if ($LASTEXITCODE -ne 0) { throw "PromptAD merge failed: $category" }
}

& $metricPython (Join-Path $PSScriptRoot "evaluate_unified.py") `
    --cache-dir $predictionDir `
    --output-dir (Join-Path $projectRoot "outputs\unified\promptad_visa_seed_${Seed}_shot_${Shot}") `
    --apro-steps 200 --workers 4
if ($LASTEXITCODE -ne 0) { throw "PromptAD unified evaluation failed" }
