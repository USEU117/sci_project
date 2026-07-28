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
$winclipRoot = Join-Path $projectRoot "methods\winclip\WinClip-master"
$python = Join-Path $projectRoot ".venv-winclip\Scripts\python.exe"
$metricPython = Join-Path $projectRoot ".venv-anomalyclip\Scripts\python.exe"
$manifest = Join-Path $projectRoot "data\splits\visa\manifest.json"
$runRoot = Join-Path $projectRoot "outputs\winclip\unified_matrix\seed_${Seed}_shot_${Shot}"
$predictionDir = Join-Path $runRoot "visa-k-$Shot\seed-$Seed\predictions"
$unifiedOutput = Join-Path $projectRoot "outputs\unified\winclip_visa_seed_${Seed}_shot_${Shot}"
$summary = Join-Path $unifiedOutput "summary.csv"
$categories = @(
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"
)

if (Test-Path -LiteralPath $summary) {
    Write-Host "Already complete: $summary"
    exit 0
}

$previousManifest = $env:WINCLIP_UNIFIED_MANIFEST
try {
    $env:WINCLIP_UNIFIED_MANIFEST = $manifest
    Push-Location $winclipRoot
    foreach ($category in $categories) {
        $prediction = Join-Path $predictionDir "$category.npz"
        $selection = Join-Path $predictionDir "${category}_selection.json"
        if ((Test-Path -LiteralPath $prediction) -and
            (Test-Path -LiteralPath $selection)) {
            Write-Host "Skip completed category: $category"
            continue
        }
        & $python "eval_WinCLIP.py" `
            --dataset visa `
            --class-name $category `
            --k-shot $Shot `
            --experiment_indx $Seed `
            --batch-size 16 `
            --img-resize 240 `
            --img-cropsize 240 `
            --resolution 240 `
            --vis false `
            --cal-pro false `
            --dump-predictions true `
            --root-dir $runRoot `
            --use-cpu 0
        if ($LASTEXITCODE -ne 0) {
            throw "WinCLIP failed: seed=$Seed shot=$Shot category=$category"
        }
    }
}
finally {
    Pop-Location
    $env:WINCLIP_UNIFIED_MANIFEST = $previousManifest
}

$predictionCount = @(Get-ChildItem -LiteralPath $predictionDir -Filter "*.npz").Count
if ($predictionCount -ne $categories.Count) {
    throw "Expected $($categories.Count) predictions, found $predictionCount"
}

& $metricPython (Join-Path $projectRoot "scripts\evaluate_unified.py") `
    --cache-dir $predictionDir `
    --output-dir $unifiedOutput `
    --apro-steps 200
if ($LASTEXITCODE -ne 0) {
    throw "Unified evaluation failed: seed=$Seed shot=$Shot"
}

Write-Host "Completed WinCLIP seed=$Seed shot=$Shot"
Get-Content -LiteralPath $summary
