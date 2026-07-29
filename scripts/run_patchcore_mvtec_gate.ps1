param(
    [ValidateSet(0, 1, 2)]
    [int]$Seed = 0,
    [ValidateSet(1, 2, 4)]
    [int]$Shot = 1,
    [string]$Category = "bottle"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$manifest = Join-Path $projectRoot "data\splits\mvtec\manifest.json"
$source = Join-Path $projectRoot "data\mvtec"
$group = "seed_${Seed}_shot_${Shot}"
$dataRoot = Join-Path $projectRoot "data\mvtec_patchcore_fewshot\$group"
$outputRoot = Join-Path $projectRoot "outputs\patchcore\mvtec_gate\$group"
$unifiedOutput = Join-Path $projectRoot "outputs\unified\patchcore_mvtec_${Category}_$group"
$patchcoreRoot = Join-Path $projectRoot "methods\patchcore\patchcore-inspection-main"
$python = Join-Path $projectRoot ".venv-patchcore\Scripts\python.exe"
$metricPython = Join-Path $projectRoot ".venv-anomalyclip\Scripts\python.exe"

& $metricPython (Join-Path $projectRoot "scripts\prepare_patchcore_fewshot.py") `
    --manifest $manifest --source $source --target $dataRoot `
    --category $Category --shot $Shot --seed $Seed
if ($LASTEXITCODE -ne 0) { throw "MVTec few-shot adapter failed" }

$datasetArgs = @("-d", $Category)
$oldPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = "src"
    Push-Location $patchcoreRoot
    & $python "bin\run_patchcore.py" `
        --gpu 0 --seed $Seed --dump_predictions `
        --log_group "mvtec_gate_$group" --log_project mvtec_gate `
        $outputRoot patch_core `
        -b wideresnet50 -le layer2 -le layer3 `
        --pretrain_embed_dimension 1024 --target_embed_dimension 256 `
        --anomaly_scorer_num_nn 1 --patchsize 3 --faiss_num_workers 1 `
        sampler -p 0.1 approx_greedy_coreset `
        dataset --resize 144 --imagesize 128 --batch_size 1 --num_workers 0 `
        @datasetArgs mvtec $dataRoot
    if ($LASTEXITCODE -ne 0) { throw "PatchCore MVTec Gate A failed" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $oldPythonPath
}

$predictionDir = Join-Path $outputRoot "mvtec_gate\mvtec_gate_$group\predictions"
& $metricPython (Join-Path $projectRoot "scripts\evaluate_unified.py") `
    --cache-dir $predictionDir --output-dir $unifiedOutput --apro-steps 200
if ($LASTEXITCODE -ne 0) { throw "PatchCore MVTec unified evaluation failed" }
Get-Content (Join-Path $unifiedOutput "summary.csv")
