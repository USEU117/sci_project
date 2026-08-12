param(
    [string]$RunId = "20260731_visa_s0_k1_calibrated_development_matrix",
    [int]$Workers = 1,
    [ValidateSet(1, 2, 4)]
    [int]$Shot = 1,
    [string]$CalibrationJson = "",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# This is deliberately CPU-only.  It is a seed-0 development comparison, not a
# final experiment: visual/text/dynamic and all fixed weights are predeclared.
$python = Join-Path $root ".venv-anomalyclip\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Missing Python environment: $python" }
if ($Workers -lt 1) { throw "Workers must be at least 1." }

$calibration = if ([string]::IsNullOrWhiteSpace($CalibrationJson)) {
    Join-Path $root "outputs\dynamic_fusion\normal_reference_predictions\20260731_visa_s0_k1_real_reference_v6_q99\calibration.json"
} else {
    Join-Path $root $CalibrationJson
}
if (-not (Test-Path -LiteralPath $calibration)) { throw "Missing passed calibration: $calibration" }
$calibrationPayload = Get-Content -LiteralPath $calibration -Raw -Encoding utf8 | ConvertFrom-Json
if ($calibrationPayload.status -ne "passed" -or $calibrationPayload.test_predictions_used -or $calibrationPayload.test_labels_used) {
    throw "Calibration must be passed and normal-reference-only."
}

$visualDir = Join-Path $root "outputs\anomalydino\unified_matrix\seed_0_shot_1\predictions"
$textDir = Join-Path $root "outputs\anomalyclip\visa_all_518_cached"
$sidecarDir = Join-Path $root "outputs\dynamic_fusion\sidecars\anomalyclip_visa_518_verified"
$categories = @("candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum")
foreach ($category in $categories) {
    foreach ($path in @(
        (Join-Path $visualDir "$category.npz"),
        (Join-Path $textDir "$category.npz"),
        (Join-Path $sidecarDir "$category.sample_ids.npz")
    )) {
        if (-not (Test-Path -LiteralPath $path)) { throw "Missing frozen input: $path" }
    }
}

$experimentDir = Join-Path $root "experiments\dynamic_fusion\$RunId"
$outputRoot = Join-Path $root "outputs\dynamic_fusion\development_matrix\$RunId"
if (-not $Resume -and ((Test-Path -LiteralPath $experimentDir) -or (Test-Path -LiteralPath $outputRoot))) {
    throw "RunId already exists; choose a new RunId to avoid overwriting evidence: $RunId"
}
if ($Resume) {
    if (-not (Test-Path -LiteralPath $experimentDir) -or -not (Test-Path -LiteralPath $outputRoot)) {
        throw "Resume requires an existing experiment and output directory: $RunId"
    }
    $attemptId = "resume_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    $attemptDir = Join-Path $experimentDir $attemptId
    New-Item -ItemType Directory -Force -Path $attemptDir | Out-Null
} else {
    New-Item -ItemType Directory -Force -Path $experimentDir, $outputRoot | Out-Null
    $attemptDir = $experimentDir
}

$gitHead = (& git rev-parse HEAD 2>$null)
$gitDirty = [bool](& git status --porcelain 2>$null)
$manifest = [ordered]@{
    schema_version = 1
    run_id = $RunId
    purpose = "VisA seed-0 $Shot-shot calibrated development comparison; not final validation"
    dataset = "visa"
    seed = 0
    shot = $Shot
    categories = $categories
    visual_branch = "anomalydino_visual"
    text_branch = "anomalyclip_text"
    gpu_used = $false
    test_predictions_used_by_router = $false
    test_labels_used_by_router = $false
    evaluation_uses_ground_truth_only_after_fusion = $true
    calibration = (Resolve-Path -LiteralPath $calibration).Path
    calibration_sha256 = (Get-FileHash -LiteralPath $calibration -Algorithm SHA256).Hash.ToLowerInvariant()
    fixed_visual_weights = @(0.0, 0.25, 0.5, 0.75, 1.0)
    git_commit = $gitHead
    git_dirty = $gitDirty
    started_at_utc = [DateTime]::UtcNow.ToString("o")
}
$runFile = if ($Resume) { Join-Path $attemptDir "run.json" } else { Join-Path $experimentDir "run.json" }
$manifest["resume"] = [bool]$Resume
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $runFile -Encoding utf8
if (-not $Resume) {
    Copy-Item -LiteralPath "configs\dynamic_fusion.yaml" -Destination (Join-Path $experimentDir "config.yaml")
} else {
    Copy-Item -LiteralPath "configs\dynamic_fusion.yaml" -Destination (Join-Path $attemptDir "config.yaml")
}

$modes = @(
    @{ Name = "visual"; Arguments = @("--fusion-mode", "visual") },
    @{ Name = "text"; Arguments = @("--fusion-mode", "text") },
    @{ Name = "fixed_w0"; Arguments = @("--fusion-mode", "fixed", "--image-visual-weight", "0.0") },
    @{ Name = "fixed_w025"; Arguments = @("--fusion-mode", "fixed", "--image-visual-weight", "0.25") },
    @{ Name = "fixed_w05"; Arguments = @("--fusion-mode", "fixed", "--image-visual-weight", "0.5") },
    @{ Name = "fixed_w075"; Arguments = @("--fusion-mode", "fixed", "--image-visual-weight", "0.75") },
    @{ Name = "fixed_w1"; Arguments = @("--fusion-mode", "fixed", "--image-visual-weight", "1.0") },
    @{ Name = "dynamic"; Arguments = @("--fusion-mode", "dynamic") }
)

$commands = New-Object System.Collections.Generic.List[string]
try {
    foreach ($mode in $modes) {
        $cacheDir = Join-Path $outputRoot $mode.Name
        New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
        foreach ($category in $categories) {
            $outputPath = Join-Path $cacheDir "$category.npz"
            $arguments = @(
                "scripts/run_dynamic_fusion_cache.py",
                "--visual-cache", (Join-Path $visualDir "$category.npz"),
                "--text-cache", (Join-Path $textDir "$category.npz"),
                "--text-sidecar", (Join-Path $sidecarDir "$category.sample_ids.npz"),
                "--calibration-json", $calibration,
                "--category", $category,
                "--output", $outputPath
            ) + $mode.Arguments
            $commands.Add(("& `"$python`" " + (($arguments | ForEach-Object { "`"$_`"" }) -join " ")))
            if ($Resume -and (Test-Path -LiteralPath $outputPath) -and (Get-Item -LiteralPath $outputPath).Length -gt 0) {
                $commands.Add("SKIP existing non-empty cache: $outputPath")
                continue
            }
            & $python @arguments
            if ($LASTEXITCODE -ne 0) { throw "Fusion failed: $($mode.Name)/$category" }
        }
        $evaluationDir = Join-Path $cacheDir "evaluation"
        $evalArgs = @("scripts/evaluate_unified.py", "--cache-dir", $cacheDir, "--output-dir", $evaluationDir, "--workers", "$Workers")
        $commands.Add(("& `"$python`" " + (($evalArgs | ForEach-Object { "`"$_`"" }) -join " ")))
        $evaluationFiles = @("summary.csv", "per_category.csv", "per_image.csv", "evaluation_report.json")
        $evaluationComplete = $Resume -and (($evaluationFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $evaluationDir $_)) }).Count -eq 0)
        if ($evaluationComplete) {
            $commands.Add("SKIP complete evaluation: $evaluationDir")
        } else {
            & $python @evalArgs
            if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $($mode.Name)" }
        }
    }
    $summary = foreach ($mode in $modes) {
        $row = Import-Csv -LiteralPath (Join-Path $outputRoot "$($mode.Name)\evaluation\summary.csv") | Select-Object -First 1
        [pscustomobject]@{ mode = $mode.Name; image_auroc = $row.image_auroc; pixel_auroc = $row.pixel_auroc; pixel_ap = $row.pixel_ap; aupro = $row.aupro }
    }
    $summary | Export-Csv -LiteralPath (Join-Path $attemptDir "report.csv") -NoTypeInformation -Encoding utf8
    [ordered]@{ schema_version = 1; run_id = $RunId; status = "passed"; resumed = [bool]$Resume; scope = "development_only_seed_0"; calibration_status = $calibrationPayload.status; modes = $summary; finished_at_utc = [DateTime]::UtcNow.ToString("o") } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $attemptDir "report.json") -Encoding utf8
    @("# Development decision", "", "- Status: passed.", "- Resume mode: $([bool]$Resume).", "- This is a VisA seed 0 development comparison only; do not use it as final generalization evidence.", "- Fixed-weight candidates were declared before reading evaluation metrics.", "- Next: freeze the selected design without using VisA seed 1/2 or MVTec outcomes, then run the independent validation queue once.") | Set-Content -LiteralPath (Join-Path $attemptDir "decision.md") -Encoding utf8
}
catch {
    [ordered]@{ schema_version = 1; run_id = $RunId; status = "failed"; resumed = [bool]$Resume; failure = $_.Exception.Message; finished_at_utc = [DateTime]::UtcNow.ToString("o") } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $attemptDir "report.json") -Encoding utf8
    throw
}
finally {
    $commands | Set-Content -LiteralPath (Join-Path $attemptDir "command.txt") -Encoding utf8
}
