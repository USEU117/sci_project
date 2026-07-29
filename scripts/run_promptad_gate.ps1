param(
    [ValidateSet("cls", "seg")]
    [string]$Task = "cls",
    [string]$Category = "candle",
    [ValidateSet(1, 2, 4)]
    [int]$Shot = 1,
    [ValidateSet(0, 1, 2)]
    [int]$Seed = 0,
    [int]$Epoch = 100,
    [string]$Python = "",
    [string]$VisaDir = "",
    [string]$Manifest = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$MethodRoot = Join-Path $ProjectRoot "methods\promptad"

if (-not $Python) {
    $Python = Join-Path $ProjectRoot ".venv-anomalyclip\Scripts\python.exe"
}
if (-not $VisaDir) {
    $VisaDir = Join-Path $ProjectRoot "methods\winclip\datasets\VisA_pytorch\1cls"
}
if (-not $Manifest) {
    $Manifest = Join-Path $ProjectRoot "data\splits\visa\manifest.json"
}

foreach ($required in @($Python, $MethodRoot, $VisaDir, $Manifest)) {
    if (-not (Test-Path $required)) {
        throw "Required path not found: $required"
    }
}

$env:PROMPTAD_VISA_DIR = (Resolve-Path $VisaDir).Path
$env:PROMPTAD_SPLIT_MANIFEST = (Resolve-Path $Manifest).Path
$env:PROMPTAD_SPLIT_SEED = [string]$Seed

$entry = if ($Task -eq "cls") { "train_cls.py" } else { "train_seg.py" }
$arguments = @(
    $entry,
    "--dataset", "visa",
    "--class_name", $Category,
    "--k-shot", [string]$Shot,
    "--seed", [string]$Seed,
    "--Epoch", [string]$Epoch,
    "--gpu-id", "0",
    "--vis", "False"
)

Push-Location $MethodRoot
try {
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "PromptAD $Task failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
