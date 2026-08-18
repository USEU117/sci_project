param(
    [switch]$ValidateOnly
)

# Build a full 15-category MVTec staged root for AdaptCLIP complete-matrix runs.
# Mirrors the single-category staging used by start_adaptclip_mvtec_gate_a.ps1,
# but keeps every category in meta.json so one serial process tests all classes.

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SourceData = Join-Path $ProjectRoot 'data\mvtec'
$StageRoot = Join-Path $ProjectRoot 'experiments\dynamic_fusion\v3\baselines\adaptclip_mvtec\staged_mvtec_full'

$SourceMeta = Join-Path $SourceData 'meta.json'
if (-not (Test-Path -LiteralPath $SourceMeta -PathType Leaf)) {
    throw "MVTec meta.json missing: $SourceMeta"
}

$Meta = Get-Content -LiteralPath $SourceMeta -Raw | ConvertFrom-Json
$TrainCats = @($Meta.train.PSObject.Properties.Name | Sort-Object)
$TestCats = @($Meta.test.PSObject.Properties.Name | Sort-Object)
if ($TrainCats.Count -ne 15 -or $TestCats.Count -ne 15) {
    throw "Expected 15 MVTec categories in train/test, got $($TrainCats.Count)/$($TestCats.Count)"
}
$Mismatch = @($TrainCats | Where-Object { $_ -notin $TestCats }) + @($TestCats | Where-Object { $_ -notin $TrainCats })
if ($Mismatch.Count -gt 0) {
    throw "train/test category mismatch: $($Mismatch -join ', ')"
}

if ($ValidateOnly) {
    Write-Host "Full MVTec staging preflight OK: $($TrainCats.Count) categories."
    Write-Host "Stage root: $StageRoot"
    exit 0
}

New-Item -ItemType Directory -Force -Path $StageRoot | Out-Null
$Staged = @{ train = $Meta.train; test = $Meta.test }
$StagedJson = ConvertTo-Json -InputObject $Staged -Depth 12
[System.IO.File]::WriteAllText(
    (Join-Path $StageRoot 'meta.json'),
    $StagedJson,
    [System.Text.UTF8Encoding]::new($false)
)

$Junctions = 0
foreach ($Category in $TrainCats) {
    $Junction = Join-Path $StageRoot $Category
    if (-not (Test-Path -LiteralPath $Junction)) {
        New-Item -ItemType Junction -Path $Junction -Target (Join-Path $SourceData $Category) | Out-Null
        $Junctions++
    }
}
Write-Host "Full MVTec staged root ready at $StageRoot (new junctions: $Junctions)."
