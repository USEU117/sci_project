param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$Workspace = (Resolve-Path -LiteralPath $Workspace).Path
$logDir = Join-Path $Workspace "outputs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$existing = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "curl.exe" -and $_.CommandLine -match "btad.zip|MPDD.zip"
}
if ($existing) {
    throw "Existing dataset curl process found: $($existing.ProcessId -join ',')"
}

$btadArgs = @(
    "-L", "--fail", "--retry", "20", "--retry-delay", "10",
    "--continue-at", "-", "--silent", "--show-error", "--output",
    (Join-Path $Workspace "data\downloads\btad.zip"),
    "https://avires.dimi.uniud.it/papers/btad/btad.zip"
)
$mpddArgs = @(
    "-L", "--fail", "--retry", "20", "--retry-delay", "10",
    "--continue-at", "-", "--silent", "--show-error", "--output",
    (Join-Path $Workspace "data\downloads\MPDD.zip"),
    "https://huggingface.co/datasets/meksamiao/mpdd/resolve/main/MPDD.zip?download=true"
)

$btad = Start-Process -FilePath "curl.exe" -ArgumentList $btadArgs `
    -RedirectStandardOutput (Join-Path $logDir "btad_download.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "btad_download.stderr.log") `
    -WindowStyle Hidden -PassThru
$mpdd = Start-Process -FilePath "curl.exe" -ArgumentList $mpddArgs `
    -RedirectStandardOutput (Join-Path $logDir "mpdd_download.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "mpdd_download.stderr.log") `
    -WindowStyle Hidden -PassThru
$preparation = Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
    (Join-Path $Workspace "scripts\complete_v2_data_preparation.ps1"),
    "-Workspace", $Workspace, "-PollSeconds", "30"
) -RedirectStandardOutput (Join-Path $logDir "v2_data_preparation.stdout.log") `
    -RedirectStandardError (Join-Path $logDir "v2_data_preparation.stderr.log") `
    -WindowStyle Hidden -PassThru

[ordered]@{
    started_at = (Get-Date).ToString("o")
    btad_download_pid = $btad.Id
    mpdd_download_pid = $mpdd.Id
    preparation_pid = $preparation.Id
    workspace = $Workspace
} | ConvertTo-Json
