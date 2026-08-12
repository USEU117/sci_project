param(
    [datetime]$Cutoff = [datetime]::Parse("2026-08-11T14:45:00+08:00")
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv-patchcore\Scripts\python.exe"
$runner = Join-Path $projectRoot "scripts\run_v2_branch_cache_queue.py"
$queue = Join-Path $projectRoot "experiments\dynamic_fusion\v2\branch_cache_queue\queue.json"
$runtimeRoot = Join-Path $projectRoot "experiments\dynamic_fusion\v2\branch_cache_queue\runtime"
$status = Join-Path $runtimeRoot "status.json"
$logRoot = Join-Path $runtimeRoot "logs"
$launcherStdout = Join-Path $runtimeRoot "launcher.stdout.log"
$launcherStderr = Join-Path $runtimeRoot "launcher.stderr.log"

New-Item -ItemType Directory -Force $runtimeRoot, $logRoot | Out-Null
$live = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and
    $_.CommandLine -match 'run_v2_branch_cache_queue.py' -and
    $_.ProcessId -ne $PID
}
if ($live) {
    throw "A V2 branch-cache queue runner is already active: PID $($live.ProcessId -join ', ')"
}
$arguments = @(
    $runner,
    "--queue", $queue,
    "--status", $status,
    "--log-root", $logRoot,
    "--cutoff", $Cutoff.ToString("yyyy-MM-ddTHH:mm:ss.ffffffK"),
    "--authorized-by-user",
    "--minimum-free-mib", "3800",
    "--maximum-temperature", "80",
    "--latest-start-minutes", "15"
)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot `
    -WindowStyle Hidden -RedirectStandardOutput $launcherStdout -RedirectStandardError $launcherStderr -PassThru
[pscustomobject]@{
    status = "started"
    pid = $process.Id
    cutoff = $Cutoff.ToString("o")
    runtime_status = $status
    stdout = $launcherStdout
    stderr = $launcherStderr
} | ConvertTo-Json
