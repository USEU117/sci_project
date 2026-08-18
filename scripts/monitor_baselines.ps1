param(
    [int]$PollSeconds = 20,
    [int]$MaxHours = 8
)

# Monitor / relay for the two remaining baselines after PromptAD completed:
#   Phase A: download torch 2.7.1+cu118 + torchvision (aliyun mirror, no GPU),
#            pip install into .venv-adaptclip, verify torch.cuda.is_available().
#   Phase B: ReMP-AD MVTec test k_shot 4 -> 2 -> 1 (GPU, serial, skip training).
#   Phase C: once A and B are both done, run AdaptCLIP MVTec Gate A (GPU).
# Progress is persisted via marker files so the script can be restarted safely.

$ErrorActionPreference = 'Continue'
$ProjectRoot = Split-Path -Parent $PSScriptRoot

$LogDir = Join-Path $ProjectRoot 'outputs\logs\baseline_monitor'
$null = New-Item -ItemType Directory -Force -Path $LogDir
$LogFile = Join-Path $LogDir 'monitor.log'
$AllDoneMarker = Join-Path $LogDir 'ALL_DONE.marker'

$DownloadDir = Join-Path $ProjectRoot 'outputs\downloads'
$null = New-Item -ItemType Directory -Force -Path $DownloadDir
$TorchWhl = Join-Path $DownloadDir 'torch-2.7.1+cu118-cp310-cp310-win_amd64.whl'
$TorchvisionWhl = Join-Path $DownloadDir 'torchvision-0.22.1+cu118-cp310-cp310-win_amd64.whl'
$TorchSize = 2817209444   # aliyun Content-Length for the torch cu118 wheel

$TorchUrl = 'https://mirrors.aliyun.com/pytorch-wheels/cu118/torch-2.7.1%2Bcu118-cp310-cp310-win_amd64.whl'
$TorchvisionUrl = 'https://mirrors.aliyun.com/pytorch-wheels/cu118/torchvision-0.22.1%2Bcu118-cp310-cp310-win_amd64.whl'

$AdaptPython = Join-Path $ProjectRoot '.venv-adaptclip\Scripts\python.exe'
$RempPython = Join-Path $ProjectRoot '.venv-remp_ad\Scripts\python.exe'
$RempRoot = Join-Path $ProjectRoot 'methods\remp_ad'
$RempConfig = Join-Path $RempRoot 'config\mvtec.yaml'
$TestData = Join-Path $ProjectRoot 'data\mvtec'
$GateAScript = Join-Path $PSScriptRoot 'start_adaptclip_mvtec_gate_a.ps1'

$rempDone = @{
    4 = Join-Path $LogDir 'remp_k4.done'
    2 = Join-Path $LogDir 'remp_k2.done'
    1 = Join-Path $LogDir 'remp_k1.done'
}
$TorchInstalled = Join-Path $LogDir 'torch_installed.done'
$GateADone = Join-Path $LogDir 'adaptclip_gatea.done'

function Write-Log($Msg) {
    $Line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Msg"
    Write-Host $Line
    Add-Content -LiteralPath $LogFile -Value $Line
}

function Get-Procs($Pattern) {
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^(python|curl)' -and $_.CommandLine -match $Pattern
    }
}

function Test-RempDone($k) {
    $log = Join-Path $RempRoot "result\mvtec\mvtec_${k}shot\log_seed_10_${k}shot.txt"
    if (Test-Path -LiteralPath $log) {
        return [bool](Select-String -Path $log -Pattern 'mean' -Quiet)
    }
    return $false
}

Write-Log "monitor started. poll=${PollSeconds}s max=${MaxHours}h"
$Deadline = (Get-Date).AddHours($MaxHours)

while ((Get-Date) -lt $Deadline) {
    # ---- Phase A: torch / torchvision download + install + CUDA verify ----
    if (-not (Test-Path -LiteralPath $TorchInstalled)) {
        $torchOk = (Test-Path -LiteralPath $TorchWhl) -and ((Get-Item -LiteralPath $TorchWhl).Length -ge $TorchSize)
        $tvOk = (Test-Path -LiteralPath $TorchvisionWhl) -and ((Get-Item -LiteralPath $TorchvisionWhl).Length -gt 1MB)

        if (-not $torchOk) {
            if (-not (Get-Procs 'torch-2\.7\.1')) {
                Write-Log "torch wheel incomplete ($(if(Test-Path $TorchWhl){[math]::Round((Get-Item $TorchWhl).Length/1MB)}else{0}) MB), starting download (aliyun)..."
                Start-Process curl.exe -ArgumentList @('-L','--retry','8','--retry-delay','3','-C','-','-o', $TorchWhl, $TorchUrl) -WindowStyle Hidden
            }
        } elseif (-not $tvOk) {
            if (-not (Get-Procs 'torchvision-0\.22\.1')) {
                Write-Log "torchvision wheel missing, starting download..."
                Start-Process curl.exe -ArgumentList @('-L','--retry','8','--retry-delay','3','-C','-','-o', $TorchvisionWhl, $TorchvisionUrl) -WindowStyle Hidden
            }
        } else {
            Write-Log "installing torch+torchvision into .venv-adaptclip (local wheels, force-reinstall)..."
            & $AdaptPython -m pip install $TorchWhl $TorchvisionWhl --no-deps --force-reinstall --no-cache-dir
            if ($LASTEXITCODE -eq 0) {
                & $AdaptPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('CUDA OK', torch.__version__)"
                if ($LASTEXITCODE -eq 0) {
                    Write-Log "torch installed and CUDA verified."
                    New-Item -ItemType File -Force -Path $TorchInstalled | Out-Null
                } else {
                    Write-Log "CUDA verification FAILED, will retry install."
                }
            } else {
                Write-Log "pip install FAILED (exit $LASTEXITCODE), will retry."
            }
        }
    }

    # ---- Phase B: ReMP-AD 4/2/1 serial relay (skip training) ----
    foreach ($k in 4, 2, 1) {
        if (-not (Test-Path -LiteralPath $rempDone[$k])) {
            if (Test-RempDone $k) {
                Write-Log "ReMP-AD k_shot=$k already finished (result file has 'mean')."
                New-Item -ItemType File -Force -Path $rempDone[$k] | Out-Null
            } elseif (-not (Get-Procs "--k_shot\s+$k\b")) {
                # Serial dependency: k_shot=2 waits for k_shot=4, k_shot=1 waits for k_shot=2.
                if (($k -eq 2 -and -not (Test-Path -LiteralPath $rempDone[4])) -or
                    ($k -eq 1 -and -not (Test-Path -LiteralPath $rempDone[2]))) {
                    continue
                }
                Write-Log "starting ReMP-AD test k_shot=$k..."
                Start-Process $RempPython `
                    -ArgumentList @('test.py','--config_path',$RempConfig,'--test_data_path',$TestData,'--k_shot',"$k") `
                    -WorkingDirectory $RempRoot -WindowStyle Hidden `
                    -RedirectStandardOutput (Join-Path $LogDir "remp_k${k}.out.log") `
                    -RedirectStandardError (Join-Path $LogDir "remp_k${k}.err.log")
            }
        }
    }

    # ---- Phase C: AdaptCLIP Gate A (needs A + B done, GPU free) ----
    if ((Test-Path -LiteralPath $TorchInstalled) -and (Test-Path -LiteralPath $rempDone[4]) -and (Test-Path -LiteralPath $rempDone[2]) -and (Test-Path -LiteralPath $rempDone[1]) -and -not (Test-Path -LiteralPath $GateADone)) {
        Write-Log "all prerequisites ready, running AdaptCLIP MVTec Gate A..."
        try {
            & $GateAScript
            if ($LASTEXITCODE -eq 0) {
                Write-Log "AdaptCLIP MVTec Gate A completed."
                New-Item -ItemType File -Force -Path $GateADone | Out-Null
            } else {
                Write-Log "AdaptCLIP Gate A exited non-zero ($LASTEXITCODE), will retry next cycle."
            }
        } catch {
            Write-Log "AdaptCLIP Gate A FAILED: $_"
        }
    }

    # ---- Completion ----
    if ((Test-Path -LiteralPath $TorchInstalled) -and (Test-Path -LiteralPath $rempDone[1]) -and (Test-Path -LiteralPath $GateADone)) {
        Write-Log "ALL BASELINES COMPLETE."
        New-Item -ItemType File -Force -Path $AllDoneMarker | Out-Null
        exit 0
    }

    Start-Sleep -Seconds $PollSeconds
}

Write-Log "monitor timed out (${MaxHours}h)."
exit 2
