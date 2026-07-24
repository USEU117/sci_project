$ErrorActionPreference = "Continue"

Write-Output "=== Date ==="
Get-Date -Format o

Write-Output "=== Python ==="
python --version
where.exe python

Write-Output "=== Git ==="
git --version
where.exe git

Write-Output "=== NVIDIA ==="
nvidia-smi

Write-Output "=== Filesystem ==="
Get-PSDrive -PSProvider FileSystem |
    Select-Object Name, Root,
        @{Name = "UsedGB"; Expression = { [math]::Round($_.Used / 1GB, 2) }},
        @{Name = "FreeGB"; Expression = { [math]::Round($_.Free / 1GB, 2) }}

