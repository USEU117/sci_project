$ErrorActionPreference = 'Stop'
$taskDeadline = [DateTimeOffset]::Parse('2026-09-08T07:00:00+08:00')
$taskRoot = 'D:\STUDY\My_github\sci_project'
$taskState = Join-Path $taskRoot 'experiments\dynamic_fusion\innovation_overnight_20260908\awake_state.json'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class OvernightPowerRequest {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
try {
    # Keep the system awake during authorized work; allow the display to turn off.
    $taskPowerResult = [OvernightPowerRequest]::SetThreadExecutionState([uint32]2147483649)
    if ($taskPowerResult -eq 0) { throw 'SetThreadExecutionState failed' }
    @{ pid=$PID; deadline=$taskDeadline.ToString('o'); status='active'; started=[DateTimeOffset]::Now.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $taskState -Encoding utf8
    while ([DateTimeOffset]::Now -lt $taskDeadline) { Start-Sleep -Seconds 30 }
} finally {
    [void][OvernightPowerRequest]::SetThreadExecutionState([uint32]2147483648)
    @{ pid=$PID; deadline=$taskDeadline.ToString('o'); status='released'; stopped=[DateTimeOffset]::Now.ToString('o') } |
        ConvertTo-Json | Set-Content -LiteralPath $taskState -Encoding utf8
}
