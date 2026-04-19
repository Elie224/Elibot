param(
    [string]$TaskName = "Elibot Weekly Live Benchmark Auto-Remediation",
    [string]$Day = "MON",
    [string]$Time = "04:30",
    [string]$ProjectRoot = "C:\Users\KOURO\Desktop\chatbot",
    [double]$FailIfBelow = 0.95,
    [switch]$RunAsSystem
)

$ErrorActionPreference = "Stop"

$runScript = Join-Path $ProjectRoot "scripts\run_live_benchmark_auto_remediation.ps1"
if (-not (Test-Path $runScript)) {
    throw "Run script not found at: $runScript"
}

$thresholdArg = $FailIfBelow.ToString([System.Globalization.CultureInfo]::InvariantCulture)
$taskCommand = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{0}" -FailIfBelow {1}' -f $runScript, $thresholdArg

cmd.exe /c ('schtasks /Delete /TN "{0}" /F >nul 2>&1' -f $TaskName) | Out-Null

if ($RunAsSystem) {
    schtasks /Create `
        /SC WEEKLY `
        /D $Day `
        /ST $Time `
        /TN "$TaskName" `
        /TR "$taskCommand" `
        /RU "SYSTEM" `
        /RL HIGHEST `
        /F | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to register task: $TaskName" }
}
else {
    schtasks /Create `
        /SC WEEKLY `
        /D $Day `
        /ST $Time `
        /TN "$TaskName" `
        /TR "$taskCommand" `
        /F | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Failed to register task: $TaskName" }
}

Write-Output "REGISTERED:$TaskName"
Write-Output "SCHEDULE: Weekly on $Day at $Time"
Write-Output "RUN_AS_SYSTEM: $($RunAsSystem.IsPresent)"
Write-Output "FAIL_IF_BELOW: $FailIfBelow"
Write-Output "COMMAND: $taskCommand"
