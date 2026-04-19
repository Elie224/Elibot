param(
    [string]$TaskName = "Elibot Weekly Live Benchmark",
    [string]$Day = "SUN",
    [string]$Time = "04:00",
    [string]$ProjectRoot = "C:\Users\KOURO\Desktop\chatbot",
    [string]$PythonExe = "C:\Users\KOURO\Desktop\chatbot\.venv_gpu\Scripts\python.exe",
    [ValidateSet("Court", "Expert")]
    [string]$Mode = "Expert",
    [switch]$RunAsSystem
)

$ErrorActionPreference = "Stop"

$runScript = Join-Path $ProjectRoot "scripts\run_live_benchmark_and_summary.ps1"
if (-not (Test-Path $runScript)) {
    throw "Run script not found at: $runScript"
}

$taskCommand = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{0}" -PythonExe "{1}" -Mode {2}' -f $runScript, $PythonExe, $Mode

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
}
else {
    schtasks /Create `
        /SC WEEKLY `
        /D $Day `
        /ST $Time `
        /TN "$TaskName" `
        /TR "$taskCommand" `
        /F | Out-Null
}

Write-Output "REGISTERED:$TaskName"
Write-Output "SCHEDULE: Weekly on $Day at $Time"
Write-Output "RUN_AS_SYSTEM: $($RunAsSystem.IsPresent)"
Write-Output "MODE: $Mode"
Write-Output "COMMAND: $taskCommand"
