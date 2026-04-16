param(
    [string]$TaskName = "Elibot Weekly Training",
    [string]$Day = "SUN",
    [string]$Time = "03:00",
    [string]$ProjectRoot = "C:\Users\KOURO\Desktop\chatbot"
)

$ErrorActionPreference = "Stop"

$runScript = Join-Path $ProjectRoot "scripts\run_weekly_training.ps1"
if (-not (Test-Path $runScript)) {
    throw "Run script not found at: $runScript"
}

$taskCommand = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $runScript

schtasks /Delete /TN "$TaskName" /F | Out-Null

schtasks /Create `
    /SC WEEKLY `
    /D $Day `
    /ST $Time `
    /TN "$TaskName" `
    /TR "$taskCommand" `
    /F | Out-Null

Write-Output "REGISTERED:$TaskName"
Write-Output "SCHEDULE: Weekly on $Day at $Time"
Write-Output "COMMAND: $taskCommand"
