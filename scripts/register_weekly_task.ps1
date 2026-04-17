param(
    [string]$TaskName = "Elibot Weekly Training",
    [string]$Day = "SUN",
    [string]$Time = "03:00",
    [string]$ProjectRoot = "C:\Users\KOURO\Desktop\chatbot",
    [ValidateSet("balanced", "strict")]
    [string]$VisionProfile = "strict",
    [switch]$RunAsSystem
)

$ErrorActionPreference = "Stop"

$runScript = Join-Path $ProjectRoot "scripts\run_weekly_training.ps1"
if (-not (Test-Path $runScript)) {
    throw "Run script not found at: $runScript"
}

$taskCommand = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{0}" -VisionProfile {1}' -f $runScript, $VisionProfile

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
Write-Output "COMMAND: $taskCommand"
Write-Output "VISION_PROFILE: $VisionProfile"
Write-Output "RUN_AS_SYSTEM: $($RunAsSystem.IsPresent)"
