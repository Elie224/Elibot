param(
    [string]$TaskName = "Elibot Daily Light Training",
    [string]$Time = "01:30",
    [string]$ProjectRoot = "C:\Users\KOURO\Desktop\chatbot"
)

$ErrorActionPreference = "Stop"

function Test-IsAdmin {
    $current = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($current)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdmin)) {
    $scriptPath = $MyInvocation.MyCommand.Path
    $argList = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", ('"{0}"' -f $scriptPath),
        "-TaskName", ('"{0}"' -f $TaskName),
        "-Time", $Time,
        "-ProjectRoot", ('"{0}"' -f $ProjectRoot)
    )

    Start-Process -FilePath "powershell.exe" -ArgumentList $argList -Verb RunAs
    Write-Output "UAC_PROMPTED: Relaunched as administrator."
    exit 0
}

$runScript = Join-Path $ProjectRoot "scripts\run_daily_light_training.ps1"
if (-not (Test-Path $runScript)) {
    throw "Run script not found at: $runScript"
}

$taskCommand = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $runScript

cmd.exe /c "schtasks /Delete /TN \"$TaskName\" /F >nul 2>&1" | Out-Null

schtasks /Create `
    /SC DAILY `
    /ST $Time `
    /TN "$TaskName" `
    /TR "$taskCommand" `
    /RU "SYSTEM" `
    /RL HIGHEST `
    /F | Out-Null

Write-Output "REGISTERED:$TaskName"
Write-Output "SCHEDULE: Daily at $Time"
Write-Output "RUN_AS: SYSTEM"
Write-Output "COMMAND: $taskCommand"
