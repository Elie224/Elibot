param(
    [string]$ProjectRoot = "C:\Users\KOURO\Desktop\chatbot",
    [string]$MainTaskName = "Elibot Weekly Live Benchmark",
    [string]$PythonExe = "C:\Users\KOURO\Desktop\chatbot\.venv_gpu\Scripts\python.exe",
    [ValidateSet("Court", "Expert")]
    [string]$Mode = "Expert",
    [double]$FailIfBelow = 0.95,
    [string]$OutJson = "reports/eval_business_live_latest.json",
    [string]$AlertFile = "reports/eval_business_live_alert.txt"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$shouldRemediate = $false
$reason = ""

try {
    $info = Get-ScheduledTaskInfo -TaskName $MainTaskName -ErrorAction Stop
    if ($info.LastTaskResult -ne 0) {
        $shouldRemediate = $true
        $reason = "last_result=$($info.LastTaskResult)"
    }
    elseif ($info.LastRunTime -lt (Get-Date).AddDays(-8)) {
        $shouldRemediate = $true
        $reason = "stale_last_run=$($info.LastRunTime.ToString('s'))"
    }
}
catch {
    $shouldRemediate = $true
    $reason = "task_info_unavailable"
}

if (-not $shouldRemediate) {
    Write-Output "SKIP: main task healthy"
    exit 0
}

Write-Output "REMEDIATE: $reason"

powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot "scripts/run_live_benchmark_and_summary.ps1") `
    -PythonExe $PythonExe `
    -OutJson $OutJson `
    -Mode $Mode `
    -FailIfBelow $FailIfBelow `
    -AlertFile $AlertFile
