param(
    [string]$ProjectRoot = "C:\Users\KOURO\Desktop\chatbot",
    [string]$PythonExe = "C:\Users\KOURO\Desktop\chatbot\.venv_gpu\Scripts\python.exe",
    [string]$OutJson = "reports/eval_business_live_latest.json",
    [ValidateSet("Court", "Expert")]
    [string]$Mode = "Expert",
    [double]$FailIfBelow = 0.95,
    [string]$AlertFile = "reports/eval_business_live_alert.txt"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$reportPath = Join-Path $ProjectRoot $OutJson
$today = (Get-Date).Date

if (Test-Path $reportPath) {
    $last = (Get-Item $reportPath).LastWriteTime.Date
    if ($last -eq $today) {
        Write-Output "SKIP: report already generated today -> $OutJson"
        exit 0
    }
}

powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $ProjectRoot "scripts/run_live_benchmark_and_summary.ps1") `
    -PythonExe $PythonExe `
    -OutJson $OutJson `
    -Mode $Mode `
    -FailIfBelow $FailIfBelow `
    -AlertFile $AlertFile
