param(
    [string]$ProjectRoot = "C:\Users\KOURO\Desktop\chatbot",
    [string]$PythonPath = "C:\Users\KOURO\Desktop\chatbot\.venv_gpu\Scripts\python.exe",
    [string]$RunnerScript = "weekly_train_runner_fr.py",
    [switch]$DryRun,
    [string[]]$ExtraRunnerArgs = @()
)

$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

if (-not (Test-Path $PythonPath)) {
    throw "Python not found at: $PythonPath"
}

$logsDir = Join-Path $ProjectRoot "reports\scheduled"
if (-not (Test-Path $logsDir)) {
    New-Item -Path $logsDir -ItemType Directory -Force | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $logsDir ("weekly_train_" + $timestamp + ".log")

"[$(Get-Date -Format o)] START weekly training run" | Out-File -FilePath $logFile -Encoding utf8
"Python: $PythonPath" | Out-File -FilePath $logFile -Append -Encoding utf8
"Project: $ProjectRoot" | Out-File -FilePath $logFile -Append -Encoding utf8

$arguments = @(
    $RunnerScript,
    "--python-exec", $PythonPath,
    "--run-report", "reports/weekly_train_runner_report.json",
    "--feeding-report", "reports/feeding_pipeline_report.json",
    "--eval-out-json", "reports/eval_weekly_latest.json",
    "--eval-out-csv", "reports/eval_weekly_latest_samples.csv"
)

if ($DryRun) {
    $arguments += "--dry-run"
}

if ($ExtraRunnerArgs.Count -gt 0) {
    $arguments += $ExtraRunnerArgs
}

& $PythonPath @arguments *>> $logFile
$exitCode = $LASTEXITCODE

"[$(Get-Date -Format o)] END weekly training run (exit=$exitCode)" | Out-File -FilePath $logFile -Append -Encoding utf8
exit $exitCode
