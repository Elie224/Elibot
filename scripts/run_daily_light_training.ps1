param(
    [string]$ProjectRoot = "C:\Users\KOURO\Desktop\chatbot",
    [string]$PythonPath = "C:\Users\KOURO\Desktop\chatbot\.venv_gpu\Scripts\python.exe",
    [string]$RunnerScript = "weekly_train_runner_fr.py",
    [switch]$DryRun
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
$logFile = Join-Path $logsDir ("daily_light_train_" + $timestamp + ".log")

"[$(Get-Date -Format o)] START daily light training run" | Out-File -FilePath $logFile -Encoding utf8
"Python: $PythonPath" | Out-File -FilePath $logFile -Append -Encoding utf8
"Project: $ProjectRoot" | Out-File -FilePath $logFile -Append -Encoding utf8

$arguments = @(
    $RunnerScript,
    "--python-exec", $PythonPath,
    "--run-report", "reports/daily_light_train_runner_report.json",
    "--feeding-report", "reports/feeding_pipeline_report_daily.json",
    "--eval-out-json", "reports/eval_daily_light_latest.json",
    "--eval-out-csv", "reports/eval_daily_light_latest_samples.csv",
    "--output-model", "models/chatbot-fr-flan-t5-small-daily-light",
    "--train-max-samples", "3000",
    "--train-max-eval-samples", "400",
    "--train-epochs", "0.4",
    "--train-batch-size", "8",
    "--train-grad-accum", "1",
    "--train-learning-rate", "1e-4",
    "--train-log-steps", "25",
    "--eval-samples", "60",
    "--max-feedback-rows", "8000",
    "--max-signature-rows", "8000"
)

if ($DryRun) {
    $arguments += "--dry-run"
}

& $PythonPath @arguments *>> $logFile
$exitCode = $LASTEXITCODE

"[$(Get-Date -Format o)] END daily light training run (exit=$exitCode)" | Out-File -FilePath $logFile -Append -Encoding utf8
exit $exitCode
