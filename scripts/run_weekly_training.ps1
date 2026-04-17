param(
    [string]$ProjectRoot = "C:\Users\KOURO\Desktop\chatbot",
    [string]$PythonPath = "C:\Users\KOURO\Desktop\chatbot\.venv_gpu\Scripts\python.exe",
    [string]$RunnerScript = "weekly_train_runner_fr.py",
    [ValidateSet("balanced", "strict")]
    [string]$VisionProfile = "strict",
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
$lockFile = Join-Path $logsDir "weekly_train.lock"

if (Test-Path $lockFile) {
    $hasActiveOwner = $false
    try {
        $lockRaw = Get-Content -Path $lockFile -Raw -ErrorAction Stop
        $lockMeta = $lockRaw | ConvertFrom-Json -ErrorAction Stop
        $lockPid = [int]$lockMeta.pid
        if ($lockPid -gt 0 -and (Get-Process -Id $lockPid -ErrorAction SilentlyContinue)) {
            $hasActiveOwner = $true
        }
    }
    catch {
        $hasActiveOwner = $false
    }

    if ($hasActiveOwner) {
        "[$(Get-Date -Format o)] SKIP: another weekly run appears active (lock owned by live process)." | Out-File -FilePath $logFile -Encoding utf8
        exit 0
    }

    Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
    "[$(Get-Date -Format o)] INFO: stale weekly lock recovered." | Out-File -FilePath $logFile -Encoding utf8
}

$lockPayload = @{
    pid = $PID
    created_at = (Get-Date -Format o)
} | ConvertTo-Json -Compress
$lockPayload | Set-Content -Path $lockFile -Encoding utf8

"[$(Get-Date -Format o)] START weekly training run" | Out-File -FilePath $logFile -Encoding utf8
"Python: $PythonPath" | Out-File -FilePath $logFile -Append -Encoding utf8
"Project: $ProjectRoot" | Out-File -FilePath $logFile -Append -Encoding utf8
"VisionProfile: $VisionProfile" | Out-File -FilePath $logFile -Append -Encoding utf8

if (-not (Test-Path (Join-Path $ProjectRoot $RunnerScript))) {
    "[$(Get-Date -Format o)] ERROR: runner script not found: $RunnerScript" | Out-File -FilePath $logFile -Append -Encoding utf8
    if (Test-Path $lockFile) { Remove-Item $lockFile -Force -ErrorAction SilentlyContinue }
    exit 1
}

$arguments = @(
    $RunnerScript,
    "--python-exec", $PythonPath,
    "--vision-profile", $VisionProfile,
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

try {
    & $PythonPath @arguments *>> $logFile
    $exitCode = $LASTEXITCODE
    "[$(Get-Date -Format o)] END weekly training run (exit=$exitCode)" | Out-File -FilePath $logFile -Append -Encoding utf8
    exit $exitCode
}
catch {
    "[$(Get-Date -Format o)] ERROR: $($_.Exception.Message)" | Out-File -FilePath $logFile -Append -Encoding utf8
    "[$(Get-Date -Format o)] END weekly training run (exit=1)" | Out-File -FilePath $logFile -Append -Encoding utf8
    exit 1
}
finally {
    if (Test-Path $lockFile) {
        Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
    }
}
