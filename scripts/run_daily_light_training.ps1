param(
    [string]$ProjectRoot = "C:\Users\KOURO\Desktop\chatbot",
    [string]$PythonPath = "C:\Users\KOURO\Desktop\chatbot\.venv_gpu\Scripts\python.exe",
    [string]$RunnerScript = "weekly_train_runner_fr.py",
    [switch]$DryRun,
    [switch]$ForceFullRun,
    [int]$MinFreeGpuMb = 1500,
    [int]$MaxCpuPercent = 90
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
$lockFile = Join-Path $logsDir "daily_light_train.lock"

if (Test-Path $lockFile) {
    "[$(Get-Date -Format o)] SKIP: another daily light run appears active (lock file present)." | Out-File -FilePath $logFile -Encoding utf8
    exit 0
}

New-Item -Path $lockFile -ItemType File -Force | Out-Null

"[$(Get-Date -Format o)] START daily light training run" | Out-File -FilePath $logFile -Encoding utf8
"Python: $PythonPath" | Out-File -FilePath $logFile -Append -Encoding utf8
"Project: $ProjectRoot" | Out-File -FilePath $logFile -Append -Encoding utf8

$effectiveDryRun = $DryRun.IsPresent
$safetyReasons = New-Object System.Collections.Generic.List[string]

if (-not $ForceFullRun) {
    $cpu = $null
    try {
        $cpuSample = Get-CimInstance -ClassName Win32_Processor -ErrorAction Stop |
            Measure-Object -Property LoadPercentage -Average
        $cpu = [double]$cpuSample.Average
    }
    catch {
        $safetyReasons.Add("cpu_check_unavailable") | Out-Null
    }

    if (($cpu -ne $null) -and ($cpu -gt $MaxCpuPercent)) {
        $effectiveDryRun = $true
        $safetyReasons.Add(("cpu_high:{0:N1}%" -f $cpu)) | Out-Null
    }

    $nvidiaSmi = Get-Command "nvidia-smi" -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        $effectiveDryRun = $true
        $safetyReasons.Add("gpu_unavailable:nvidia-smi-not-found") | Out-Null
    }
    else {
        $gpuRaw = & $nvidiaSmi.Source --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
        $gpuValues = @($gpuRaw | ForEach-Object { ($_ -as [int]) } | Where-Object { $_ -ne $null })
        if ($gpuValues.Count -eq 0) {
            $effectiveDryRun = $true
            $safetyReasons.Add("gpu_unavailable:no-memory-data") | Out-Null
        }
        else {
            $maxFree = ($gpuValues | Measure-Object -Maximum).Maximum
            if ($maxFree -lt $MinFreeGpuMb) {
                $effectiveDryRun = $true
                $safetyReasons.Add(("gpu_low_free_mb:{0}" -f $maxFree)) | Out-Null
            }
        }
    }
}

$safetyLine = if ($safetyReasons.Count -gt 0) { $safetyReasons -join "," } else { "none" }
"SafetyDryRun: $effectiveDryRun" | Out-File -FilePath $logFile -Append -Encoding utf8
"SafetyReasons: $safetyLine" | Out-File -FilePath $logFile -Append -Encoding utf8

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

if ($effectiveDryRun) {
    $arguments += "--dry-run"
}

try {
    & $PythonPath @arguments *>> $logFile
    $exitCode = $LASTEXITCODE
    "[$(Get-Date -Format o)] END daily light training run (exit=$exitCode)" | Out-File -FilePath $logFile -Append -Encoding utf8
    exit $exitCode
}
finally {
    if (Test-Path $lockFile) {
        Remove-Item $lockFile -Force -ErrorAction SilentlyContinue
    }
}
