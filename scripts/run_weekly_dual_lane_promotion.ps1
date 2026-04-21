param(
    [string]$PythonExe = "c:/Users/KOURO/Desktop/chatbot/.venv/Scripts/python.exe",
    [int]$HeartbeatStaleSeconds = 300
)

$ErrorActionPreference = "Stop"
Set-Location "C:/Users/KOURO/Desktop/chatbot"

$LockPath = "reports/.weekly_dual_lane.lock"
$HeartbeatPath = "reports/training_heartbeat.json"
$InterruptedReportPath = "reports/training_interrupted.json"
$WeeklyStdoutPath = "reports/weekly_train_runner_stdout.log"
$WeeklyStderrPath = "reports/weekly_train_runner_stderr.log"
$FailedRunAnalysisPath = "reports/failed_run_analysis.json"

function Get-DescendantProcessIds {
    param([int]$RootPid)

    $children = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.ParentProcessId -eq $RootPid }
    $ids = @()
    foreach ($child in $children) {
        $ids += [int]$child.ProcessId
        $ids += Get-DescendantProcessIds -RootPid ([int]$child.ProcessId)
    }
    return $ids
}

function Stop-ProcessTree {
    param([int]$RootPid)

    $descendants = Get-DescendantProcessIds -RootPid $RootPid | Sort-Object -Unique
    foreach ($pid in ($descendants | Sort-Object -Descending)) {
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $RootPid -Force -ErrorAction SilentlyContinue
}

if (Test-Path $LockPath) {
    $staleLock = $false
    try {
        $lockInfo = Get-Content $LockPath -Raw | ConvertFrom-Json
        $lockPid = [int]$lockInfo.pid
        $lockCreatedAt = [datetime]$lockInfo.created_at
        $ageMinutes = ((Get-Date) - $lockCreatedAt).TotalMinutes
        $ownerAlive = Get-Process -Id $lockPid -ErrorAction SilentlyContinue
        if (-not $ownerAlive -or $ageMinutes -gt 720) {
            $staleLock = $true
        }
    } catch {
        $staleLock = $true
    }

    if ($staleLock) {
        Remove-Item $LockPath -Force
        Write-Warning "Removed stale lock file: $LockPath"
    } else {
        Write-Error "Another weekly dual-lane run appears active (lock file exists: $LockPath)."
        exit 2
    }
}

$lockPayload = @{
    pid = $PID
    created_at = (Get-Date).ToString("o")
    host = $env:COMPUTERNAME
} | ConvertTo-Json -Compress
$lockPayload | Set-Content -Path $LockPath -Encoding UTF8

$existingTrain = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -like '*train_chatbot_fr.py*'
}
if ($existingTrain) {
    Write-Error "Existing train_chatbot_fr.py process detected. Stop concurrent training before running dual-lane pipeline."
    exit 3
}

$existingWeeklyRunner = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -like '*weekly_train_runner_fr.py*'
}
if ($existingWeeklyRunner) {
    Write-Error "Existing weekly_train_runner_fr.py process detected. Stop concurrent weekly runner before running dual-lane pipeline."
    exit 4
}

try {

# 1) Weekly train/eval on strict profile with held-out eval enforced.
if (Test-Path $WeeklyStdoutPath) { Remove-Item $WeeklyStdoutPath -Force }
if (Test-Path $WeeklyStderrPath) { Remove-Item $WeeklyStderrPath -Force }

$runnerArgs = @(
    "weekly_train_runner_fr.py",
    "--vision-profile", "strict",
    "--eval-data-file", "data/eval/weekly_holdout_fr.csv",
    "--train-heartbeat-file", $HeartbeatPath,
    "--train-heartbeat-stale-seconds", "$HeartbeatStaleSeconds",
    "--train-interrupted-report", $InterruptedReportPath
)

$runnerProc = Start-Process -FilePath $PythonExe -ArgumentList $runnerArgs -NoNewWindow -PassThru -RedirectStandardOutput $WeeklyStdoutPath -RedirectStandardError $WeeklyStderrPath
$heartbeatKilled = $false
while (-not $runnerProc.HasExited) {
    Start-Sleep -Seconds 30
    if (Test-Path $HeartbeatPath) {
        $heartbeatAge = (Get-Date) - (Get-Item $HeartbeatPath).LastWriteTime
        if ($heartbeatAge.TotalSeconds -gt $HeartbeatStaleSeconds) {
            Write-Warning "Training heartbeat stale for $([int]$heartbeatAge.TotalSeconds)s. Stopping weekly runner process PID=$($runnerProc.Id)."
            Stop-ProcessTree -RootPid $runnerProc.Id
            $heartbeatKilled = $true
            break
        }
    }
}

if (-not $runnerProc.HasExited) {
    $runnerProc.WaitForExit()
}

[int]$weeklyRc = 1
if ($null -ne $runnerProc.ExitCode) {
    $weeklyRc = [int]$runnerProc.ExitCode
}
if ($heartbeatKilled) {
    $weeklyRc = 124
}

if ($weeklyRc -ne 0) {
    Stop-ProcessTree -RootPid $runnerProc.Id
    Write-Warning "weekly_train_runner_fr.py failed with rc=$weeklyRc. Running post-mortem analyzer."
    & $PythonExe scripts/analyze_failed_run.py --weekly-report reports/weekly_train_runner_report.json --runner-stdout $WeeklyStdoutPath --runner-stderr $WeeklyStderrPath --lock-file $LockPath --heartbeat-file $HeartbeatPath --heartbeat-stale-seconds $HeartbeatStaleSeconds --interrupted-report $InterruptedReportPath --out-json $FailedRunAnalysisPath
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Post-mortem analyzer failed; continuing with question-lane recovery attempt."
    }

    if (Test-Path "models/chatbot-fr-flan-t5-small-weekly") {
        Write-Warning "Attempting question-lane recovery on existing weekly model."
        & $PythonExe evaluate_model_fr.py --model-dir models/chatbot-fr-flan-t5-small-weekly --data-file data/eval/weekly_holdout_fr.csv --samples 150 --seed 42 --max-new-tokens 96 --out-json reports/eval_weekly_laneb_recovery.json --out-csv reports/eval_weekly_laneb_recovery_samples.csv
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        & $PythonExe evaluate_business_benchmark_fr.py --model-dir models/chatbot-fr-flan-t5-small-weekly --cases-file data/eval/business_benchmark_fr.json --max-cases 0 --max-new-tokens 128 --repetition-penalty 1.1 --no-repeat-ngram 3 --out-json reports/eval_business_laneb_recovery.json --out-csv reports/eval_business_laneb_recovery_samples.csv
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        & $PythonExe evaluate_conversation_fr.py --model-dir models/chatbot-fr-flan-t5-small-weekly --history-mode full --history-turns 4 --use-slot-memory --max-new-tokens 96 --out-json reports/eval_conversation_model_only_laneb_recovery.json --out-csv reports/eval_conversation_model_only_laneb_recovery_details.csv
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        & $PythonExe scripts/evaluate_question_recognition_targeted.py --model-dir models/chatbot-fr-flan-t5-small-weekly --prompts-file data/eval/question_recognition_unseen_fr.txt --train-dataset data/processed/chatbot_train_fr_question_recognition.csv --out-json reports/eval_question_recognition_unseen_laneb_recovery.json --out-csv reports/eval_question_recognition_unseen_laneb_recovery.csv
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        & $PythonExe scripts/log_best_checkpoint.py --model-dir models/chatbot-fr-flan-t5-small-weekly --eval-json reports/eval_weekly_laneb_recovery.json --business-json reports/eval_business_laneb_recovery.json --conversation-json reports/eval_conversation_model_only_laneb_recovery.json --question-json reports/eval_question_recognition_unseen_laneb_recovery.json --latest-best-json reports/best_checkpoint_question_lane_latest.json --history-jsonl reports/best_checkpoint_question_lane_history.jsonl --snapshots-dir reports/best_snapshots_question_lane --tag weekly_question_lane_recovery --lane question --primary-metric question_recognition_rate --min-primary-improvement 0.01 --min-question-recognition-rate 0.80 --max-question-regression 0.05 --allow-memory-regression
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

        Write-Warning "Global lane skipped due to failed weekly runner; question lane recovery completed."
        exit 0
    }

    exit $weeklyRc
}

# 2) Conversation eval in model-only mode (no rule memory shortcut).
& $PythonExe evaluate_conversation_fr.py --model-dir models/chatbot-fr-flan-t5-small-weekly --history-mode full --history-turns 4 --use-slot-memory --max-new-tokens 96 --out-json reports/eval_conversation_model_only_weekly.json --out-csv reports/eval_conversation_model_only_weekly_details.csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 3) Unseen question-recognition benchmark + overlap diagnostics.
& $PythonExe scripts/evaluate_question_recognition_targeted.py --model-dir models/chatbot-fr-flan-t5-small-weekly --prompts-file data/eval/question_recognition_unseen_fr.txt --train-dataset data/processed/chatbot_train_fr_question_recognition.csv --out-json reports/eval_question_recognition_unseen_weekly.json --out-csv reports/eval_question_recognition_unseen_weekly.csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 4a) Global lane promotion decision.
& $PythonExe scripts/log_best_checkpoint.py --model-dir models/chatbot-fr-flan-t5-small-weekly --eval-json reports/eval_weekly_latest.json --business-json reports/eval_business_weekly.json --conversation-json reports/eval_conversation_model_only_weekly.json --question-json reports/eval_question_recognition_unseen_weekly.json --latest-best-json reports/best_checkpoint_global_latest.json --history-jsonl reports/best_checkpoint_global_history.jsonl --snapshots-dir reports/best_snapshots_global --tag weekly_global --lane global --primary-metric composite --min-composite-improvement 0.002 --max-score-regression 0.005 --max-pass-rate-regression 0.02 --max-question-regression 0.05
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# 4b) Question-challenger lane promotion decision.
& $PythonExe scripts/log_best_checkpoint.py --model-dir models/chatbot-fr-flan-t5-small-weekly --eval-json reports/eval_weekly_latest.json --business-json reports/eval_business_weekly.json --conversation-json reports/eval_conversation_model_only_weekly.json --question-json reports/eval_question_recognition_unseen_weekly.json --latest-best-json reports/best_checkpoint_question_lane_latest.json --history-jsonl reports/best_checkpoint_question_lane_history.jsonl --snapshots-dir reports/best_snapshots_question_lane --tag weekly_question_lane --lane question --primary-metric question_recognition_rate --min-primary-improvement 0.01 --min-question-recognition-rate 0.80 --max-question-regression 0.05 --allow-memory-regression
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "DONE: weekly dual-lane promotion complete"

} finally {
    if (Test-Path $LockPath) {
        Remove-Item $LockPath -Force
    }
}
