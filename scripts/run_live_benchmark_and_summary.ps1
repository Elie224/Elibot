param(
    [string]$PythonExe = ".venv_gpu/Scripts/python.exe",
    [string]$SpaceUrl = "https://elie224-elibot-chat.hf.space",
    [string]$Benchmark = "data/eval/business_benchmark_fr.json",
    [string]$OutJson = "reports/eval_business_live_latest.json",
    [string]$OutCsv = "reports/eval_business_live_latest.csv",
    [string]$OutMd = "reports/eval_business_live_latest.md",
    [ValidateSet("Court", "Expert")]
    [string]$Mode = "Expert",
    [double]$FailIfBelow = -1,
    [string]$AlertFile = "reports/eval_business_live_alert.txt"
)

$ErrorActionPreference = "Stop"

& $PythonExe scripts/run_live_business_benchmark.py `
    --space-url $SpaceUrl `
    --benchmark $Benchmark `
    --out-json $OutJson `
    --out-csv $OutCsv `
    --mode $Mode

& $PythonExe scripts/build_live_benchmark_summary.py `
    --in-json $OutJson `
    --out-md $OutMd

if ($FailIfBelow -ge 0) {
    & $PythonExe scripts/assert_live_benchmark_threshold.py `
        --in-json $OutJson `
        --threshold $FailIfBelow `
        --alert-file $AlertFile
}
