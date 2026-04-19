param(
    [string]$PythonExe = ".venv_gpu/Scripts/python.exe",
    [string]$SpaceUrl = "https://elie224-elibot-chat.hf.space",
    [string]$Benchmark = "data/eval/business_benchmark_fr.json",
    [string]$OutJson = "reports/eval_business_live_latest.json",
    [string]$OutCsv = "reports/eval_business_live_latest.csv",
    [ValidateSet("Court", "Expert")]
    [string]$Mode = "Expert"
)

$ErrorActionPreference = "Stop"

& $PythonExe scripts/run_live_business_benchmark.py `
    --space-url $SpaceUrl `
    --benchmark $Benchmark `
    --out-json $OutJson `
    --out-csv $OutCsv `
    --mode $Mode
