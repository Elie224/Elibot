param(
    [string]$PythonExe = "c:/Users/KOURO/Desktop/chatbot/.venv_gpu/Scripts/python.exe"
)

$ErrorActionPreference = "Stop"
Set-Location "C:/Users/KOURO/Desktop/chatbot"

& $PythonExe weekly_train_runner_fr.py --vision-profile strict
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe evaluate_conversation_fr.py --model-dir models/chatbot-fr-flan-t5-small-weekly --history-mode full --history-turns 4 --use-slot-memory --max-new-tokens 96 --out-json reports/eval_conversation_weekly.json --out-csv reports/eval_conversation_weekly_details.csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe -c "import json, pathlib
base=json.loads(pathlib.Path('reports/v3_locked_baseline_snapshot.json').read_text(encoding='utf-8'))
cur_w=json.loads(pathlib.Path('reports/eval_weekly_latest.json').read_text(encoding='utf-8'))
cur_b=json.loads(pathlib.Path('reports/eval_business_weekly.json').read_text(encoding='utf-8'))
cur_c=json.loads(pathlib.Path('reports/eval_conversation_weekly.json').read_text(encoding='utf-8'))
feed=json.loads(pathlib.Path('reports/feeding_pipeline_report.json').read_text(encoding='utf-8'))
now={
 'token_f1_mean': cur_w.get('token_f1_mean'),
 'exact_match_rate': cur_w.get('exact_match_rate'),
 'score_mean': cur_b.get('score_mean'),
 'pass_rate': cur_b.get('pass_rate'),
 'keyword_coverage_mean': cur_b.get('keyword_coverage_mean'),
 'memory_recall_rate': cur_c.get('memory_recall_rate')
}
out={'baseline':base.get('metrics',{}),'current':now,'delta':{},'bundle_counts':feed.get('counts',{}),'bundle_policy':feed.get('policy',{})}
for k,v in now.items():
 b=out['baseline'].get(k)
 out['delta'][k]=None if (b is None or v is None) else round(v-b,6)
path=pathlib.Path('reports/v4_comparison_against_locked_baseline.json')
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'saved':str(path),'current':now,'delta':out['delta']},ensure_ascii=False))"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "DONE: reports/v4_comparison_against_locked_baseline.json"
