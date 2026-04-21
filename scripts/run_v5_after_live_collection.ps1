param(
    [string]$PythonExe = "c:/Users/KOURO/Desktop/chatbot/.venv_gpu/Scripts/python.exe"
)

$ErrorActionPreference = "Stop"
Set-Location "C:/Users/KOURO/Desktop/chatbot"

# Baseline ratio is locked in weekly_train_runner strict profile:
# 0.40 / 0.12 / 0.28 / 0.10 / 0.10
& $PythonExe weekly_train_runner_fr.py --vision-profile strict
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe evaluate_conversation_fr.py --model-dir models/chatbot-fr-flan-t5-small-weekly --history-mode full --history-turns 4 --use-slot-memory --max-new-tokens 96 --out-json reports/eval_conversation_weekly.json --out-csv reports/eval_conversation_weekly_details.csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe -c "import json, pathlib
v4=json.loads(pathlib.Path('reports/v4_locked_snapshot.json').read_text(encoding='utf-8'))
v3=json.loads(pathlib.Path('reports/v3_locked_baseline_snapshot.json').read_text(encoding='utf-8'))
cur_w=json.loads(pathlib.Path('reports/eval_weekly_latest.json').read_text(encoding='utf-8'))
cur_b=json.loads(pathlib.Path('reports/eval_business_weekly.json').read_text(encoding='utf-8'))
cur_c=json.loads(pathlib.Path('reports/eval_conversation_weekly.json').read_text(encoding='utf-8'))
feed=json.loads(pathlib.Path('reports/feeding_pipeline_report.json').read_text(encoding='utf-8'))
cur={
 'token_f1_mean': cur_w.get('token_f1_mean'),
 'exact_match_rate': cur_w.get('exact_match_rate'),
 'score_mean': cur_b.get('score_mean'),
 'pass_rate': cur_b.get('pass_rate'),
 'keyword_coverage_mean': cur_b.get('keyword_coverage_mean'),
 'memory_recall_rate': cur_c.get('memory_recall_rate')
}
out={
 'current': cur,
 'v4_baseline': v4.get('metrics',{}),
 'v3_baseline': v3.get('metrics',{}),
 'delta_vs_v4': {},
 'delta_vs_v3': {},
 'bundle_counts': feed.get('counts',{}),
 'bundle_policy': feed.get('policy',{})
}
for k,v in cur.items():
 b4=out['v4_baseline'].get(k)
 b3=out['v3_baseline'].get(k)
 out['delta_vs_v4'][k]=None if (b4 is None or v is None) else round(v-b4,6)
 out['delta_vs_v3'][k]=None if (b3 is None or v is None) else round(v-b3,6)
path=pathlib.Path('reports/v5_comparison_against_v4_and_v3.json')
path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'saved':str(path),'current':cur,'delta_vs_v4':out['delta_vs_v4'],'delta_vs_v3':out['delta_vs_v3']},ensure_ascii=False))"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "DONE: reports/v5_comparison_against_v4_and_v3.json"
