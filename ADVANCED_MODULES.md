# Elibot Advanced Modules

This document summarizes advanced modules integrated in the API.

## Modules implemented

1. Tone adaptation (`infer_tone`, `apply_tone`)
2. Metacognition trace (`build_metacognition`)
3. Intelligent rewrite (`/rewrite`)
4. Scenario simulation (`/simulate`)
5. Compliance checks (`compliance_check`)
6. Performance analytics (`/metrics/performance`)
7. User personalization (`/user/{session_id}/preferences`)
8. Internal skill routing (`detect_skill`, `skill_guidance`)
9. Clarification prompts for ambiguous requests
10. Audit trail (`/audit/recent`)
11. Long-context compression (already integrated in session summary)
12. External tool suggestion (`/tools/suggest`)

## New endpoints

- `POST /rewrite`
- `POST /simulate`
- `POST /tools/suggest`
- `GET /audit/recent`
- `GET /metrics/performance`
- `GET /user/{session_id}/preferences`

## Quick examples

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/rewrite -ContentType "application/json" -Body '{"text":"Explique l architecture du pipeline", "mode":"simple"}'
```

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/simulate -ContentType "application/json" -Body '{"plan":{"steps":[{"action":"call_api_get"},{"action":"store_jsonl"}]}}'
```

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/metrics/performance
```
