# Elibot Next-Gen Modules

This layer adds 10 advanced modules on top of existing Elibot capabilities.

## Included modules

1. Context Injection
2. Automatic Tool Selection
3. Internal State Machine
4. Task Memory
5. Knowledge Distillation Plan
6. Sandbox Simulation
7. Advanced Guardrails
8. Multi-Persona Routing
9. Internal Metrics Scores
10. Advanced Memory Compression

## New files

- `nextgen_orchestrator_fr.py`
- `task_memory_fr.py`

## API endpoints

- `GET /nextgen/decision`
- `POST /sandbox/simulate-action`
- `POST /tasks/upsert`
- `POST /tasks/progress`
- `GET /tasks/{session_id}`
- `GET /nextgen/distillation-plan`
- `GET /metrics/internal-scores`

## Integration in /chat

The chat pipeline now includes:

- persona detection
- tool decision
- state trace
- context injection
- advanced guardrails
- internal confidence/coherence/quality/risk/uncertainty scoring
- advanced memory compression for summaries

## Quick tests

Decision preview:

```powershell
$headers = @{ "X-API-Key" = "elibot-basic-key" }
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/nextgen/decision?message=Automatiser%20un%20workflow%20API" -Headers $headers
```

Create task:

```powershell
$headers = @{ "X-API-Key" = "elibot-advanced-key"; "Content-Type" = "application/json" }
$body = '{"session_id":"demo-session","title":"Pipeline MLOps","steps":["collect data","train model","deploy api"]}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/tasks/upsert -Headers $headers -Body $body
```

Sandbox simulation:

```powershell
$headers = @{ "X-API-Key" = "elibot-advanced-key"; "Content-Type" = "application/json" }
$body = '{"action":"api_call","payload":{"url":"https://api.github.com"}}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/sandbox/simulate-action -Headers $headers -Body $body
```
