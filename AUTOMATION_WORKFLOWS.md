# Elibot Automation Workflows

Elibot peut planifier et executer des workflows techniques via l'API.

## Endpoints

- `POST /automation/plan`
- `POST /automation/run`
- `POST /automation/run-integrations`

## 1) Generer un plan de workflow

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/automation/plan -ContentType "application/json" -Body '{"goal":"Appeler une API meteo puis stocker le resultat en JSONL"}'
```

## 2) Executer un workflow en mode simulation (recommande)

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/automation/run -ContentType "application/json" -Body '{"goal":"Appeler https://api.github.com puis stocker en JSONL","dry_run":true}'
```

## 3) Executer un plan explicite

```powershell
$plan = @{
  name = "demo_api"
  steps = @(
    @{ action = "call_api_get"; params = @{ url = "https://api.github.com" } },
    @{ action = "store_jsonl"; params = @{ file_name = "github_demo.jsonl"; data_source = "last_api_response" } },
    @{ action = "respond"; params = @{ message = "Workflow termine" } }
  )
}

$body = @{ plan = $plan; dry_run = $true } | ConvertTo-Json -Depth 8
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/automation/run -ContentType "application/json" -Body $body
```

## Actions autorisees

- `call_api_get`
- `transform_extract`
- `store_jsonl`
- `respond`

## 4) Orchestrer des integrations externes (batch)

```powershell
$headers = @{ "X-API-Key" = "elibot-advanced-key"; "Content-Type" = "application/json" }
$body = @'
{
  "dry_run": true,
  "max_actions": 5,
  "stop_on_error": true,
  "items": [
    {"provider":"notion","action":"create_page","payload":{"title":"Daily Summary","content":"Workflow result"}},
    {"provider":"slack","action":"send_message","payload":{"text":"Summary created"}}
  ]
}
'@
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/automation/run-integrations -Headers $headers -Body $body
```

Securite appliquee:

- role minimum `advanced`
- limite d'actions (`max_actions`, max 20)
- arret sur erreur optionnel (`stop_on_error`)
- audit de chaque execution

## Remarques de securite

- Les URLs locales (`localhost`, `127.0.0.1`) sont refusees.
- Commence par `dry_run=true`.
- Les sorties JSONL sont ecrites dans `data/automation/runs`.
