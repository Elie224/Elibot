# Intent Classification for Elibot

Elibot includes an intent classifier to route requests before generation.

## Endpoint

- `POST /intent/classify`

### Example

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/intent/classify -ContentType "application/json" -Body '{"message":"Peux-tu me faire un workflow API + stockage JSONL ?"}'
```

Expected fields:

- `intent`: `greeting|gratitude|technical_question|automation_request|code_request|out_of_domain|unknown`
- `confidence`: float
- `reasons`: list of matched signals

## Chat Routing Behavior

In `POST /chat`:

- `automation_request` can be routed to workflow planning guidance.
- `out_of_domain` is redirected by domain policies.
- all events are logged with intent metadata in `data/logs/elibot_chat_events.jsonl`.
