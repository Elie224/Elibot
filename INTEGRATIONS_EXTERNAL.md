# External Integrations (Point 12)

Elibot exposes a unified integration layer for:

- GitHub
- Notion
- Google Drive
- Discord
- Slack
- Trello
- Jira

## Endpoints

- `GET /integrations/providers`
- `POST /integrations/execute`
- `POST /integrations/execute-async`
- `GET /integrations/jobs/{job_id}`
- `POST /integrations/jobs/purge` (admin)
- `GET /integrations/templates`
- `POST /integrations/execute-template`
- `POST /automation/run-integrations`
- `GET /integrations/metrics` (admin)
- `GET /dashboard/integrations` (admin)

`/integrations/execute` is protected with role `advanced`.

## Dry-run first

Always start with `dry_run=true`.

```powershell
$headers = @{ "X-API-Key" = "elibot-advanced-key"; "Content-Type" = "application/json" }
$body = '{"provider":"github","action":"create_issue","payload":{"title":"Test","body":"From Elibot"},"dry_run":true}'
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/integrations/execute -Headers $headers -Body $body
```

## Provider env variables

### GitHub
- `GITHUB_TOKEN`
- `GITHUB_REPO` (example: `owner/repo`)

### Notion
- `NOTION_TOKEN`
- `NOTION_DATABASE_ID`

### Google Drive
- `GOOGLE_DRIVE_ACCESS_TOKEN`
- optional: `GOOGLE_DRIVE_FOLDER_ID`

### Discord
- `DISCORD_WEBHOOK_URL`

### Slack
- `SLACK_WEBHOOK_URL`

### Trello
- `TRELLO_KEY`
- `TRELLO_TOKEN`
- `TRELLO_LIST_ID`

### Jira
- `JIRA_BASE_URL`
- `JIRA_EMAIL`
- `JIRA_API_TOKEN`
- `JIRA_PROJECT_KEY`

## Resilience and queue env variables

- `INTEGRATION_MAX_RETRIES` (default `2`)
- `INTEGRATION_BACKOFF_BASE_MS` (default `300`)
- `INTEGRATION_CIRCUIT_FAIL_THRESHOLD` (default `3`)
- `INTEGRATION_CIRCUIT_OPEN_SECONDS` (default `60`)
- `INTEGRATION_RUNTIME_STATE_PATH` (default `data/logs/integration_runtime_state.json`)
- `INTEGRATION_ALERT_WEBHOOK_URL` (optional, receives circuit-open alerts)
- `INTEGRATION_QUEUE_MAX_SIZE` (default `1000`)
- `INTEGRATION_MAX_PENDING_PER_PRINCIPAL` (default `50`)
- `INTEGRATION_JOB_TTL_SECONDS` (default `172800`)
- `INTEGRATION_JOBS_PATH` (default `data/automation/integration_jobs.json`)

## Actions supported

- GitHub: `create_issue`
- Notion: `create_page`
- Google Drive: `create_text_file`
- Discord: `send_message`
- Slack: `send_message`
- Trello: `create_card`
- Jira: `create_issue`

## Template execution

List templates:

```powershell
$headers = @{ "X-API-Key" = "elibot-basic-key" }
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/integrations/templates -Headers $headers
```

Run a template (dry-run):

```powershell
$headers = @{ "X-API-Key" = "elibot-advanced-key"; "Content-Type" = "application/json" }
$body = @'
{
	"template_id":"github_issue_from_error",
	"variables":{
		"service":"api_server",
		"error_code":"HTTP500",
		"context":"/chat",
		"details":"unexpected integration failure"
	},
	"dry_run":true
}
'@
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/integrations/execute-template -Headers $headers -Body $body
```

## Batch integration automation

You can execute multiple integration actions in one request with safeguards (`max_actions`, `stop_on_error`).

```powershell
$headers = @{ "X-API-Key" = "elibot-advanced-key"; "Content-Type" = "application/json" }
$body = @'
{
	"dry_run": true,
	"max_actions": 5,
	"stop_on_error": true,
	"items": [
		{"provider":"slack","action":"send_message","payload":{"text":"Test alert from Elibot"}},
		{"provider":"github","action":"create_issue","payload":{"title":"Integration test","body":"From automation"}}
	]
}
'@
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/automation/run-integrations -Headers $headers -Body $body
```

## Async execution queue

Submit an async integration job:

```powershell
$headers = @{ "X-API-Key" = "elibot-advanced-key"; "Content-Type" = "application/json" }
$job = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/integrations/execute-async -Headers $headers -Body '{"provider":"slack","action":"send_message","payload":{"text":"Async test"},"dry_run":true}'
$job
```

Check job status:

```powershell
$headers = @{ "X-API-Key" = "elibot-advanced-key" }
Invoke-RestMethod -Method Get -Uri ("http://127.0.0.1:8000/integrations/jobs/" + $job.job_id) -Headers $headers
```

## Operations dashboard

- Metrics JSON: `GET /integrations/metrics` (admin)
- HTML dashboard: `GET /dashboard/integrations` (admin)

## Jobs retention and purge

Completed async jobs are persisted and automatically purged after TTL.

Manual purge:

```powershell
$headers = @{ "X-API-Key" = "elibot-admin-key" }
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/integrations/jobs/purge -Headers $headers
```
