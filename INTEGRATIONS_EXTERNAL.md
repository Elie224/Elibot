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
- `GET /integrations/templates`
- `POST /integrations/execute-template`
- `POST /automation/run-integrations`

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
