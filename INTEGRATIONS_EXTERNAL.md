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
