# Security Hardening

Elibot API now includes:

- API key authentication
- role-based access control (basic, advanced, admin)
- rate limiting per minute
- daily quota enforcement
- persistent API key store with rotation endpoints
- metrics dashboard and CSV export

## Environment variables

- `REQUIRE_API_KEY` (default `true`)
- `API_KEYS` format: `key1:admin,key2:advanced,key3:basic`
- `RATE_LIMIT_BASIC_PER_MIN` (default `30`)
- `RATE_LIMIT_ADVANCED_PER_MIN` (default `120`)
- `RATE_LIMIT_ADMIN_PER_MIN` (default `300`)
- `DAILY_QUOTA_BASIC` (default `800`)
- `DAILY_QUOTA_ADVANCED` (default `5000`)
- `DAILY_QUOTA_ADMIN` (default `20000`)

## Default development keys

If `API_KEYS` is not set, the API uses:

- `elibot-admin-key` -> admin
- `elibot-advanced-key` -> advanced
- `elibot-basic-key` -> basic

Rotate these keys in production.

## Example calls

```powershell
$headers = @{ "X-API-Key" = "elibot-admin-key"; "Content-Type" = "application/json" }
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/metrics/performance -Headers $headers
```

```powershell
$headers = @{ "X-API-Key" = "elibot-advanced-key"; "Content-Type" = "application/json" }
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/automation/plan -Headers $headers -Body '{"goal":"Automatiser un workflow API"}'
```

## Dashboard

- HTML dashboard: `GET /dashboard/metrics` (admin)
- CSV export: `GET /metrics/performance.csv` (admin)

## Key rotation endpoints (admin)

- `GET /admin/keys`
- `POST /admin/keys`
- `POST /admin/keys/revoke`

Example create key:

```powershell
$headers = @{ "X-API-Key" = "elibot-admin-key"; "Content-Type" = "application/json" }
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/admin/keys -Headers $headers -Body '{"role":"advanced","label":"ci-runner"}'
```

## Access/Quota status endpoint

- `GET /access/status` (basic+)

Returns current principal role, rate limit, and quota usage.

## External tools integration

Elibot also supports a unified integration API for GitHub, Notion, Google Drive, Discord, Slack, Trello and Jira.

See `INTEGRATIONS_EXTERNAL.md` for endpoint usage, provider actions and required environment variables.

Additional secured endpoints:

- `POST /integrations/execute-async` (advanced)
- `GET /integrations/jobs/{job_id}` (owner or admin)
- `POST /integrations/jobs/purge` (admin)
- `GET /integrations/metrics` (admin)
- `GET /dashboard/integrations` (admin)

Operational controls:

- async jobs are persisted on disk and retained with TTL
- manual purge endpoint for expired jobs
- optional webhook alerts when an integration provider circuit breaker opens
