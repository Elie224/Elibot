from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Any


DEFAULT_TIMEOUT_S = 20
INTEGRATION_MAX_RETRIES = int(os.getenv("INTEGRATION_MAX_RETRIES", "2"))
INTEGRATION_BACKOFF_BASE_MS = int(os.getenv("INTEGRATION_BACKOFF_BASE_MS", "300"))
INTEGRATION_CIRCUIT_FAIL_THRESHOLD = int(os.getenv("INTEGRATION_CIRCUIT_FAIL_THRESHOLD", "3"))
INTEGRATION_CIRCUIT_OPEN_SECONDS = int(os.getenv("INTEGRATION_CIRCUIT_OPEN_SECONDS", "60"))
INTEGRATION_RUNTIME_STATE_PATH = Path(os.getenv("INTEGRATION_RUNTIME_STATE_PATH", "data/logs/integration_runtime_state.json"))
INTEGRATION_ALERT_WEBHOOK_URL = os.getenv("INTEGRATION_ALERT_WEBHOOK_URL", "").strip()

_state_lock = Lock()
_provider_health: dict[str, dict[str, Any]] = {}
_provider_alerts: dict[str, dict[str, Any]] = {}
_integration_metrics: dict[str, Any] = {
    "total": 0,
    "success": 0,
    "error": 0,
    "updated_at": "",
    "by_provider": {},
}


def _persist_runtime_state_locked() -> None:
    INTEGRATION_RUNTIME_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider_health": _provider_health,
        "provider_alerts": _provider_alerts,
        "integration_metrics": _integration_metrics,
        "saved_at": _now_iso(),
    }
    INTEGRATION_RUNTIME_STATE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_runtime_state() -> None:
    if not INTEGRATION_RUNTIME_STATE_PATH.exists():
        return
    try:
        raw = json.loads(INTEGRATION_RUNTIME_STATE_PATH.read_text(encoding="utf-8"))
        ph = raw.get("provider_health")
        pa = raw.get("provider_alerts")
        im = raw.get("integration_metrics")
        if isinstance(ph, dict):
            _provider_health.update(ph)
        if isinstance(pa, dict):
            _provider_alerts.update(pa)
        if isinstance(im, dict):
            _integration_metrics.update(im)
            by_provider = im.get("by_provider")
            if isinstance(by_provider, dict):
                _integration_metrics["by_provider"] = by_provider
    except Exception:
        return


def _emit_circuit_alert(provider: str, failures: int, error_message: str, open_until: float) -> None:
    if not INTEGRATION_ALERT_WEBHOOK_URL:
        return
    payload = {
        "event": "integration_circuit_open",
        "provider": provider,
        "consecutive_failures": failures,
        "error": str(error_message)[:500],
        "open_until_epoch": open_until,
        "timestamp": _now_iso(),
    }
    try:
        _http_json("POST", INTEGRATION_ALERT_WEBHOOK_URL, payload=payload)
    except Exception:
        return


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _ensure_provider_state(provider: str) -> None:
    if provider not in _provider_health:
        _provider_health[provider] = {
            "consecutive_failures": 0,
            "circuit_open_until": 0.0,
            "last_error": "",
            "last_failure_at": "",
            "last_success_at": "",
        }
    if provider not in _integration_metrics["by_provider"]:
        _integration_metrics["by_provider"][provider] = {
            "total": 0,
            "success": 0,
            "error": 0,
            "last_latency_ms": 0,
            "updated_at": "",
        }
    if provider not in _provider_alerts:
        _provider_alerts[provider] = {
            "is_open": False,
            "acknowledged": True,
            "opened_at": "",
            "acknowledged_at": "",
            "acknowledged_by": "",
            "last_message": "",
        }


def _mark_success(provider: str, latency_ms: int) -> None:
    with _state_lock:
        _ensure_provider_state(provider)
        _provider_health[provider]["consecutive_failures"] = 0
        _provider_health[provider]["circuit_open_until"] = 0.0
        _provider_health[provider]["last_success_at"] = _now_iso()
        _provider_health[provider]["last_error"] = ""
        _provider_alerts[provider]["is_open"] = False

        _integration_metrics["total"] += 1
        _integration_metrics["success"] += 1
        _integration_metrics["updated_at"] = _now_iso()

        byp = _integration_metrics["by_provider"][provider]
        byp["total"] += 1
        byp["success"] += 1
        byp["last_latency_ms"] = int(latency_ms)
        byp["updated_at"] = _now_iso()
        _persist_runtime_state_locked()


def _mark_error(provider: str, error_message: str) -> None:
    should_alert = False
    alert_until = 0.0
    alert_failures = 0
    with _state_lock:
        _ensure_provider_state(provider)
        failures = int(_provider_health[provider]["consecutive_failures"]) + 1
        _provider_health[provider]["consecutive_failures"] = failures
        _provider_health[provider]["last_error"] = str(error_message)[:400]
        _provider_health[provider]["last_failure_at"] = _now_iso()
        previous_open_until = float(_provider_health[provider].get("circuit_open_until", 0.0))
        if failures >= INTEGRATION_CIRCUIT_FAIL_THRESHOLD:
            next_open_until = time.time() + max(1, INTEGRATION_CIRCUIT_OPEN_SECONDS)
            _provider_health[provider]["circuit_open_until"] = next_open_until
            if previous_open_until <= time.time():
                should_alert = True
                alert_until = next_open_until
                alert_failures = failures
                _provider_alerts[provider] = {
                    "is_open": True,
                    "acknowledged": False,
                    "opened_at": _now_iso(),
                    "acknowledged_at": "",
                    "acknowledged_by": "",
                    "last_message": str(error_message)[:400],
                }

        _integration_metrics["total"] += 1
        _integration_metrics["error"] += 1
        _integration_metrics["updated_at"] = _now_iso()

        byp = _integration_metrics["by_provider"][provider]
        byp["total"] += 1
        byp["error"] += 1
        byp["updated_at"] = _now_iso()
        _persist_runtime_state_locked()

    if should_alert:
        _emit_circuit_alert(
            provider=provider,
            failures=alert_failures,
            error_message=error_message,
            open_until=alert_until,
        )


def get_provider_health() -> dict[str, Any]:
    status = provider_status()
    with _state_lock:
        for provider in status:
            _ensure_provider_state(provider)
        return {
            k: dict(v)
            for k, v in _provider_health.items()
        }


def get_provider_alerts() -> dict[str, Any]:
    status = provider_status()
    with _state_lock:
        for provider in status:
            _ensure_provider_state(provider)
        return {
            k: dict(v)
            for k, v in _provider_alerts.items()
        }


def acknowledge_provider_alert(provider: str, acknowledged_by: str = "admin") -> dict[str, Any]:
    provider = (provider or "").strip().lower()
    status = provider_status()
    if provider not in status:
        raise ValueError(f"unsupported provider: {provider}")

    with _state_lock:
        _ensure_provider_state(provider)
        alert = _provider_alerts[provider]
        alert["acknowledged"] = True
        alert["acknowledged_at"] = _now_iso()
        alert["acknowledged_by"] = str(acknowledged_by)[:120]
        _provider_alerts[provider] = alert
        _persist_runtime_state_locked()
        return dict(alert)


def get_integration_metrics() -> dict[str, Any]:
    status = provider_status()
    with _state_lock:
        for provider in status:
            _ensure_provider_state(provider)
        return {
            "total": int(_integration_metrics["total"]),
            "success": int(_integration_metrics["success"]),
            "error": int(_integration_metrics["error"]),
            "updated_at": str(_integration_metrics.get("updated_at", "")),
            "by_provider": {
                k: dict(v)
                for k, v in _integration_metrics["by_provider"].items()
            },
        }


def _http_json(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    data_bytes = None
    req_headers = dict(headers or {})

    if payload is not None:
        data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(url=url, method=method.upper(), headers=req_headers, data=data_bytes)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {"raw": raw}


def _http_form_post(url: str, form: dict[str, str], timeout_s: int = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    data_bytes = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(url=url, method="POST", data=data_bytes)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {"raw": raw}


def _drive_multipart_body(name: str, content: str, folder_id: str | None = None) -> tuple[bytes, str]:
    boundary = "elibot_boundary_7d2f"
    metadata = {"name": name, "mimeType": "text/plain"}
    if folder_id:
        metadata["parents"] = [folder_id]

    part1 = (
        f"--{boundary}\r\n"
        "Content-Type: application/json; charset=UTF-8\r\n\r\n"
        f"{json.dumps(metadata, ensure_ascii=False)}\r\n"
    )
    part2 = (
        f"--{boundary}\r\n"
        "Content-Type: text/plain; charset=UTF-8\r\n\r\n"
        f"{content}\r\n"
        f"--{boundary}--\r\n"
    )
    body = (part1 + part2).encode("utf-8")
    content_type = f"multipart/related; boundary={boundary}"
    return body, content_type


def provider_status() -> dict[str, dict[str, Any]]:
    return {
        "github": {
            "configured": bool(os.getenv("GITHUB_TOKEN") and os.getenv("GITHUB_REPO")),
            "required_env": ["GITHUB_TOKEN", "GITHUB_REPO"],
            "actions": ["create_issue"],
        },
        "notion": {
            "configured": bool(os.getenv("NOTION_TOKEN") and os.getenv("NOTION_DATABASE_ID")),
            "required_env": ["NOTION_TOKEN", "NOTION_DATABASE_ID"],
            "actions": ["create_page"],
        },
        "google_drive": {
            "configured": bool(os.getenv("GOOGLE_DRIVE_ACCESS_TOKEN")),
            "required_env": ["GOOGLE_DRIVE_ACCESS_TOKEN"],
            "actions": ["create_text_file"],
        },
        "discord": {
            "configured": bool(os.getenv("DISCORD_WEBHOOK_URL")),
            "required_env": ["DISCORD_WEBHOOK_URL"],
            "actions": ["send_message"],
        },
        "slack": {
            "configured": bool(os.getenv("SLACK_WEBHOOK_URL")),
            "required_env": ["SLACK_WEBHOOK_URL"],
            "actions": ["send_message"],
        },
        "trello": {
            "configured": bool(os.getenv("TRELLO_KEY") and os.getenv("TRELLO_TOKEN") and os.getenv("TRELLO_LIST_ID")),
            "required_env": ["TRELLO_KEY", "TRELLO_TOKEN", "TRELLO_LIST_ID"],
            "actions": ["create_card"],
        },
        "jira": {
            "configured": bool(
                os.getenv("JIRA_BASE_URL")
                and os.getenv("JIRA_EMAIL")
                and os.getenv("JIRA_API_TOKEN")
                and os.getenv("JIRA_PROJECT_KEY")
            ),
            "required_env": ["JIRA_BASE_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_PROJECT_KEY"],
            "actions": ["create_issue"],
        },
    }


def integration_templates() -> dict[str, dict[str, Any]]:
    return {
        "github_issue_from_error": {
            "provider": "github",
            "action": "create_issue",
            "description": "Create a GitHub issue from an API/runtime error.",
            "payload_template": {
                "title": "[Elibot] {service} error: {error_code}",
                "body": "Context: {context}\nDetails: {details}",
                "labels": ["bug", "elibot"],
            },
            "required_vars": ["service", "error_code", "context", "details"],
        },
        "notion_page_from_summary": {
            "provider": "notion",
            "action": "create_page",
            "description": "Create a Notion page from session summary.",
            "payload_template": {
                "title": "Session summary - {project}",
                "content": "Date: {date}\nKey points:\n{summary}",
            },
            "required_vars": ["project", "date", "summary"],
        },
        "slack_alert_from_audit": {
            "provider": "slack",
            "action": "send_message",
            "description": "Send an audit alert message to Slack.",
            "payload_template": {
                "text": "[Elibot alert] {level}: {message} (ref: {ref_id})",
            },
            "required_vars": ["level", "message", "ref_id"],
        },
        "discord_alert_from_audit": {
            "provider": "discord",
            "action": "send_message",
            "description": "Send an audit alert message to Discord.",
            "payload_template": {
                "text": "[Elibot alert] {level}: {message} (ref: {ref_id})",
            },
            "required_vars": ["level", "message", "ref_id"],
        },
        "trello_card_from_task": {
            "provider": "trello",
            "action": "create_card",
            "description": "Create a Trello card from a technical task.",
            "payload_template": {
                "name": "{task_title}",
                "desc": "Owner: {owner}\nPriority: {priority}\nNotes: {notes}",
            },
            "required_vars": ["task_title", "owner", "priority", "notes"],
        },
        "jira_issue_from_incident": {
            "provider": "jira",
            "action": "create_issue",
            "description": "Create a Jira issue from an incident.",
            "payload_template": {
                "issue_type": "Task",
                "summary": "{service} incident - {impact}",
                "description": "When: {date}\nImpact: {impact}\nDetails: {details}",
            },
            "required_vars": ["service", "date", "impact", "details"],
        },
        "drive_file_from_summary": {
            "provider": "google_drive",
            "action": "create_text_file",
            "description": "Create a text file in Google Drive from summary.",
            "payload_template": {
                "name": "{project}_{date}_summary.txt",
                "content": "Project: {project}\nDate: {date}\n{summary}",
            },
            "required_vars": ["project", "date", "summary"],
        },
    }


def _render_value(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**variables)
    if isinstance(value, list):
        return [_render_value(v, variables) for v in value]
    if isinstance(value, dict):
        return {k: _render_value(v, variables) for k, v in value.items()}
    return value


def build_template_request(template_id: str, variables: dict[str, Any]) -> dict[str, Any]:
    catalog = integration_templates()
    tid = (template_id or "").strip()
    if tid not in catalog:
        raise ValueError(f"unknown template_id: {template_id}")

    meta = catalog[tid]
    required_vars = list(meta.get("required_vars", []))
    missing = [k for k in required_vars if k not in variables or variables.get(k) in {None, ""}]
    if missing:
        raise ValueError("missing template variables: " + ", ".join(missing))

    try:
        payload = _render_value(meta.get("payload_template", {}), variables)
    except KeyError as exc:
        raise ValueError(f"missing template variable: {exc}") from exc

    return {
        "template_id": tid,
        "provider": meta["provider"],
        "action": meta["action"],
        "payload": payload,
    }


def _execute_integration_once(provider: str, action: str, payload: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    provider = (provider or "").strip().lower()
    action = (action or "").strip().lower()

    status = provider_status()
    if provider not in status:
        raise ValueError(f"unsupported provider: {provider}")
    if action not in status[provider]["actions"]:
        raise ValueError(f"unsupported action '{action}' for provider '{provider}'")

    if dry_run:
        return {
            "dry_run": True,
            "provider": provider,
            "action": action,
            "payload": payload,
            "configured": status[provider]["configured"],
        }

    if not status[provider]["configured"]:
        req = ", ".join(status[provider]["required_env"])
        raise ValueError(f"provider '{provider}' not configured. missing env in: {req}")

    if provider == "github" and action == "create_issue":
        repo = os.getenv("GITHUB_REPO", "")
        token = os.getenv("GITHUB_TOKEN", "")
        url = f"https://api.github.com/repos/{repo}/issues"
        body = {
            "title": str(payload.get("title") or "Elibot issue"),
            "body": str(payload.get("body") or ""),
        }
        labels = payload.get("labels")
        if isinstance(labels, list):
            body["labels"] = labels
        res = _http_json("POST", url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}, payload=body)
        return {"provider": provider, "action": action, "result": res}

    if provider == "notion" and action == "create_page":
        token = os.getenv("NOTION_TOKEN", "")
        db_id = os.getenv("NOTION_DATABASE_ID", "")
        url = "https://api.notion.com/v1/pages"
        title = str(payload.get("title") or "Elibot note")
        content = str(payload.get("content") or "")
        body = {
            "parent": {"database_id": db_id},
            "properties": {
                "Name": {
                    "title": [{"text": {"content": title}}]
                }
            },
            "children": [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": content[:1800]}}]},
                }
            ],
        }
        res = _http_json(
            "POST",
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Notion-Version": "2022-06-28",
                "Content-Type": "application/json",
            },
            payload=body,
        )
        return {"provider": provider, "action": action, "result": res}

    if provider == "google_drive" and action == "create_text_file":
        access_token = os.getenv("GOOGLE_DRIVE_ACCESS_TOKEN", "")
        folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        name = str(payload.get("name") or "elibot_note.txt")
        content = str(payload.get("content") or "")
        body, ctype = _drive_multipart_body(name=name, content=content, folder_id=folder_id)

        req = urllib.request.Request(
            url="https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
            method="POST",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": ctype,
            },
            data=body,
        )
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        try:
            res = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            res = {"raw": raw}
        return {"provider": provider, "action": action, "result": res}

    if provider == "discord" and action == "send_message":
        webhook = os.getenv("DISCORD_WEBHOOK_URL", "")
        text = str(payload.get("text") or payload.get("content") or "Elibot message")
        res = _http_json("POST", webhook, payload={"content": text})
        return {"provider": provider, "action": action, "result": res}

    if provider == "slack" and action == "send_message":
        webhook = os.getenv("SLACK_WEBHOOK_URL", "")
        text = str(payload.get("text") or "Elibot message")
        res = _http_json("POST", webhook, payload={"text": text})
        return {"provider": provider, "action": action, "result": res}

    if provider == "trello" and action == "create_card":
        key = os.getenv("TRELLO_KEY", "")
        token = os.getenv("TRELLO_TOKEN", "")
        list_id = os.getenv("TRELLO_LIST_ID", "")
        url = "https://api.trello.com/1/cards"
        form = {
            "key": key,
            "token": token,
            "idList": list_id,
            "name": str(payload.get("name") or "Elibot card"),
            "desc": str(payload.get("desc") or ""),
        }
        res = _http_form_post(url, form)
        return {"provider": provider, "action": action, "result": res}

    if provider == "jira" and action == "create_issue":
        base_url = os.getenv("JIRA_BASE_URL", "").rstrip("/")
        email = os.getenv("JIRA_EMAIL", "")
        api_token = os.getenv("JIRA_API_TOKEN", "")
        project_key = os.getenv("JIRA_PROJECT_KEY", "")
        auth = base64.b64encode(f"{email}:{api_token}".encode("utf-8")).decode("ascii")

        issue_type = str(payload.get("issue_type") or "Task")
        summary = str(payload.get("summary") or "Elibot issue")
        description = str(payload.get("description") or "")

        body = {
            "fields": {
                "project": {"key": project_key},
                "summary": summary,
                "description": {
                    "type": "doc",
                    "version": 1,
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": description[:3000]}],
                        }
                    ],
                },
                "issuetype": {"name": issue_type},
            }
        }

        res = _http_json(
            "POST",
            f"{base_url}/rest/api/3/issue",
            headers={
                "Authorization": f"Basic {auth}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            payload=body,
        )
        return {"provider": provider, "action": action, "result": res}

    raise ValueError("unsupported provider/action combination")


def execute_integration(provider: str, action: str, payload: dict[str, Any], dry_run: bool = True) -> dict[str, Any]:
    provider = (provider or "").strip().lower()
    action = (action or "").strip().lower()

    if dry_run:
        return _execute_integration_once(provider=provider, action=action, payload=payload, dry_run=True)

    with _state_lock:
        _ensure_provider_state(provider)
        open_until = float(_provider_health[provider].get("circuit_open_until", 0.0))
    if open_until > time.time():
        raise RuntimeError(f"circuit open for provider '{provider}'")

    attempts = max(1, INTEGRATION_MAX_RETRIES + 1)
    last_exc: Exception | None = None

    for attempt in range(1, attempts + 1):
        started = time.time()
        try:
            result = _execute_integration_once(provider=provider, action=action, payload=payload, dry_run=False)
            latency_ms = int((time.time() - started) * 1000)
            _mark_success(provider, latency_ms=latency_ms)
            return {
                **result,
                "retry": {
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "latency_ms": latency_ms,
                },
            }
        except Exception as exc:
            last_exc = exc
            _mark_error(provider, str(exc))
            if attempt < attempts:
                backoff_s = (max(1, INTEGRATION_BACKOFF_BASE_MS) / 1000.0) * attempt
                time.sleep(backoff_s)

    raise RuntimeError(f"integration failed after retries: {last_exc}")


_load_runtime_state()
