from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_TIMEOUT_S = 20


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


def execute_integration(provider: str, action: str, payload: dict[str, Any], dry_run: bool = True) -> dict[str, Any]:
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
