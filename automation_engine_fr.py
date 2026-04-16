import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ALLOWED_ACTIONS = {"call_api_get", "transform_extract", "store_jsonl", "respond"}
DEFAULT_RUNS_DIR = Path("data/automation/runs")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = text[start : i + 1]
                try:
                    obj = json.loads(block)
                except json.JSONDecodeError:
                    return None
                if isinstance(obj, dict):
                    return obj
                return None
    return None


def _safe_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc:
        return False

    host = (parsed.hostname or "").lower()
    forbidden = {"localhost", "127.0.0.1", "0.0.0.0"}
    if host in forbidden:
        return False

    return True


def _deep_get(data: Any, dotted_path: str) -> Any:
    value = data
    for part in dotted_path.split("."):
        part = part.strip()
        if not part:
            continue
        if isinstance(value, dict) and part in value:
            value = value[part]
            continue
        if isinstance(value, list) and part.isdigit():
            idx = int(part)
            if 0 <= idx < len(value):
                value = value[idx]
                continue
        return None
    return value


def sanitize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    name = str(plan.get("name") or "workflow_elibot")
    raw_steps = plan.get("steps")
    if not isinstance(raw_steps, list):
        raw_steps = []

    steps: list[dict[str, Any]] = []
    for step in raw_steps[:10]:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").strip()
        if action not in ALLOWED_ACTIONS:
            continue
        params = step.get("params")
        if not isinstance(params, dict):
            params = {}
        steps.append({"action": action, "params": params})

    if not steps:
        steps = [
            {
                "action": "respond",
                "params": {
                    "message": "Je peux concevoir un workflow technique. Donne un objectif data/IA/API plus precis.",
                },
            }
        ]

    return {"name": name, "steps": steps}


def heuristic_plan(goal: str) -> dict[str, Any]:
    goal_l = goal.lower()
    plan = {
        "name": "workflow_elibot_heuristic",
        "steps": [],
    }

    if "api" in goal_l or "http" in goal_l or "webhook" in goal_l:
        url_match = re.search(r"https?://[^\s]+", goal)
        url = url_match.group(0) if url_match else "https://api.github.com"
        plan["steps"].append({"action": "call_api_get", "params": {"url": url}})

    if "extra" in goal_l or "extract" in goal_l or "champ" in goal_l or "field" in goal_l:
        plan["steps"].append(
            {
                "action": "transform_extract",
                "params": {"path": ""},
            }
        )

    if "sauve" in goal_l or "stock" in goal_l or "jsonl" in goal_l or "base" in goal_l:
        plan["steps"].append(
            {
                "action": "store_jsonl",
                "params": {"file_name": "workflow_output.jsonl", "data_source": "last_api_response"},
            }
        )

    plan["steps"].append(
        {
            "action": "respond",
            "params": {
                "message": "Workflow prepare. Je peux l'ajuster avec tes contraintes (auth, retries, validation).",
            },
        }
    )

    return sanitize_plan(plan)


def build_workflow_plan(
    goal: str,
    model: Any,
    tokenizer: Any,
    device: str,
    max_input_length: int = 512,
    max_new_tokens: int = 220,
) -> tuple[dict[str, Any], str]:
    prompt = (
        "Systeme: Tu es un architecte d'automatisation. Reponds UNIQUEMENT en JSON valide.\n"
        "Schema attendu: {\"name\":\"...\",\"steps\":[{\"action\":\"call_api_get|transform_extract|store_jsonl|respond\",\"params\":{...}}]}\n"
        "Contraintes: max 6 steps, aucune action hors schema, pas de texte hors JSON.\n"
        f"Objectif utilisateur: {goal}\n"
        "Assistant:"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(device)
        with torch_no_grad(model):
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.08,
                no_repeat_ngram_size=3,
                num_beams=1,
            )
        raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        candidate = _extract_json_object(raw)
        if candidate is not None:
            return sanitize_plan(candidate), "model"
    except Exception:
        pass

    return heuristic_plan(goal), "heuristic"


class torch_no_grad:
    def __init__(self, model: Any):
        self._model = model
        self._ctx = None

    def __enter__(self):
        import torch

        self._ctx = torch.no_grad()
        self._ctx.__enter__()
        return self._model

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)
        return False


def _http_get_json(url: str, timeout_s: int = 15) -> Any:
    req = Request(url, headers={"User-Agent": "Elibot-Automation/1.0"}, method="GET")
    with urlopen(req, timeout=timeout_s) as resp:
        payload = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"raw": payload}


def execute_plan(plan: dict[str, Any], dry_run: bool = True) -> dict[str, Any]:
    safe_plan = sanitize_plan(plan)
    context: dict[str, Any] = {
        "last_api_response": None,
        "extracted": None,
        "stored_path": None,
        "last_message": None,
    }
    step_results: list[dict[str, Any]] = []

    for idx, step in enumerate(safe_plan["steps"], start=1):
        action = step["action"]
        params = step.get("params", {})
        result: dict[str, Any] = {"index": idx, "action": action, "status": "ok"}

        try:
            if action == "call_api_get":
                url = str(params.get("url") or "")
                if not _safe_url(url):
                    raise ValueError("URL non autorisee")
                if dry_run:
                    result["output"] = {"dry_run": True, "url": url}
                else:
                    data = _http_get_json(url)
                    context["last_api_response"] = data
                    result["output_preview"] = str(data)[:300]

            elif action == "transform_extract":
                path = str(params.get("path") or "").strip()
                source_key = str(params.get("source") or "last_api_response")
                source = context.get(source_key)
                if path:
                    extracted = _deep_get(source, path)
                else:
                    extracted = source
                context["extracted"] = extracted
                result["output_preview"] = str(extracted)[:300]

            elif action == "store_jsonl":
                file_name = str(params.get("file_name") or "workflow_output.jsonl")
                file_name = file_name.replace("..", "").replace("/", "_").replace("\\", "_")
                data_source = str(params.get("data_source") or "extracted")
                payload = context.get(data_source)
                if payload is None:
                    payload = context.get("last_api_response")
                out_path = DEFAULT_RUNS_DIR / file_name
                if dry_run:
                    result["output"] = {"dry_run": True, "file": str(out_path), "data_source": data_source}
                else:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    row = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "payload": payload,
                    }
                    with out_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    context["stored_path"] = str(out_path)
                    result["output"] = {"file": str(out_path)}

            elif action == "respond":
                msg = str(params.get("message") or "Workflow execute.")
                context["last_message"] = msg
                result["output"] = msg

            else:
                raise ValueError("Action non supportee")

        except Exception as exc:
            result["status"] = "error"
            result["error"] = str(exc)

        step_results.append(result)

    return {
        "dry_run": dry_run,
        "plan": safe_plan,
        "steps": step_results,
        "context": {
            "stored_path": context.get("stored_path"),
            "last_message": context.get("last_message"),
        },
    }
