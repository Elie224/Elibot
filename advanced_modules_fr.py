import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_LOG_PATH = Path("data/logs/elibot_audit.jsonl")
PERF_METRICS_PATH = Path("data/logs/elibot_perf_metrics.json")
USER_PREFS_PATH = Path("data/memory/user_preferences.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def infer_tone(user_text: str, profile: dict[str, str], intent: dict) -> str:
    q = user_text.lower()
    if "explique" in q or "debutant" in q or "simple" in q:
        return "pedagogique"
    if "detail" in q or "complet" in q:
        return "detaille"
    if "rapid" in q or "court" in q or "resume" in q:
        return "concis"
    if intent.get("intent") in {"automation_request", "code_request", "technical_question"}:
        return "expert"
    if profile.get("tone_preference"):
        return profile["tone_preference"]
    return "professionnel"


def apply_tone(answer: str, tone: str) -> str:
    text = " ".join((answer or "").split())
    if not text:
        return text

    if tone == "concis" and len(text) > 240:
        return text[:237].rstrip() + "..."

    if tone == "pedagogique" and not text.lower().startswith("voici"):
        return "Voici une explication simple: " + text

    if tone == "expert" and "1)" not in text and len(text) > 80:
        return "Approche recommandee: " + text

    return text


def build_metacognition(user_text: str, intent: dict, selected_sources: list[str] | None = None) -> dict:
    return {
        "intent": intent,
        "strategy": [
            "classer_la_demande",
            "recuperer_contexte",
            "generer_reponse",
            "verifier_qualite",
        ],
        "sources": selected_sources or [],
        "note": f"Traitement de la demande: {user_text[:140]}",
    }


def rewrite_text(text: str, mode: str) -> str:
    value = " ".join((text or "").split())
    if not value:
        return value

    if mode == "simple":
        value = re.sub(r"\bimplementation\b", "mise en place", value, flags=re.IGNORECASE)
        value = re.sub(r"\barchitecture\b", "organisation", value, flags=re.IGNORECASE)
        return value

    if mode == "formal":
        if not value.endswith("."):
            value += "."
        return "Veuillez noter: " + value

    if mode == "bullet":
        parts = re.split(r"(?<=[.!?])\s+", value)
        parts = [p.strip() for p in parts if p.strip()]
        return "\n".join(f"- {p}" for p in parts)

    return value


def simulate_workflow(plan: dict) -> dict:
    steps = plan.get("steps", []) if isinstance(plan, dict) else []
    simulation = []
    for i, step in enumerate(steps, start=1):
        action = step.get("action", "unknown") if isinstance(step, dict) else "unknown"
        simulation.append(
            {
                "step": i,
                "action": action,
                "status": "simulated",
                "estimated_ms": 120 + i * 30,
            }
        )

    return {
        "simulated": True,
        "steps": simulation,
        "total_steps": len(simulation),
    }


def compliance_check(answer: str, in_domain: bool) -> dict:
    text = (answer or "").lower()
    violations = []

    if in_domain and any(x in text for x in ["politique", "diagnostic", "astrologie", "religion"]):
        violations.append("out_of_domain_content")

    if len(text.split()) < 3:
        violations.append("too_short")

    return {"ok": len(violations) == 0, "violations": violations}


def update_performance_metrics(event: dict) -> dict:
    metrics = _read_json(
        PERF_METRICS_PATH,
        {
            "total": 0,
            "corrected_by_verifier": 0,
            "rule_reply": 0,
            "out_of_domain": 0,
            "by_intent": {},
            "updated_at": _now_iso(),
        },
    )

    metrics["total"] += 1
    if event.get("corrected_by_verifier"):
        metrics["corrected_by_verifier"] += 1
    if event.get("used_rule_reply"):
        metrics["rule_reply"] += 1
    if not event.get("in_domain", True):
        metrics["out_of_domain"] += 1

    intent_name = (event.get("intent") or {}).get("intent", "unknown")
    by_intent = metrics.get("by_intent", {})
    by_intent[intent_name] = by_intent.get(intent_name, 0) + 1
    metrics["by_intent"] = by_intent
    metrics["updated_at"] = _now_iso()

    _write_json(PERF_METRICS_PATH, metrics)
    return metrics


def get_performance_metrics() -> dict:
    return _read_json(PERF_METRICS_PATH, {"total": 0, "by_intent": {}, "updated_at": _now_iso()})


def update_user_preferences(session_id: str, user_text: str) -> dict:
    prefs = _read_json(USER_PREFS_PATH, {})
    profile = prefs.get(session_id, {})

    q = user_text.lower()
    if "reponse courte" in q or "sois court" in q:
        profile["tone_preference"] = "concis"
    elif "detail" in q or "explique" in q:
        profile["tone_preference"] = "detaille"

    if "python" in q:
        profile["favorite_tooling"] = "python"
    elif "sql" in q:
        profile["favorite_tooling"] = "sql"

    profile["updated_at"] = _now_iso()
    prefs[session_id] = profile
    _write_json(USER_PREFS_PATH, prefs)
    return profile


def get_user_preferences(session_id: str) -> dict:
    prefs = _read_json(USER_PREFS_PATH, {})
    return prefs.get(session_id, {})


def detect_skill(user_text: str) -> str:
    q = user_text.lower()
    if any(k in q for k in ["dataset", "csv", "nettoyage"]):
        return "analyse_dataset"
    if any(k in q for k in ["code", "python", "bug", "erreur"]):
        return "debug_code"
    if any(k in q for k in ["pipeline ml", "model", "modele", "entrainement"]):
        return "pipeline_ml"
    if any(k in q for k in ["workflow", "automatisation", "n8n", "api"]):
        return "workflow_automation"
    return "general_technical"


def skill_guidance(skill: str) -> str:
    templates = {
        "analyse_dataset": "Skill active: analyse_dataset. Je peux proposer schema, nettoyage, validation et metriques.",
        "debug_code": "Skill active: debug_code. Donne le stacktrace et un extrait minimal reproductible.",
        "pipeline_ml": "Skill active: pipeline_ml. Je peux fournir architecture, preprocessing, training et evaluation.",
        "workflow_automation": "Skill active: workflow_automation. Je peux construire un plan d orchestration en JSON.",
        "general_technical": "Skill active: general_technical. Je reste sur data, IA et automatisation.",
    }
    return templates.get(skill, templates["general_technical"])


def clarification_prompt(user_text: str) -> str | None:
    q = user_text.strip()
    if len(q) <= 10:
        return "Peux-tu preciser le contexte, l entree, la sortie attendue et la contrainte principale ?"
    if "aide moi" in q.lower() and not any(k in q.lower() for k in ["python", "api", "dataset", "sql", "pipeline"]):
        return "Je peux t aider. Precise si c est Python, SQL, API, dataset ou pipeline ML."
    return None


def tool_suggestions(user_text: str) -> list[str]:
    q = user_text.lower()
    tools = []
    if "github" in q:
        tools.append("github")
    if "notion" in q:
        tools.append("notion")
    if "slack" in q or "discord" in q:
        tools.append("messaging_webhook")
    if "drive" in q:
        tools.append("google_drive")
    if "jira" in q or "trello" in q:
        tools.append("project_management")
    return tools


def write_audit(event: dict) -> None:
    payload = {"timestamp": _now_iso(), **event}
    _append_jsonl(AUDIT_LOG_PATH, payload)


def read_recent_audit(limit: int = 20) -> list[dict]:
    if not AUDIT_LOG_PATH.exists():
        return []

    lines = AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines[-limit:]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out
