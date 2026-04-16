from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INTERNAL_METRICS_PATH = Path("data/logs/elibot_internal_metrics.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_persona(user_text: str, intent: dict[str, Any], skill: str) -> str:
    q = (user_text or "").lower()
    if any(k in q for k in ["debug", "erreur", "stacktrace", "exception"]) or skill == "debug_code":
        return "debugging"
    if any(k in q for k in ["dataset", "csv", "sql", "analyse"]) or skill == "analyse_dataset":
        return "data_analyst"
    if any(k in q for k in ["workflow", "automatisation", "n8n", "orchestration"]) or skill == "workflow_automation":
        return "automation"
    if any(k in q for k in ["doc", "documentation", "readme", "spec"]):
        return "documentation"
    if intent.get("intent") in {"technical_question", "code_request"}:
        return "expert_ia"
    return "expert_ia"


def select_tools(user_text: str, intent: dict[str, Any], profile: dict[str, str], has_tasks: bool) -> dict[str, Any]:
    q = (user_text or "").lower()
    selected: list[str] = []
    reasons: list[str] = []

    if intent.get("intent") == "automation_request" or "workflow" in q:
        selected.append("automation_plan")
        reasons.append("automation intent detected")

    if any(k in q for k in ["resume", "résumé", "synthese", "synthèse"]):
        selected.append("memory_summary")
        reasons.append("summarization keywords")

    if any(k in q for k in ["api", "github", "jira", "notion", "slack", "discord", "trello", "drive"]):
        selected.append("integrations")
        reasons.append("external tool keywords")

    if any(k in q for k in ["expliquer", "pourquoi", "justifie", "justifier"]):
        selected.append("metacognition")
        reasons.append("explanation requested")

    if has_tasks or any(k in q for k in ["tache", "tâche", "etape", "étape", "avancement"]):
        selected.append("task_memory")
        reasons.append("task tracking context")

    if not selected:
        selected.append("generation")
        reasons.append("default generation path")

    return {
        "selected": selected,
        "reasons": reasons,
        "profile_hint": profile.get("favorite_tooling", ""),
    }


def build_state_machine_trace(tool_selection: dict[str, Any]) -> list[dict[str, Any]]:
    states = [
        {"state": "analyse", "status": "done"},
        {"state": "planification", "status": "done"},
    ]

    selected = tool_selection.get("selected", [])
    if "automation_plan" in selected or "integrations" in selected:
        states.append({"state": "execution", "status": "planned"})
    else:
        states.append({"state": "execution", "status": "skipped"})

    states.append({"state": "verification", "status": "done"})
    states.append({"state": "reponse_finale", "status": "done"})
    return states


def _trim(value: str, max_chars: int) -> str:
    v = " ".join((value or "").split())
    if len(v) <= max_chars:
        return v
    return v[: max(0, max_chars - 3)].rstrip() + "..."


def build_context_injection(
    user_text: str,
    summary: str,
    knowledge_context: str,
    profile: dict[str, str],
    tasks: list[dict[str, Any]],
    max_chars: int = 1600,
) -> dict[str, Any]:
    topic = "general_technical"
    q = user_text.lower()
    if any(k in q for k in ["api", "webhook", "endpoint"]):
        topic = "api_integration"
    elif any(k in q for k in ["dataset", "csv", "sql", "analyse"]):
        topic = "data_analysis"
    elif any(k in q for k in ["workflow", "automatisation"]):
        topic = "automation"

    task_lines = []
    for t in tasks[:3]:
        task_lines.append(
            f"- {t.get('title', 'task')} | status={t.get('status', 'active')} | remaining={t.get('remaining_steps', 0)}"
        )

    blocks = [
        f"[topic]\n{topic}",
        f"[profile]\n{json.dumps(profile, ensure_ascii=False)}" if profile else "",
        f"[task_memory]\n" + "\n".join(task_lines) if task_lines else "",
        f"[summary]\n{summary}" if summary else "",
        f"[knowledge]\n{knowledge_context}" if knowledge_context else "",
    ]
    payload = "\n\n".join([b for b in blocks if b]).strip()

    return {
        "topic": topic,
        "used_blocks": [
            k
            for k, b in [
                ("profile", profile),
                ("task_memory", task_lines),
                ("summary", summary),
                ("knowledge", knowledge_context),
            ]
            if b
        ],
        "injected_context": _trim(payload, max_chars=max_chars),
    }


def advanced_memory_compression(previous_summary: str, history: list[tuple[str, str]], max_chars: int = 1400) -> str:
    # Hierarchical compression: keep previous summary + key recent turns.
    tail = history[-6:]
    lines = []
    for who, text in tail:
        lines.append(f"{who}: {text}")

    merged = "\n".join([x for x in [previous_summary.strip(), "\n".join(lines)] if x]).strip()
    merged = re.sub(r"\s+", " ", merged)
    if len(merged) <= max_chars:
        return merged

    head_budget = int(max_chars * 0.55)
    tail_budget = max_chars - head_budget - 5
    return merged[:head_budget].rstrip() + " ... " + merged[-tail_budget:].lstrip()


def sandbox_simulate_action(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    action_l = (action or "").strip().lower()
    if action_l == "api_call":
        return {
            "simulated": True,
            "action": action_l,
            "result": {
                "status_code": 200,
                "latency_ms": 180,
                "preview": f"Would call URL: {payload.get('url', '')}",
            },
        }
    if action_l == "workflow_run":
        steps = payload.get("steps", [])
        return {
            "simulated": True,
            "action": action_l,
            "result": {
                "steps": len(steps) if isinstance(steps, list) else 0,
                "estimated_total_ms": 250 + (len(steps) if isinstance(steps, list) else 0) * 120,
            },
        }
    if action_l == "code_exec":
        code = str(payload.get("code", ""))
        return {
            "simulated": True,
            "action": action_l,
            "result": {
                "lint": "ok" if code else "empty",
                "risk": "low" if "os.system(" not in code else "medium",
            },
        }
    return {
        "simulated": True,
        "action": action_l,
        "result": {
            "note": "unknown action simulated with no side effects",
        },
    }


def apply_advanced_guardrails(user_text: str, answer: str, in_domain: bool, verifier_issues: list[str]) -> dict[str, Any]:
    filtered = " ".join((answer or "").split())
    flags: list[str] = []

    if not in_domain:
        flags.append("domain_filter")
    if verifier_issues:
        flags.append("quality_filter")
    if len(filtered) < 8:
        flags.append("coherence_filter")
    if any(x in filtered.lower() for x in ["je pense peut-etre", "inconnu", "??"]):
        flags.append("uncertainty_filter")

    blocked = bool(flags and ("domain_filter" in flags or "coherence_filter" in flags))
    if blocked:
        filtered = "Je reste sur data, IA et automatisation. Precise ton objectif technique, les entrees et le resultat attendu."

    return {
        "answer": filtered,
        "blocked": blocked,
        "flags": flags,
    }


def compute_internal_scores(answer: str, in_domain: bool, verifier_issues: list[str], tools_selected: list[str]) -> dict[str, float]:
    length = max(1, len((answer or "").split()))
    uncertainty = 0.15 if length > 6 else 0.45
    uncertainty += min(0.35, 0.1 * len(verifier_issues))
    uncertainty = max(0.0, min(1.0, uncertainty))

    coherence = 1.0 - min(0.7, len(verifier_issues) * 0.2)
    if len(answer or "") < 16:
        coherence -= 0.2
    coherence = max(0.0, min(1.0, coherence))

    confidence = 0.85 if in_domain else 0.45
    confidence -= min(0.4, 0.12 * len(verifier_issues))
    confidence += min(0.1, 0.02 * len(tools_selected))
    confidence = max(0.0, min(1.0, confidence))

    quality = (coherence * 0.45) + (confidence * 0.55)
    risk = 1.0 - ((confidence * 0.55) + (coherence * 0.45))

    return {
        "confidence": round(confidence, 4),
        "coherence": round(coherence, 4),
        "quality": round(max(0.0, min(1.0, quality)), 4),
        "risk": round(max(0.0, min(1.0, risk)), 4),
        "uncertainty": round(uncertainty, 4),
    }


def write_internal_metrics(event: dict[str, Any]) -> None:
    INTERNAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INTERNAL_METRICS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": _now_iso(), **event}, ensure_ascii=False) + "\n")


def build_distillation_plan() -> dict[str, Any]:
    return {
        "objective": "distill elibot style and workflows into smaller model",
        "dataset_sources": [
            "data/logs/elibot_chat_events.jsonl",
            "data/logs/elibot_audit.jsonl",
            "data/processed/chatbot_train_fr_signature_v2_domain.csv",
        ],
        "steps": [
            "collect high-quality prompt/response pairs",
            "filter by internal quality score >= 0.75",
            "train student model with supervised fine-tuning",
            "evaluate vs teacher on domain benchmarks",
        ],
    }
