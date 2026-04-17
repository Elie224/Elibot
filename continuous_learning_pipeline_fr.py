import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any


GENERALIST_BANNED_KEYWORDS = [
    "salut",
    "ca va",
    "ça va",
    "blague",
    "humour",
    "roleplay",
    "jeu de role",
    "opinion perso",
    "vie quotidienne",
    "emotion",
    "émotion",
]

OFF_DOMAIN_BANNED_KEYWORDS = [
    "politique",
    "religion",
    "psychologie",
    "sante",
    "santé",
    "relation amoureuse",
    "astrologie",
    "sport",
    "people",
    "divertissement",
]

TOXIC_BANNED_KEYWORDS = [
    "insulte",
    "haine",
    "raciste",
    "sexiste",
    "violent",
    "violence",
    "agression",
]

TOXIC_NEGATION_HINTS = [
    "sans",
    "eviter",
    "éviter",
    "interdit",
    "refuser",
    "ne pas",
    "pas de",
]

SMS_BANNED_PATTERNS = [
    r"\bmdr\b",
    r"\blol\b",
    r"\bpk\b",
    r"\bstp\b",
    r"\bsvp\b",
    r"\bjtm\b",
]

LANGUAGE_NOISE_PATTERNS = [
    r"\bslt\b",
    r"\bbjr\b",
    r"\bcv\b",
    r"\bpkoi\b",
    r"\btt\b",
    r"\bdsl\b",
    r"\btkt\b",
    r"(.)\1{4,}",
    r"[!?]{3,}",
]

SUBJECTIVE_PATTERNS = [
    r"\bje pense que\b",
    r"\ba mon avis\b",
    r"\bà mon avis\b",
    r"\bje ressens\b",
    r"\bc[' ]est triste\b",
    r"\bc[' ]est joyeux\b",
    r"\bc[' ]est frustrant\b",
]

HESITATION_PATTERNS = [
    r"\bje ne suis pas sur\b",
    r"\bje ne suis pas sûr\b",
    r"\bpeut[- ]etre\b",
    r"\bpeut[- ]être\b",
    r"\bje crois que\b",
    r"\bpas certain\b",
]

LEGACY_LIMITATION_PATTERNS = [
    r"\bje ne peux pas\b",
    r"\bje ne peux rien executer\b",
    r"\bje ne peux rien exécuter\b",
    r"\bje ne peux pas executer\b",
    r"\bje ne peux pas exécuter\b",
    r"\bje ne peux pas envoyer\b",
    r"\bje ne peux pas automatiser\b",
]

EXCESSIVE_APOLOGY_PATTERNS = [
    r"\bdesole\b",
    r"\bdésolé\b",
    r"\bpardon\b",
    r"\bje suis vraiment desole\b",
    r"\bje suis vraiment désolé\b",
]

STYLE_INCONSISTENT_PATTERNS = [
    r"\bfranchement\b",
    r"\bcarr[ée]ment\b",
    r"\bgrave\b",
    r"\btruc\b",
    r"\bgenre\b",
    r"\bkif\b",
    r"\bcool\b",
    r"\bsympa\b",
]

OUTDATED_TECH_PATTERNS = [
    r"\bpython\s*2\b",
    r"\btensorflow\s*1\b",
    r"\bsklearn\.cross_validation\b",
    r"\bimp\s+module\b",
    r"\bfrom\s+__future__\s+import\s+print_function\b",
]

HALLUCINATION_RISK_PATTERNS = [
    r"\bapi\s+magique\b",
    r"\boutil\s+secret\b",
    r"\bendpoint\s+fictif\b",
    r"\bworkflow\s+impossible\b",
    r"\b100%\s+garanti\b",
]

LOW_SIGNAL_RESPONSES = {
    "bien sur",
    "bien sûr",
    "voici",
    "ok",
    "d'accord",
    "daccord",
    "oui",
    "non",
}

MAX_RESPONSE_CHARS = 2200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weekly continual-learning dataset bundle for Elibot")
    parser.add_argument("--chat-log", default="data/logs/elibot_chat_events.jsonl")
    parser.add_argument("--audit-log", default="data/logs/elibot_audit.jsonl")
    parser.add_argument("--signature-dataset", default="data/processed/chatbot_train_fr_signature_v2_domain.csv")
    parser.add_argument("--signature-datasets", nargs="*", default=[])
    parser.add_argument("--core-datasets", nargs="*", default=[])
    parser.add_argument("--agent-datasets", nargs="*", default=[])
    parser.add_argument("--memory-datasets", nargs="*", default=[])
    parser.add_argument("--out-feedback", default="data/processed/chatbot_train_fr_feedback_weekly.csv")
    parser.add_argument("--out-bundle", default="data/processed/chatbot_train_fr_weekly_bundle.csv")
    parser.add_argument("--out-report", default="reports/feeding_pipeline_report.json")
    parser.add_argument("--min-response-chars", type=int, default=40)
    parser.add_argument("--min-quality-score", type=float, default=0.65)
    parser.add_argument("--max-feedback-rows", type=int, default=30000)
    parser.add_argument("--max-signature-rows", type=int, default=15000)
    parser.add_argument("--max-core-rows", type=int, default=15000)
    parser.add_argument("--max-agent-rows", type=int, default=12000)
    parser.add_argument("--max-memory-rows", type=int, default=10000)
    parser.add_argument("--bundle-target-rows", type=int, default=10000)
    parser.add_argument("--ratio-core", type=float, default=0.40)
    parser.add_argument("--ratio-signature", type=float, default=0.30)
    parser.add_argument("--ratio-agent", type=float, default=0.20)
    parser.add_argument("--ratio-memory", type=float, default=0.10)
    parser.add_argument(
        "--strict-domain",
        action="store_true",
        help="Filter core/signature/agent rows to keep only technical in-domain examples",
    )
    parser.add_argument(
        "--domain-keywords",
        nargs="*",
        default=[
            "ia",
            "ml",
            "machine learning",
            "python",
            "sql",
            "api",
            "workflow",
            "automatisation",
            "pipeline",
            "dataset",
            "debug",
            "classification",
            "planification",
        ],
    )
    parser.add_argument(
        "--dedupe-bundle",
        action="store_true",
        help="Deduplicate final bundle by instruction/response pair",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                out.append(value)
    return out


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_phrase(normalized_text: str, keyword: str) -> bool:
    phrase = _norm_text(keyword)
    if not phrase:
        return False
    parts = [re.escape(p) for p in phrase.split(" ") if p]
    if not parts:
        return False
    pattern = r"\b" + r"\s+".join(parts) + r"\b"
    return bool(re.search(pattern, normalized_text))


def _contains_any(text: str, keywords: list[str]) -> bool:
    normalized = _norm_text(text)
    return any(_contains_phrase(normalized, k) for k in keywords)


def _is_structured_response(response: str) -> bool:
    value = response.strip()
    if len(value) < 60:
        return True
    if "\n" in value:
        return True
    if any(marker in value for marker in ["1)", "2)", "- ", ":", ";", "{"]):
        return True
    sentences = re.split(r"[.!?]+", value)
    return len([s for s in sentences if s.strip()]) >= 2


def _has_sms_style(text: str) -> bool:
    lowered = _norm_text(text)
    return any(re.search(pattern, lowered) for pattern in SMS_BANNED_PATTERNS)


def _has_language_noise(text: str) -> bool:
    normalized = _norm_text(text)
    if any(re.search(pattern, normalized) for pattern in LANGUAGE_NOISE_PATTERNS):
        return True

    tokens = [t for t in re.split(r"\s+", normalized) if t]
    if len(tokens) >= 8:
        very_short = sum(1 for t in tokens if len(t) <= 2 and t not in {"ia", "ml", "ai", "de", "du", "la", "le", "et"})
        if very_short / len(tokens) > 0.35:
            return True

    return False


def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    normalized = _norm_text(text)
    return any(re.search(p, normalized) for p in patterns)


def _is_low_signal_response(response: str) -> bool:
    normalized = _norm_text(response).strip(" .,!?:;\"'")
    return normalized in LOW_SIGNAL_RESPONSES


def _contains_toxic_content(text: str) -> bool:
    normalized = _norm_text(text)
    for keyword in TOXIC_BANNED_KEYWORDS:
        if not _contains_phrase(normalized, keyword):
            continue

        m = re.search(r"\b" + re.escape(_norm_text(keyword)) + r"\b", normalized)
        if not m:
            continue
        idx = m.start()
        window_start = max(0, idx - 24)
        window_end = min(len(normalized), idx + len(keyword) + 24)
        window = normalized[window_start:window_end]
        if any(hint in window for hint in TOXIC_NEGATION_HINTS):
            continue
        return True
    return False


def _heuristic_quality(event: dict[str, Any], min_response_chars: int) -> float:
    score = 0.0
    assistant_text = (event.get("assistant_text") or "").strip()

    if event.get("in_domain", False):
        score += 0.30
    if not event.get("is_low_quality", False):
        score += 0.25
    if not event.get("corrected_by_verifier", False):
        score += 0.15

    issues = event.get("verifier_issues") or []
    if not issues:
        score += 0.15

    if len(assistant_text) >= min_response_chars:
        score += 0.15

    return max(0.0, min(1.0, score))


def _event_quality(event: dict[str, Any], min_response_chars: int) -> float:
    internal = event.get("internal_scores") or {}
    if isinstance(internal, dict) and "quality" in internal:
        return max(0.0, min(1.0, _safe_float(internal.get("quality"), 0.0)))
    return _heuristic_quality(event, min_response_chars=min_response_chars)


def _extract_feedback_rows(
    events: list[dict[str, Any]],
    min_response_chars: int,
    min_quality_score: float,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    seen = set()

    stats = {
        "seen_events": 0,
        "kept_events": 0,
        "filtered_empty": 0,
        "filtered_short": 0,
        "filtered_domain": 0,
        "filtered_low_quality": 0,
        "filtered_score": 0,
        "filtered_duplicate": 0,
    }

    for event in events:
        stats["seen_events"] += 1
        user_text = (event.get("user_text") or "").strip()
        assistant_text = (event.get("assistant_text") or "").strip()

        if not user_text or not assistant_text:
            stats["filtered_empty"] += 1
            continue

        if len(assistant_text) < min_response_chars:
            stats["filtered_short"] += 1
            continue

        if not event.get("in_domain", False):
            stats["filtered_domain"] += 1
            continue

        if event.get("is_low_quality", False):
            stats["filtered_low_quality"] += 1
            continue

        quality = _event_quality(event, min_response_chars=min_response_chars)
        if quality < min_quality_score:
            stats["filtered_score"] += 1
            continue

        key = (_norm_text(user_text), _norm_text(assistant_text))
        if key in seen:
            stats["filtered_duplicate"] += 1
            continue
        seen.add(key)

        rows.append(
            {
                "instruction": user_text,
                "response": assistant_text,
                "history": (event.get("summary") or "")[:1200],
                "source": f"feedback_q{quality:.2f}",
            }
        )
        stats["kept_events"] += 1

    return rows, stats


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _sample_rows(rows: list[dict[str, str]], max_rows: int, seed: int) -> list[dict[str, str]]:
    if len(rows) <= max_rows:
        return list(rows)
    rnd = random.Random(seed)
    idx = list(range(len(rows)))
    rnd.shuffle(idx)
    picked = [rows[i] for i in idx[:max_rows]]
    return picked


def _normalize_source_name(path: Path) -> str:
    stem = path.stem.strip().lower()
    return stem or "extra_dataset"


def _prepare_dataset_rows(rows: list[dict[str, str]], source_name: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for r in rows:
        instruction = (r.get("instruction") or "").strip()
        response = (r.get("response") or "").strip()
        history = (r.get("history") or "").strip()
        source = (r.get("source") or source_name).strip() or source_name

        if not instruction or not response:
            continue

        out.append(
            {
                "instruction": instruction,
                "response": response,
                "history": history,
                "source": source,
            }
        )
    return out


def _dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for r in rows:
        key = (_norm_text(r["instruction"]), _norm_text(r["response"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)

    return out


def _read_group_rows(paths: list[str], max_rows_per_file: int, seed: int) -> tuple[list[dict[str, str]], dict[str, int]]:
    all_rows: list[dict[str, str]] = []
    per_source: dict[str, int] = {}

    for p in paths:
        path = Path(p)
        if not path.exists():
            continue

        source_name = _normalize_source_name(path)
        rows = _prepare_dataset_rows(_read_csv_rows(path), source_name=source_name)
        rows = _sample_rows(rows, max_rows=max(1, max_rows_per_file), seed=seed)

        all_rows.extend(rows)
        per_source[source_name] = len(rows)

    return all_rows, per_source


def _compute_targets(total_rows: int, ratio_core: float, ratio_signature: float, ratio_agent: float, ratio_memory: float) -> dict[str, int]:
    ratios = {
        "core": max(0.0, ratio_core),
        "signature": max(0.0, ratio_signature),
        "agent": max(0.0, ratio_agent),
        "memory": max(0.0, ratio_memory),
    }

    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0.0:
        ratios = {"core": 0.40, "signature": 0.30, "agent": 0.20, "memory": 0.10}
        ratio_sum = 1.0

    normalized = {k: v / ratio_sum for k, v in ratios.items()}
    targets = {k: int(total_rows * normalized[k]) for k in normalized}

    # Distribute rounding remainder deterministically.
    remainder = total_rows - sum(targets.values())
    for key in ["core", "signature", "agent", "memory"]:
        if remainder <= 0:
            break
        targets[key] += 1
        remainder -= 1

    return targets


def _is_domain_row(row: dict[str, str], keywords: list[str]) -> bool:
    text = " ".join(
        [
            row.get("instruction", ""),
            row.get("response", ""),
            row.get("history", ""),
            row.get("source", ""),
        ]
    ).lower()
    return any(k.strip().lower() in text for k in keywords if k.strip())


def _filter_domain_rows(rows: list[dict[str, str]], keywords: list[str]) -> tuple[list[dict[str, str]], int]:
    if not keywords:
        return rows, 0

    kept = [r for r in rows if _is_domain_row(r, keywords)]
    removed = len(rows) - len(kept)
    return kept, removed


def _drop_contradictions(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = _norm_text(row.get("instruction", ""))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, str]] = []
    removed = 0

    for _, group in grouped.items():
        by_response: dict[str, list[dict[str, str]]] = {}
        for row in group:
            rkey = _norm_text(row.get("response", ""))
            by_response.setdefault(rkey, []).append(row)

        # Keep the dominant response variant for identical instruction keys.
        best_key = max(by_response.keys(), key=lambda k: len(by_response[k]))
        kept_group = by_response[best_key]
        out.extend(kept_group)
        removed += len(group) - len(kept_group)

    return out, removed


def _clean_rows_by_policy(rows: list[dict[str, str]], min_response_chars: int, scope: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    stats = {
        "removed_empty": 0,
        "removed_generalist": 0,
        "removed_off_domain": 0,
        "removed_toxic": 0,
        "removed_sms_style": 0,
        "removed_language_quality": 0,
        "removed_subjective": 0,
        "removed_hesitation": 0,
        "removed_legacy_limitations": 0,
        "removed_excessive_apology": 0,
        "removed_style_inconsistent": 0,
        "removed_obsolete": 0,
        "removed_hallucination_risk": 0,
        "removed_low_signal": 0,
        "removed_too_short": 0,
        "removed_too_long": 0,
        "removed_unstructured": 0,
        "removed_duplicate": 0,
        "removed_contradiction": 0,
    }

    cleaned: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        instruction = (row.get("instruction") or "").strip()
        response = (row.get("response") or "").strip()
        history = (row.get("history") or "").strip()
        joint = f"{instruction} {response} {history}"
        topical_text = f"{instruction} {history}"

        if not instruction or not response:
            stats["removed_empty"] += 1
            continue

        if _contains_any(topical_text, GENERALIST_BANNED_KEYWORDS):
            stats["removed_generalist"] += 1
            continue

        if _contains_any(topical_text, OFF_DOMAIN_BANNED_KEYWORDS):
            stats["removed_off_domain"] += 1
            continue

        # Toxicity is evaluated on user/problem context to avoid dropping
        # safe policy responses that mention prohibited terms.
        if _contains_toxic_content(topical_text):
            stats["removed_toxic"] += 1
            continue

        if _has_sms_style(joint):
            stats["removed_sms_style"] += 1
            continue

        if _has_language_noise(f"{instruction} {history}"):
            stats["removed_language_quality"] += 1
            continue

        if _matches_any_pattern(response, SUBJECTIVE_PATTERNS):
            stats["removed_subjective"] += 1
            continue

        if _matches_any_pattern(response, HESITATION_PATTERNS):
            stats["removed_hesitation"] += 1
            continue

        if _matches_any_pattern(response, LEGACY_LIMITATION_PATTERNS):
            stats["removed_legacy_limitations"] += 1
            continue

        if _matches_any_pattern(response, EXCESSIVE_APOLOGY_PATTERNS):
            stats["removed_excessive_apology"] += 1
            continue

        if _matches_any_pattern(response, STYLE_INCONSISTENT_PATTERNS):
            stats["removed_style_inconsistent"] += 1
            continue

        if _matches_any_pattern(joint, OUTDATED_TECH_PATTERNS):
            stats["removed_obsolete"] += 1
            continue

        if _matches_any_pattern(joint, HALLUCINATION_RISK_PATTERNS):
            stats["removed_hallucination_risk"] += 1
            continue

        if _is_low_signal_response(response):
            stats["removed_low_signal"] += 1
            continue

        if len(response) < min_response_chars:
            stats["removed_too_short"] += 1
            continue

        if len(response) > MAX_RESPONSE_CHARS:
            stats["removed_too_long"] += 1
            continue

        if not _is_structured_response(response):
            stats["removed_unstructured"] += 1
            continue

        key = (_norm_text(instruction), _norm_text(response))
        if key in seen:
            stats["removed_duplicate"] += 1
            continue
        seen.add(key)

        cleaned.append(
            {
                "instruction": instruction,
                "response": response,
                "history": history,
                "source": (row.get("source") or "cleaned").strip() or "cleaned",
            }
        )

    cleaned, contradiction_removed = _drop_contradictions(cleaned)
    stats["removed_contradiction"] = contradiction_removed

    return cleaned, stats


def main() -> None:
    args = parse_args()

    chat_log_path = Path(args.chat_log)
    audit_log_path = Path(args.audit_log)
    out_feedback_path = Path(args.out_feedback)
    out_bundle_path = Path(args.out_bundle)
    out_report_path = Path(args.out_report)

    events = _read_jsonl(chat_log_path)
    audits = _read_jsonl(audit_log_path)

    feedback_rows, feedback_stats = _extract_feedback_rows(
        events=events,
        min_response_chars=max(1, args.min_response_chars),
        min_quality_score=max(0.0, min(1.0, args.min_quality_score)),
    )

    feedback_rows = _sample_rows(feedback_rows, max_rows=max(1, args.max_feedback_rows), seed=args.seed)
    _write_csv(out_feedback_path, feedback_rows)

    signature_paths = list(args.signature_datasets)
    if args.signature_dataset:
        signature_paths.append(args.signature_dataset)
    # Keep insertion order while removing duplicates.
    signature_paths = list(dict.fromkeys(signature_paths))

    core_rows, core_sources = _read_group_rows(paths=args.core_datasets, max_rows_per_file=args.max_core_rows, seed=args.seed)
    signature_rows, signature_sources = _read_group_rows(paths=signature_paths, max_rows_per_file=args.max_signature_rows, seed=args.seed)
    agent_rows, agent_sources = _read_group_rows(paths=args.agent_datasets, max_rows_per_file=args.max_agent_rows, seed=args.seed)
    memory_rows, memory_sources = _read_group_rows(paths=args.memory_datasets, max_rows_per_file=args.max_memory_rows, seed=args.seed)

    core_rows, core_clean_stats = _clean_rows_by_policy(
        core_rows,
        min_response_chars=max(20, args.min_response_chars),
        scope="core",
    )
    signature_rows, signature_clean_stats = _clean_rows_by_policy(
        signature_rows,
        min_response_chars=max(20, args.min_response_chars),
        scope="signature",
    )
    agent_rows, agent_clean_stats = _clean_rows_by_policy(
        agent_rows,
        min_response_chars=max(20, args.min_response_chars),
        scope="agent",
    )
    memory_rows, memory_clean_stats = _clean_rows_by_policy(
        memory_rows,
        min_response_chars=max(20, args.min_response_chars // 2),
        scope="memory",
    )

    domain_filter_removed = {"core": 0, "signature": 0, "agent": 0}
    if args.strict_domain:
        core_rows, domain_filter_removed["core"] = _filter_domain_rows(core_rows, args.domain_keywords)
        signature_rows, domain_filter_removed["signature"] = _filter_domain_rows(signature_rows, args.domain_keywords)
        agent_rows, domain_filter_removed["agent"] = _filter_domain_rows(agent_rows, args.domain_keywords)

    targets = _compute_targets(
        total_rows=max(1, args.bundle_target_rows),
        ratio_core=args.ratio_core,
        ratio_signature=args.ratio_signature,
        ratio_agent=args.ratio_agent,
        ratio_memory=args.ratio_memory,
    )

    selected_core = _sample_rows(core_rows, max_rows=targets["core"], seed=args.seed)
    selected_signature = _sample_rows(signature_rows, max_rows=targets["signature"], seed=args.seed)
    selected_agent = _sample_rows(agent_rows, max_rows=targets["agent"], seed=args.seed)
    selected_memory = _sample_rows(memory_rows, max_rows=targets["memory"], seed=args.seed)

    bundle_rows = selected_core + selected_signature + selected_agent + selected_memory + feedback_rows
    if args.dedupe_bundle:
        bundle_rows = _dedupe_rows(bundle_rows)

    # Keep deterministic ordering with signature first then feedback.
    _write_csv(out_bundle_path, bundle_rows)

    if not args.allow_empty and not bundle_rows:
        raise RuntimeError("No rows generated for training bundle. Use --allow-empty to bypass.")

    report = {
        "inputs": {
            "chat_log": str(chat_log_path),
            "audit_log": str(audit_log_path),
            "core_datasets": list(args.core_datasets),
            "signature_datasets": signature_paths,
            "agent_datasets": list(args.agent_datasets),
            "memory_datasets": list(args.memory_datasets),
        },
        "outputs": {
            "feedback_dataset": str(out_feedback_path),
            "weekly_bundle": str(out_bundle_path),
        },
        "counts": {
            "chat_events": len(events),
            "audit_events": len(audits),
            "feedback_rows": len(feedback_rows),
            "core_rows": len(selected_core),
            "signature_rows": len(selected_signature),
            "agent_rows": len(selected_agent),
            "memory_rows": len(selected_memory),
            "bundle_rows": len(bundle_rows),
        },
        "sources": {
            "core": core_sources,
            "signature": signature_sources,
            "agent": agent_sources,
            "memory": memory_sources,
        },
        "targets": targets,
        "feedback_filter_stats": feedback_stats,
        "domain_filter": {
            "strict_domain": bool(args.strict_domain),
            "keywords": list(args.domain_keywords),
            "removed_rows": domain_filter_removed,
        },
        "cleaning_filter": {
            "core": core_clean_stats,
            "signature": signature_clean_stats,
            "agent": agent_clean_stats,
            "memory": memory_clean_stats,
        },
        "policy": {
            "min_response_chars": args.min_response_chars,
            "min_quality_score": args.min_quality_score,
            "max_feedback_rows": args.max_feedback_rows,
            "max_core_rows": args.max_core_rows,
            "max_signature_rows": args.max_signature_rows,
            "max_agent_rows": args.max_agent_rows,
            "max_memory_rows": args.max_memory_rows,
            "bundle_target_rows": args.bundle_target_rows,
            "ratio_core": args.ratio_core,
            "ratio_signature": args.ratio_signature,
            "ratio_agent": args.ratio_agent,
            "ratio_memory": args.ratio_memory,
            "strict_domain": bool(args.strict_domain),
            "dedupe_bundle": bool(args.dedupe_bundle),
        },
        "training_hint": {
            "weekly": "Fine-tune with weekly_bundle (LoRA/light).",
            "monthly": "Regenerate signature dataset and refresh bundle policy.",
            "quarterly": "Add new modules and evaluation scenarios.",
        },
    }

    out_report_path.parent.mkdir(parents=True, exist_ok=True)
    out_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
