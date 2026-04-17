from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


IN_DOMAIN_KEYWORDS = {
    "ia",
    "ml",
    "ai",
    "machine learning",
    "python",
    "sql",
    "api",
    "workflow",
    "automatisation",
    "pipeline",
    "dataset",
    "classification",
    "regression",
    "debug",
    "lora",
    "n8n",
}

OFF_DOMAIN_KEYWORDS = {
    "politique",
    "religion",
    "psychologie",
    "sante",
    "santé",
    "sport",
    "divertissement",
    "actualite",
    "actualité",
    "relation",
}

TOXIC_PATTERNS = [
    r"\bidiot\b",
    r"\bcon\b",
    r"\bfdp\b",
    r"\bmerde\b",
    r"\bferme ta gueule\b",
]

EMOTIONAL_PATTERNS = [
    r"\bje me sens\b",
    r"\bje suis triste\b",
    r"\bje suis deprime\b",
    r"\bje suis déprimé\b",
    r"\bje t'aime\b",
    r"\bje t aime\b",
]

HESITATION_PATTERNS = [
    r"\bje ne suis pas sur\b",
    r"\bje ne suis pas sûr\b",
    r"\bpeut[- ]etre\b",
    r"\bpeut[- ]être\b",
    r"\bje crois que\b",
]

LEGACY_LIMITATION_PATTERNS = [
    r"\bje ne peux pas\b",
    r"\bje ne peux rien executer\b",
    r"\bje ne peux rien exécuter\b",
    r"\bje ne peux pas envoyer\b",
    r"\bje ne peux pas automatiser\b",
]

LOW_SIGNAL_RESPONSES = {
    "ok",
    "oui",
    "non",
    "voici",
    "bien sur",
    "bien sûr",
    "d'accord",
    "daccord",
}


@dataclass
class PipelineStats:
    conversations_ingested: int = 0
    messages_ingested: int = 0
    conversations_scored: int = 0
    conversations_filtered: int = 0
    conversations_selected: int = 0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_any_keyword(text: str, keywords: set[str]) -> bool:
    value = _norm(text)
    return any(k in value for k in keywords)


def _matches_any(text: str, patterns: list[str]) -> bool:
    value = _norm(text)
    return any(re.search(p, value) for p in patterns)


def _infer_structure_type(text: str) -> str:
    value = (text or "").strip()
    stripped = value.lstrip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return "json"
    if "```" in value:
        return "code"
    if "1)" in value or "2)" in value or "\n- " in value:
        return "steps"
    return "plain"


def _infer_intent(user_text: str, assistant_text: str) -> str:
    value = _norm(f"{user_text} {assistant_text}")
    if "resume" in value or "résume" in value or "summary" in value:
        return "summary"
    if "workflow" in value or "json" in value or "action" in value:
        return "workflow"
    if "erreur" in value or "debug" in value or "traceback" in value:
        return "code_help"
    if "plan" in value or "etape" in value or "étape" in value:
        return "planification"
    return "question"


def _infer_tags(user_text: str, assistant_text: str) -> list[str]:
    value = _norm(f"{user_text} {assistant_text}")
    tags: list[str] = []
    for token in ["ia", "ml", "python", "sql", "api", "workflow", "automatisation", "debug", "dataset"]:
        if token in value:
            tags.append(token)
    if not tags:
        tags.append("general")
    return sorted(set(tags))


def _score_message(role: str, content: str, is_in_domain: bool, is_technical: bool, has_error: bool) -> float:
    value = _norm(content)
    words = [w for w in value.split(" ") if w]

    score = 1.0
    if not is_in_domain:
        score -= 0.45
    if not is_technical:
        score -= 0.20
    if len(words) < 4:
        score -= 0.25
    if _matches_any(value, TOXIC_PATTERNS):
        score -= 0.70
    if _matches_any(value, EMOTIONAL_PATTERNS):
        score -= 0.50
    if has_error:
        score -= 0.30

    if role == "assistant":
        if _matches_any(value, HESITATION_PATTERNS):
            score -= 0.25
        if _matches_any(value, LEGACY_LIMITATION_PATTERNS):
            score -= 0.35
        if value.strip(" .,!?:;\"'") in LOW_SIGNAL_RESPONSES:
            score -= 0.40

    return max(0.0, min(1.0, score))


def _conversation_id_from_event(event: dict[str, Any], line_index: int) -> str:
    for key in ["conversation_id", "session_id", "trace_id", "request_id"]:
        value = str(event.get(key) or "").strip()
        if value:
            return value
    return f"log_conv_{line_index}"


def _conversation_id_from_csv_row(source_name: str, row_index: int) -> str:
    return f"csv_{source_name}_{row_index}"


def _merge_messages_dedup(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for msg in existing:
        mid = str(msg.get("id") or "")
        if not mid or mid in seen_ids:
            continue
        seen_ids.add(mid)
        merged.append(msg)

    added = 0
    for msg in incoming:
        mid = str(msg.get("id") or "")
        if not mid or mid in seen_ids:
            continue
        seen_ids.add(mid)
        merged.append(msg)
        added += 1

    merged.sort(key=lambda m: str(m.get("created_at") or ""))
    return merged, added


def _connect(uri: str, db_name: str) -> Database:
    client = MongoClient(uri)
    return client[db_name]


def _get_collections(db: Database, conversation_collection_name: str) -> tuple[Collection, Collection]:
    return db[conversation_collection_name], db["_migrations"]


def apply_migrations(db: Database, conversation_collection_name: str) -> dict[str, Any]:
    conversations, migrations = _get_collections(db, conversation_collection_name)

    migration_steps: list[tuple[str, str, Any]] = [
        (
            "001_create_indexes",
            "create_indexes",
            lambda: [
                conversations.create_index("status", name="idx_status"),
                conversations.create_index("quality_score", name="idx_quality_score"),
                conversations.create_index("tags", name="idx_tags"),
                conversations.create_index("ended_at", name="idx_ended_at"),
                conversations.create_index("messages.role", name="idx_messages_role"),
            ],
        ),
        (
            "002_backfill_status",
            "backfill_status",
            lambda: conversations.update_many(
                {"status": {"$exists": False}},
                {"$set": {"status": "raw", "quality_score": 0.0}},
            ).modified_count,
        ),
        (
            "003_backfill_message_fields",
            "backfill_message_fields",
            lambda: conversations.update_many(
                {},
                {
                    "$set": {
                        "messages.$[].is_technical": False,
                        "messages.$[].is_in_domain": False,
                        "messages.$[].has_error": False,
                        "messages.$[].quality_score": 0.0,
                        "messages.$[].intent": "question",
                        "messages.$[].structure_type": "plain",
                    }
                },
            ).modified_count,
        ),
    ]

    applied: list[str] = []
    skipped: list[str] = []

    for migration_id, migration_name, operation in migration_steps:
        exists = migrations.find_one({"_id": migration_id})
        if exists:
            skipped.append(migration_id)
            continue

        result = operation()
        migrations.insert_one(
            {
                "_id": migration_id,
                "name": migration_name,
                "applied_at": _now_iso(),
                "result": str(result),
            }
        )
        applied.append(migration_id)

    return {
        "applied": applied,
        "skipped": skipped,
        "collection": conversation_collection_name,
        "db": db.name,
    }


def ingest_jsonl(db: Database, conversation_collection_name: str, log_path: Path, max_events: int | None = None) -> PipelineStats:
    conversations, _ = _get_collections(db, conversation_collection_name)
    if not log_path.exists():
        return PipelineStats()

    stats = PipelineStats()

    with log_path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f, start=1):
            if max_events is not None and line_index > max_events:
                break

            raw = line.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            user_text = str(event.get("user_text") or "").strip()
            assistant_text = str(event.get("assistant_text") or "").strip()
            if not user_text or not assistant_text:
                continue

            conv_id = _conversation_id_from_event(event, line_index)
            ts = str(event.get("timestamp") or event.get("created_at") or _now_iso())
            tags = event.get("tags") if isinstance(event.get("tags"), list) else _infer_tags(user_text, assistant_text)
            topic = str(event.get("topic") or (tags[0] if tags else "ia_data_automation"))
            user_id = str(event.get("user_id") or "unknown")

            messages = []
            for msg_index, (role, content) in enumerate((("user", user_text), ("assistant", assistant_text))):
                is_in_domain = bool(_contains_any_keyword(content, IN_DOMAIN_KEYWORDS) and not _contains_any_keyword(content, OFF_DOMAIN_KEYWORDS))
                is_technical = bool(_contains_any_keyword(content, IN_DOMAIN_KEYWORDS))
                has_error = bool("traceback" in _norm(content) or "exception" in _norm(content) or "error" in _norm(content))
                quality_score = _score_message(role, content, is_in_domain, is_technical, has_error)
                messages.append(
                    {
                        "id": f"{conv_id}_{line_index}_{role}_{msg_index}",
                        "role": role,
                        "content": content,
                        "created_at": ts,
                        "is_technical": is_technical,
                        "is_in_domain": is_in_domain,
                        "has_error": has_error,
                        "quality_score": quality_score,
                        "intent": _infer_intent(user_text, assistant_text),
                        "structure_type": _infer_structure_type(content),
                    }
                )
            existing_doc = conversations.find_one({"_id": conv_id}, {"messages": 1, "started_at": 1}) or {}
            existing_messages = existing_doc.get("messages") or []
            merged_messages, added_count = _merge_messages_dedup(existing_messages, messages)

            started_at = str(existing_doc.get("started_at") or ts)
            stats.messages_ingested += added_count

            conversations.update_one(
                {"_id": conv_id},
                {
                    "$set": {
                        "user_id": user_id,
                        "started_at": started_at,
                        "ended_at": ts,
                        "topic": topic,
                        "tags": tags,
                        "quality_score": 0.0,
                        "status": "raw",
                        "messages": merged_messages,
                        "updated_at": _now_iso(),
                    },
                    "$setOnInsert": {"created_at": _now_iso()},
                },
                upsert=True,
            )
            stats.conversations_ingested += 1

    return stats


def ingest_seed_csvs(db: Database, conversation_collection_name: str, csv_paths: list[Path], max_rows_per_file: int) -> PipelineStats:
    conversations, _ = _get_collections(db, conversation_collection_name)
    stats = PipelineStats()

    for csv_path in csv_paths:
        if not csv_path.exists():
            continue

        source_name = csv_path.stem
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                if i > max_rows_per_file:
                    break

                user_text = str(row.get("instruction") or row.get("prompt_fr") or row.get("prompt_en") or "").strip()
                assistant_text = str(row.get("response") or row.get("response_fr") or row.get("response_en") or "").strip()
                history = str(row.get("history") or "").strip()
                if not user_text or not assistant_text:
                    continue

                conv_id = _conversation_id_from_csv_row(source_name, i)
                ts = _now_iso()
                tags = _infer_tags(user_text, assistant_text)
                topic = tags[0] if tags else "ia_data_automation"

                messages = []
                for msg_index, (role, content) in enumerate((("user", user_text), ("assistant", assistant_text))):
                    is_in_domain = bool(_contains_any_keyword(content, IN_DOMAIN_KEYWORDS) and not _contains_any_keyword(content, OFF_DOMAIN_KEYWORDS))
                    is_technical = bool(_contains_any_keyword(content, IN_DOMAIN_KEYWORDS))
                    has_error = bool("traceback" in _norm(content) or "exception" in _norm(content) or "error" in _norm(content))
                    quality_score = _score_message(role, content, is_in_domain, is_technical, has_error)
                    messages.append(
                        {
                            "id": f"{conv_id}_{role}_{msg_index}",
                            "role": role,
                            "content": content,
                            "created_at": ts,
                            "is_technical": is_technical,
                            "is_in_domain": is_in_domain,
                            "has_error": has_error,
                            "quality_score": quality_score,
                            "intent": _infer_intent(user_text, assistant_text),
                            "structure_type": _infer_structure_type(content),
                        }
                    )

                if history:
                    messages.insert(
                        0,
                        {
                            "id": f"{conv_id}_history_0",
                            "role": "user",
                            "content": history,
                            "created_at": ts,
                            "is_technical": True,
                            "is_in_domain": True,
                            "has_error": False,
                            "quality_score": 0.75,
                            "intent": "summary",
                            "structure_type": "plain",
                        },
                    )

                existing_doc = conversations.find_one({"_id": conv_id}, {"messages": 1, "started_at": 1}) or {}
                existing_messages = existing_doc.get("messages") or []
                merged_messages, added_count = _merge_messages_dedup(existing_messages, messages)
                started_at = str(existing_doc.get("started_at") or ts)
                stats.messages_ingested += added_count

                conversations.update_one(
                    {"_id": conv_id},
                    {
                        "$set": {
                            "user_id": "seed_csv",
                            "started_at": started_at,
                            "ended_at": ts,
                            "topic": topic,
                            "tags": tags,
                            "quality_score": 0.0,
                            "status": "raw",
                            "messages": merged_messages,
                            "updated_at": _now_iso(),
                        },
                        "$setOnInsert": {"created_at": _now_iso()},
                    },
                    upsert=True,
                )
                stats.conversations_ingested += 1

    return stats


def score_and_filter_conversations(db: Database, conversation_collection_name: str, min_conversation_score: float) -> PipelineStats:
    conversations, _ = _get_collections(db, conversation_collection_name)
    stats = PipelineStats()

    cursor = conversations.find({}, {"messages": 1})
    for doc in cursor:
        messages = doc.get("messages") or []
        if not messages:
            continue

        assistant_values = [_norm(str(m.get("content") or "")) for m in messages if str(m.get("role") or "") == "assistant"]
        contradiction_penalty = 0.05 if len(set(assistant_values)) > 1 and len(assistant_values) > 1 else 0.0

        avg = sum(float(m.get("quality_score") or 0.0) for m in messages) / len(messages)
        in_domain_ratio = sum(1 for m in messages if bool(m.get("is_in_domain"))) / len(messages)
        has_error_ratio = sum(1 for m in messages if bool(m.get("has_error"))) / len(messages)

        conv_score = avg * 0.75 + in_domain_ratio * 0.20 + (1.0 - has_error_ratio) * 0.05
        conv_score = max(0.0, min(1.0, conv_score - contradiction_penalty))
        status = "filtered" if conv_score >= min_conversation_score else "raw"

        conversations.update_one(
            {"_id": doc["_id"]},
            {"$set": {"quality_score": conv_score, "status": status, "updated_at": _now_iso()}},
        )
        stats.conversations_scored += 1
        if status == "filtered":
            stats.conversations_filtered += 1

    return stats


def select_best_conversations(db: Database, conversation_collection_name: str, limit: int, min_score: float) -> int:
    conversations, _ = _get_collections(db, conversation_collection_name)
    selected = list(
        conversations.find(
            {"status": "filtered", "quality_score": {"$gte": min_score}},
            {"_id": 1},
        )
        .sort([("quality_score", -1), ("ended_at", -1)])
        .limit(limit)
    )

    ids = [x["_id"] for x in selected]
    if not ids:
        return 0

    conversations.update_many({"status": "selected"}, {"$set": {"status": "filtered", "updated_at": _now_iso()}})
    conversations.update_many({"_id": {"$in": ids}}, {"$set": {"status": "selected", "updated_at": _now_iso()}})
    return len(ids)


def _build_training_rows_from_selected(db: Database, conversation_collection_name: str) -> list[dict[str, str]]:
    conversations, _ = _get_collections(db, conversation_collection_name)
    out: list[dict[str, str]] = []

    cursor = conversations.find({"status": "selected"}, {"messages": 1}).sort("_id", 1)
    for doc in cursor:
        conv_id = str(doc["_id"])
        messages = sorted(doc.get("messages") or [], key=lambda m: str(m.get("created_at") or ""))

        history_parts: list[str] = []
        current_user: str | None = None

        for item in messages:
            role = str(item.get("role") or "")
            content = str(item.get("content") or "").strip()
            intent = str(item.get("intent") or "question")
            structure = str(item.get("structure_type") or "plain")

            if not content:
                continue

            if role == "user":
                current_user = content
                history_parts.append(f"Utilisateur: {content}")
                continue

            if role == "assistant" and current_user:
                out.append(
                    {
                        "conversation_id": conv_id,
                        "instruction": current_user,
                        "response": content,
                        "history": " ||| ".join(history_parts[:-1])[:1400],
                        "intent": intent,
                        "structure_type": structure,
                        "source": "continual_db_selected",
                    }
                )
                history_parts.append(f"Assistant: {content}")
                current_user = None

    return out


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_selected_datasets(db: Database, conversation_collection_name: str, out_dir: Path) -> dict[str, Any]:
    pairs = _build_training_rows_from_selected(db, conversation_collection_name)

    core_qa: list[dict[str, str]] = []
    agent_actions: list[dict[str, str]] = []
    context_memory: list[dict[str, str]] = []
    style_signature: list[dict[str, str]] = []
    csv_rows: list[dict[str, str]] = []

    for p in pairs:
        instruction = p["instruction"]
        response = p["response"]
        history = p["history"]
        intent = p["intent"]
        structure = p["structure_type"]

        train_record = {
            "instruction": instruction,
            "input": history,
            "output": response,
        }

        if structure == "json" or intent == "workflow":
            agent_actions.append(train_record)
        elif intent == "summary":
            context_memory.append(train_record)
        elif "ton" in _norm(instruction) or "style" in _norm(instruction):
            style_signature.append(train_record)
        else:
            core_qa.append(train_record)

        csv_rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": history,
                "source": "continual_db_selected",
            }
        )

    _write_jsonl(out_dir / "core_qa.jsonl", core_qa)
    _write_jsonl(out_dir / "agent_actions.jsonl", agent_actions)
    _write_jsonl(out_dir / "context_memory.jsonl", context_memory)
    _write_jsonl(out_dir / "style_signature.jsonl", style_signature)

    csv_path = out_dir / "chatbot_train_fr_continual_selected.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(csv_rows)

    return {
        "selected_pairs": len(pairs),
        "core_qa_rows": len(core_qa),
        "agent_actions_rows": len(agent_actions),
        "context_memory_rows": len(context_memory),
        "style_signature_rows": len(style_signature),
        "csv_rows": len(csv_rows),
        "out_dir": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled continual-learning DB pipeline (MongoDB, FR)")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default="Elibot")
    parser.add_argument("--collection-name", default="mon_chatbot")
    parser.add_argument("--chat-log", default="data/logs/elibot_chat_events.jsonl")
    parser.add_argument(
        "--seed-csvs",
        nargs="*",
        default=[
            "data/processed/chatbot_train_fr_core_intelligence.csv",
            "data/processed/chatbot_train_fr_signature_v2_domain.csv",
            "data/processed/chatbot_train_fr_agent_actions_tools.csv",
            "data/processed/chatbot_train_fr_memory_context_summary.csv",
        ],
    )
    parser.add_argument("--max-seed-rows-per-file", type=int, default=3000)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--min-conversation-score", type=float, default=0.70)
    parser.add_argument("--selection-limit", type=int, default=5000)
    parser.add_argument("--selection-min-score", type=float, default=0.75)
    parser.add_argument("--out-dir", default="data/processed/continual_db")
    parser.add_argument(
        "--mode",
        choices=["migrate", "ingest", "score", "select", "export", "run-all"],
        default="run-all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    log_path = Path(args.chat_log)
    seed_csv_paths = [Path(p) for p in args.seed_csvs]
    max_events = args.max_events if args.max_events > 0 else None

    db = _connect(args.mongo_uri, args.db_name)

    report: dict[str, Any] = {
        "mode": args.mode,
        "db_name": args.db_name,
        "collection_name": args.collection_name,
        "started_at": _now_iso(),
    }

    if args.mode in {"migrate", "run-all"}:
        report["migrations"] = apply_migrations(db, args.collection_name)

    if args.mode in {"ingest", "run-all"}:
        ingest_stats = ingest_jsonl(db, args.collection_name, log_path=log_path, max_events=max_events)
        seed_stats = PipelineStats()
        used_seed_csv = False
        if ingest_stats.conversations_ingested == 0:
            seed_stats = ingest_seed_csvs(
                db,
                args.collection_name,
                csv_paths=seed_csv_paths,
                max_rows_per_file=max(1, args.max_seed_rows_per_file),
            )
            used_seed_csv = seed_stats.conversations_ingested > 0

        report["ingest"] = {
            "chat_log_found": log_path.exists(),
            "chat_log": ingest_stats.__dict__,
            "seed_csv_used": used_seed_csv,
            "seed_csv": seed_stats.__dict__,
            "seed_csv_paths": [str(p) for p in seed_csv_paths],
        }

    if args.mode in {"score", "run-all"}:
        score_stats = score_and_filter_conversations(db, args.collection_name, min_conversation_score=args.min_conversation_score)
        report["score"] = score_stats.__dict__

    if args.mode in {"select", "run-all"}:
        selected = select_best_conversations(db, args.collection_name, limit=args.selection_limit, min_score=args.selection_min_score)
        report["selection"] = {
            "selected_conversations": selected,
            "selection_limit": args.selection_limit,
            "selection_min_score": args.selection_min_score,
        }

    if args.mode in {"export", "run-all"}:
        report["export"] = export_selected_datasets(db, args.collection_name, out_dir=out_dir)

    report["finished_at"] = _now_iso()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
