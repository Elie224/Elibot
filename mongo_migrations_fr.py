from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

from pymongo import MongoClient


MIGRATIONS = [
    {
        "id": "001_create_indexes",
        "description": "Create performance indexes for mon_chatbot",
    },
    {
        "id": "002_backfill_status",
        "description": "Backfill missing status/quality_score",
    },
    {
        "id": "003_backfill_message_fields",
        "description": "Backfill message metadata fields",
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_migrations(mongo_uri: str, db_name: str, collection_name: str) -> dict:
    client = MongoClient(mongo_uri)
    db = client[db_name]
    col = db[collection_name]
    migration_col = db["_migrations"]

    applied = []
    skipped = []

    for migration in MIGRATIONS:
        mid = migration["id"]
        if migration_col.find_one({"_id": mid}):
            skipped.append(mid)
            continue

        if mid == "001_create_indexes":
            col.create_index("status", name="idx_status")
            col.create_index("quality_score", name="idx_quality_score")
            col.create_index("tags", name="idx_tags")
            col.create_index("ended_at", name="idx_ended_at")
            col.create_index("messages.role", name="idx_messages_role")
            result = "indexes_created"

        elif mid == "002_backfill_status":
            result = col.update_many(
                {"status": {"$exists": False}},
                {"$set": {"status": "raw", "quality_score": 0.0}},
            ).modified_count

        elif mid == "003_backfill_message_fields":
            result = col.update_many(
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
            ).modified_count

        else:
            result = "unknown"

        migration_col.insert_one(
            {
                "_id": mid,
                "description": migration["description"],
                "applied_at": _now_iso(),
                "result": str(result),
            }
        )
        applied.append(mid)

    return {
        "db_name": db_name,
        "collection_name": collection_name,
        "applied": applied,
        "skipped": skipped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MongoDB migrations for Elibot")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    parser.add_argument("--db-name", default="Elibot")
    parser.add_argument("--collection-name", default="mon_chatbot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_migrations(args.mongo_uri, args.db_name, args.collection_name)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
