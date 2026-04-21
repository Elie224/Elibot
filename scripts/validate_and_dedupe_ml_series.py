import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

DATASETS = [
    Path("data/processed/chatbot_train_fr_ml_200_essentials.csv"),
    Path("data/processed/chatbot_train_fr_ml_150_advanced.csv"),
    Path("data/processed/chatbot_train_fr_ml_120_classical_advanced.csv"),
    Path("data/processed/chatbot_train_fr_ml_120_classical_series4.csv"),
]

REPORT_PATH = Path("reports/ml_series_validation_report.json")


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        # Keep header even if empty.
        fieldnames = ["instruction", "response", "history", "source"]
    else:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    report = {
        "datasets": {},
        "cross_dataset_duplicates": {},
        "actions": [],
    }

    # Track first owner of each normalized question across datasets.
    global_question_owner: Dict[str, str] = {}
    global_question_dupes: Dict[str, List[str]] = {}

    for dataset in DATASETS:
        if not dataset.exists():
            report["datasets"][str(dataset)] = {"exists": False}
            continue

        rows = load_rows(dataset)
        original_count = len(rows)

        cleaned_rows: List[Dict[str, str]] = []
        seen_questions = set()
        seen_qr = set()

        empty_response = 0
        duplicate_question_within = 0
        duplicate_qr_within = 0

        for row in rows:
            instruction = (row.get("instruction") or "").strip()
            response = (row.get("response") or "").strip()

            qn = normalize_text(instruction)
            rn = normalize_text(response)

            if not qn:
                # Drop empty question row.
                continue

            if not rn:
                empty_response += 1
                continue

            qr_key = qn + "|||" + rn
            if qr_key in seen_qr:
                duplicate_qr_within += 1
                continue

            if qn in seen_questions:
                duplicate_question_within += 1
                continue

            seen_qr.add(qr_key)
            seen_questions.add(qn)
            cleaned_rows.append(row)

        # Cross-dataset duplicate question handling:
        # keep first appearance by DATASETS order, remove from later files.
        final_rows: List[Dict[str, str]] = []
        cross_removed = 0
        for row in cleaned_rows:
            qn = normalize_text((row.get("instruction") or "").strip())
            if qn not in global_question_owner:
                global_question_owner[qn] = str(dataset)
                final_rows.append(row)
            else:
                cross_removed += 1
                owner = global_question_owner[qn]
                global_question_dupes.setdefault(qn, [owner])
                global_question_dupes[qn].append(str(dataset))

        # Rewrite file only if changed.
        if len(final_rows) != original_count:
            write_rows(dataset, final_rows)
            report["actions"].append(
                {
                    "dataset": str(dataset),
                    "action": "rewritten",
                    "before": original_count,
                    "after": len(final_rows),
                }
            )

        report["datasets"][str(dataset)] = {
            "exists": True,
            "before_rows": original_count,
            "after_rows": len(final_rows),
            "removed_empty_response": empty_response,
            "removed_duplicate_question_within": duplicate_question_within,
            "removed_duplicate_qr_within": duplicate_qr_within,
            "removed_duplicate_question_cross_dataset": cross_removed,
        }

    # Build compact cross-dataset duplicate summary.
    # Keep only first 30 to keep report readable.
    sample = []
    for idx, (q, owners) in enumerate(global_question_dupes.items()):
        if idx >= 30:
            break
        sample.append({"question_normalized": q, "datasets": owners})

    report["cross_dataset_duplicates"] = {
        "total_duplicate_questions": len(global_question_dupes),
        "sample": sample,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "report": str(REPORT_PATH),
        "actions": report["actions"],
        "cross_dataset_duplicate_questions": len(global_question_dupes),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
