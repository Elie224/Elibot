import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build continual-learning dataset from API chat logs")
    parser.add_argument("--log-file", default="data/logs/elibot_chat_events.jsonl")
    parser.add_argument("--out-file", default="data/processed/chatbot_train_fr_continual.csv")
    parser.add_argument("--min-chars", type=int, default=24)
    return parser.parse_args()


def _is_good_row(event: dict, min_chars: int) -> bool:
    user_text = (event.get("user_text") or "").strip()
    assistant_text = (event.get("assistant_text") or "").strip()

    if not user_text or not assistant_text:
        return False
    if len(assistant_text) < min_chars:
        return False
    if not event.get("in_domain", False):
        return False
    if event.get("is_low_quality", False):
        return False

    return True


def main() -> None:
    args = parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    rows = []
    seen = set()

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not _is_good_row(event, args.min_chars):
                continue

            instruction = event["user_text"].strip()
            response = event["assistant_text"].strip()
            key = (instruction, response)
            if key in seen:
                continue
            seen.add(key)

            rows.append(
                {
                    "instruction": instruction,
                    "response": response,
                    "history": "",
                    "source": "continual_log",
                }
            )

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print({
        "input_log": str(log_path),
        "written_rows": len(rows),
        "output_file": str(out_path),
    })


if __name__ == "__main__":
    main()
