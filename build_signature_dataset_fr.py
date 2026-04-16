import argparse
import csv
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a phase-2 signature dataset from chatbot_train_fr.csv")
    parser.add_argument("--input", default="data/processed/chatbot_train_fr.csv")
    parser.add_argument("--output", default="data/processed/chatbot_train_fr_signature_v1.csv")
    parser.add_argument("--target-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_complex_text(text: str) -> bool:
    words = text.split()
    if len(words) >= 18:
        return True
    punct = sum(text.count(x) for x in [",", ";", ":", "(", ")", "-"])
    return punct >= 2


def score_row(row: dict, freq_starts: Counter) -> float:
    instruction = (row.get("instruction") or "").strip()
    response = (row.get("response") or "").strip()

    if not instruction or not response:
        return -1e9

    score = 0.0

    # Prefer richer and less generic responses.
    resp_words = len(response.split())
    instr_words = len(instruction.split())

    score += min(resp_words, 40) * 0.08
    score += min(instr_words, 30) * 0.05

    if is_complex_text(response):
        score += 1.2
    if is_complex_text(instruction):
        score += 0.8

    if "?" in instruction:
        score += 0.4

    # Penalize very short replies.
    if resp_words <= 8:
        score -= 1.2

    # Penalize extremely frequent response openings.
    start = " ".join(response.lower().split()[:2])
    score -= min(freq_starts.get(start, 0) / 1200.0, 2.0)

    # Light source balancing signal.
    source = (row.get("source") or "").lower()
    if source and "synthetic" in source:
        score -= 0.15

    return score


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    with open(args.input, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    # Strict dedup on exact pair.
    seen = set()
    deduped = []
    for row in rows:
        key = ((row.get("instruction") or "").strip(), (row.get("response") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    starts = Counter()
    for row in deduped:
        response = (row.get("response") or "").strip()
        starts[" ".join(response.lower().split()[:2])] += 1

    scored = []
    for row in deduped:
        scored.append((score_row(row, starts), row))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Keep a high-quality core and add a random tail slice for diversity.
    target = min(args.target_size, len(scored))
    core_size = int(target * 0.8)
    tail_size = target - core_size

    core = [row for _, row in scored[: core_size * 2]]
    random.shuffle(core)
    core = core[:core_size]

    tail_pool = [row for _, row in scored[core_size * 2 :]]
    random.shuffle(tail_pool)
    tail = tail_pool[:tail_size]

    selected = core + tail
    random.shuffle(selected)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(
        {
            "input_rows": len(rows),
            "deduped_rows": len(deduped),
            "target_rows": target,
            "written_rows": len(selected),
            "output": str(out_path),
        }
    )


if __name__ == "__main__":
    main()
