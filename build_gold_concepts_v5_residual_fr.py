import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gold v5 dataset from residual failed benchmark cases")
    parser.add_argument("--benchmark-samples", default="reports/eval_business_weekly_samples.csv")
    parser.add_argument("--rows", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed/chatbot_train_fr_gold_concepts_v5.csv")
    return parser.parse_args()


def load_failed_cases(path: str) -> list[dict[str, list[str] | str]]:
    failed: list[dict[str, list[str] | str]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            passed = int(row.get("passed", "0") or "0")
            if passed == 1:
                continue
            q = (row.get("question") or "").strip()
            kws = [k.strip() for k in (row.get("expected_keywords") or "").split(",") if k.strip()]
            if q and kws:
                failed.append({"question": q, "keywords": kws})
    if not failed:
        raise RuntimeError("No failed benchmark cases found")
    return failed


def build_response(question: str, keywords: list[str]) -> str:
    keywords_text = ", ".join(keywords)
    return (
        f"Definition: Reponse operationnelle pour '{question}' dans un contexte machine learning en production.\n"
        f"Bonnes pratiques: structurer le workflow, valider hors echantillon, et monitorer en continu.\n"
        f"Points cles: {keywords_text}.\n"
        f"Checklist: verifier explicitement {keywords_text}."
    )


def main() -> None:
    args = parse_args()
    rnd = random.Random(args.seed)
    failed_cases = load_failed_cases(args.benchmark_samples)

    styles = [
        "Reponds directement.",
        "Version claire et concise.",
        "Sans detour.",
    ]
    contexts = [
        "Contexte: projet data orienté performance.",
        "Contexte: pipeline ML en production.",
        "Contexte: API et monitoring en exploitation.",
    ]
    profiles = [
        "Profil: engineer MLOps.",
        "Profil: junior data scientist.",
        "Profil: intermediaire.",
    ]

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    target = max(1, args.rows)

    attempts = 0
    max_attempts = target * 50
    while len(rows) < target and attempts < max_attempts:
        attempts += 1
        case = rnd.choice(failed_cases)
        question = str(case["question"])
        keywords = [str(k) for k in case["keywords"]]
        instruction = f"{question} {rnd.choice(styles)} {rnd.choice(contexts)} {rnd.choice(profiles)}"
        response = build_response(question, keywords)
        key = (instruction.lower(), response.lower())
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: Je veux une reponse technique exploitable. ||| Assistant: Je reponds avec structure et mots cles explicites.",
                "source": "gold_concepts_v5",
            }
        )

    i = 0
    while len(rows) < target:
        case = failed_cases[i % len(failed_cases)]
        question = str(case["question"])
        keywords = [str(k) for k in case["keywords"]]
        instruction = f"{question} {styles[i % len(styles)]} {contexts[i % len(contexts)]} {profiles[i % len(profiles)]}"
        rows.append(
            {
                "instruction": instruction,
                "response": build_response(question, keywords),
                "history": "Utilisateur: Je veux une reponse technique exploitable. ||| Assistant: Je reponds avec structure et mots cles explicites.",
                "source": "gold_concepts_v5",
            }
        )
        i += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print({"rows": len(rows), "failed_cases": len(failed_cases), "out": str(out)})


if __name__ == "__main__":
    main()
