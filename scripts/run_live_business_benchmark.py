import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from gradio_client import Client


BAD_MARKERS = [
    "navigation privee",
    "extension",
    "mot de passe",
    "incognito",
    "compte bloque",
    "sur le sport",
    "journal de recherche",
    "je ne sais pas",
]


def score_answer(answer: str, keywords: list[str]) -> dict:
    text = (answer or "").lower().strip()
    hits = [k for k in keywords if k.lower() in text]
    coverage = len(hits) / len(keywords) if keywords else 0.0
    has_bad = any(marker in text for marker in BAD_MARKERS)
    length_ok = 1.0 if 80 <= len((answer or "").strip()) <= 1200 else 0.0
    safety = 0.0 if has_bad else 1.0
    final = 0.70 * coverage + 0.20 * safety + 0.10 * length_ok

    return {
        "coverage": round(coverage, 4),
        "hits": len(hits),
        "bad": int(has_bad),
        "score": round(final, 4),
        "passed": int(final >= 0.6),
        "matched": hits,
    }


def run_benchmark(space_url: str, benchmark_path: Path, out_json: Path, out_csv: Path, mode: str) -> dict:
    cases = json.loads(benchmark_path.read_text(encoding="utf-8"))
    client = Client(space_url)

    rows = []
    for case in cases:
        question = case["question"]
        expected_keywords = case["expected_keywords"]

        out = client.predict(question, mode, api_name="/handle_submit")
        conversation = out[0]
        answer = conversation[-1][1] if conversation else ""

        s = score_answer(answer, expected_keywords)
        rows.append(
            {
                "question": question,
                "expected_keywords": expected_keywords,
                "answer": answer,
                **s,
            }
        )

    summary = {
        "cases": len(rows),
        "pass_rate": round(mean(r["passed"] for r in rows), 4),
        "score_mean": round(mean(r["score"] for r in rows), 4),
        "coverage_mean": round(mean(r["coverage"] for r in rows), 4),
        "bad_marker_count": sum(r["bad"] for r in rows),
        "mode": mode,
        "space_url": space_url,
        "benchmark_path": str(benchmark_path).replace("\\", "/"),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "coverage", "hits", "bad", "score", "passed"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in ["question", "coverage", "hits", "bad", "score", "passed"]})

    return {
        "summary": summary,
        "failures": [
            {
                "question": row["question"],
                "score": row["score"],
                "coverage": row["coverage"],
            }
            for row in rows
            if row["passed"] == 0
        ],
        "out_json": str(out_json).replace("\\", "/"),
        "out_csv": str(out_csv).replace("\\", "/"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live business benchmark against HF Space.")
    parser.add_argument("--space-url", default="https://elie224-elibot-chat.hf.space")
    parser.add_argument("--benchmark", default="data/eval/business_benchmark_fr.json")
    parser.add_argument("--out-json", default="reports/eval_business_live_latest.json")
    parser.add_argument("--out-csv", default="reports/eval_business_live_latest.csv")
    parser.add_argument("--mode", choices=["Court", "Expert"], default="Expert")
    args = parser.parse_args()

    result = run_benchmark(
        space_url=args.space_url,
        benchmark_path=Path(args.benchmark),
        out_json=Path(args.out_json),
        out_csv=Path(args.out_csv),
        mode=args.mode,
    )

    print(json.dumps(result["summary"], ensure_ascii=False))
    print(f"saved_json={result['out_json']}")
    print(f"saved_csv={result['out_csv']}")
    print(f"fails={len(result['failures'])}")
    for failure in result["failures"]:
        print(f"- {failure['question']} | score={failure['score']} | coverage={failure['coverage']}")


if __name__ == "__main__":
    main()
