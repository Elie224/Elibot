import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def _load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _build_idf(docs_tokens: list[list[str]]) -> dict[str, float]:
    n = len(docs_tokens)
    df: Counter[str] = Counter()
    for toks in docs_tokens:
        df.update(set(toks))

    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((1 + n) / (1 + freq)) + 1.0
    return idf


def _score(query_toks: list[str], doc_toks: list[str], idf: dict[str, float]) -> float:
    if not query_toks or not doc_toks:
        return 0.0
    tf = Counter(doc_toks)
    score = 0.0
    for term in query_toks:
        score += tf.get(term, 0) * idf.get(term, 0.0)

    overlap = len(set(query_toks) & set(doc_toks)) / max(1, len(set(query_toks)))
    return score + 2.0 * overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query local web knowledge chunks with lightweight lexical retrieval")
    parser.add_argument("--index-jsonl", default="data/web/web_chunks.jsonl")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-json", default="reports/web_retrieval_latest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_path = Path(args.index_jsonl)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_chunks(index_path)
    docs_tokens = [_tokens(r.get("text", "")) for r in rows]
    idf = _build_idf(docs_tokens)
    q_toks = _tokens(args.query)

    ranked = []
    for row, d_toks in zip(rows, docs_tokens):
        s = _score(q_toks, d_toks, idf)
        if s <= 0:
            continue
        ranked.append(
            {
                "score": round(float(s), 6),
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "chunk_index": row.get("chunk_index", 0),
                "text": row.get("text", ""),
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    topk = ranked[: max(1, args.top_k)]

    payload = {
        "generated_at": _now_iso(),
        "query": args.query,
        "index_jsonl": str(index_path),
        "hits": len(topk),
        "results": topk,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"status": "ok", "hits": len(topk), "out_json": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
