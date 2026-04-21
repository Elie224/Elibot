import argparse
import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def _domain(url: str) -> str:
    return (urlparse(url).netloc or "").lower()


def _is_allowed(url: str, allowed_domains: set[str]) -> bool:
    if not allowed_domains:
        return True
    d = _domain(url)
    return any(d == allowed or d.endswith("." + allowed) for allowed in allowed_domains)


def _fetch_html(url: str, timeout: int, user_agent: str) -> str:
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="replace")


def _extract_text(html: str) -> tuple[str, str]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    title = unescape(title_match.group(1).strip()) if title_match else ""

    cleaned = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<style.*?>.*?</style>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<noscript.*?>.*?</noscript>", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return title, cleaned


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[str]:
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    step = max(1, chunk_size - max(0, chunk_overlap))
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start += step
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest approved web pages into local chunked knowledge store")
    parser.add_argument("--sources-file", default="data/web/sources_allowlist.txt")
    parser.add_argument("--allow-domains-file", default="data/web/allowed_domains.txt")
    parser.add_argument("--out-jsonl", default="data/web/web_chunks.jsonl")
    parser.add_argument("--manifest-json", default="reports/web_ingest_manifest.json")
    parser.add_argument("--max-urls", type=int, default=40)
    parser.add_argument("--timeout-seconds", type=int, default=20)
    parser.add_argument("--max-chars-per-page", type=int, default=120000)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--user-agent", default="ElibotWebIngest/1.0 (+https://github.com/Elie224/Elibot)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sources_file = Path(args.sources_file)
    allow_domains_file = Path(args.allow_domains_file)
    out_jsonl = Path(args.out_jsonl)
    manifest_json = Path(args.manifest_json)

    urls = _read_lines(sources_file)[: max(0, args.max_urls)]
    allowed_domains = {d.lower() for d in _read_lines(allow_domains_file)}

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    blocked: list[str] = []
    failed: list[dict] = []

    for url in urls:
        if not _is_allowed(url, allowed_domains):
            blocked.append(url)
            continue

        try:
            html = _fetch_html(url, timeout=args.timeout_seconds, user_agent=args.user_agent)
            title, text = _extract_text(html)
            if args.max_chars_per_page > 0:
                text = text[: args.max_chars_per_page]

            chunks = list(_chunk_text(text, args.chunk_size, args.chunk_overlap))
            for idx, chunk in enumerate(chunks):
                records.append(
                    {
                        "id": f"{_domain(url)}::{idx}",
                        "url": url,
                        "domain": _domain(url),
                        "title": title,
                        "chunk_index": idx,
                        "text": chunk,
                        "text_length": len(chunk),
                        "ingested_at": _now_iso(),
                    }
                )
        except Exception as exc:
            failed.append({"url": url, "error": str(exc)})

    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "generated_at": _now_iso(),
        "sources_file": str(sources_file),
        "allow_domains_file": str(allow_domains_file),
        "out_jsonl": str(out_jsonl),
        "urls_seen": len(urls),
        "urls_blocked": len(blocked),
        "urls_failed": len(failed),
        "chunks_written": len(records),
        "blocked_urls": blocked,
        "failed_urls": failed,
    }
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"status": "ok", "manifest": str(manifest_json), "chunks": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
