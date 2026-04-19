import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def build_markdown(payload: dict) -> str:
    summary = payload.get("summary", {})
    rows = payload.get("rows", [])
    failures = [r for r in rows if int(r.get("passed", 0)) == 0]

    lines = []
    lines.append("# Live Business Benchmark Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")
    lines.append("## Key Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Cases | {summary.get('cases', 0)} |")
    lines.append(f"| Pass rate | {summary.get('pass_rate', 0):.4f} |")
    lines.append(f"| Score mean | {summary.get('score_mean', 0):.4f} |")
    lines.append(f"| Coverage mean | {summary.get('coverage_mean', 0):.4f} |")
    lines.append(f"| Bad marker count | {summary.get('bad_marker_count', 0)} |")

    lines.append("")
    lines.append("## Failures")
    lines.append("")
    if not failures:
        lines.append("No failing cases.")
    else:
        lines.append("| Question | Score | Coverage |")
        lines.append("|---|---:|---:|")
        for row in failures:
            q = str(row.get("question", "")).replace("|", "\\|")
            lines.append(f"| {q} | {float(row.get('score', 0)):.4f} | {float(row.get('coverage', 0)):.4f} |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Markdown summary from live benchmark JSON.")
    parser.add_argument("--in-json", default="reports/eval_business_live_latest.json")
    parser.add_argument("--out-md", default="reports/eval_business_live_latest.md")
    args = parser.parse_args()

    in_json = Path(args.in_json)
    out_md = Path(args.out_md)

    payload = json.loads(in_json.read_text(encoding="utf-8"))
    markdown = build_markdown(payload)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")

    print(f"saved_md={str(out_md).replace('\\', '/')}")


if __name__ == "__main__":
    main()
