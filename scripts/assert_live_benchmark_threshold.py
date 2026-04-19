import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assert live benchmark pass_rate threshold.")
    parser.add_argument("--in-json", default="reports/eval_business_live_latest.json")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--alert-file", default="reports/eval_business_live_alert.txt")
    args = parser.parse_args()

    payload = json.loads(Path(args.in_json).read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    pass_rate = float(summary.get("pass_rate", 0.0))

    status = "OK" if pass_rate >= args.threshold else "ALERT"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"timestamp={ts}",
        f"status={status}",
        f"pass_rate={pass_rate:.4f}",
        f"threshold={args.threshold:.4f}",
        f"cases={summary.get('cases', 0)}",
        f"score_mean={float(summary.get('score_mean', 0.0)):.4f}",
        f"coverage_mean={float(summary.get('coverage_mean', 0.0)):.4f}",
        f"bad_marker_count={int(summary.get('bad_marker_count', 0))}",
    ]

    alert_path = Path(args.alert_file)
    alert_path.parent.mkdir(parents=True, exist_ok=True)
    alert_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"alert_file={str(alert_path).replace('\\', '/')}")
    print(f"status={status}")
    print(f"pass_rate={pass_rate:.4f}")
    print(f"threshold={args.threshold:.4f}")

    if status == "ALERT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
