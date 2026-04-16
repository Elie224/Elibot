import re


TECH_MARKERS = {
    "data",
    "dataset",
    "csv",
    "json",
    "sql",
    "pandas",
    "numpy",
    "pipeline",
    "api",
    "fastapi",
    "python",
    "modele",
    "model",
    "entrainement",
    "evaluation",
    "automatisation",
    "workflow",
}

OUT_DOMAIN_MARKERS = {
    "politique",
    "election",
    "religion",
    "medecine",
    "medical",
    "diagnostic",
    "amour",
    "relation",
    "astrologie",
    "voyance",
}

BAD_PATTERNS = [
    "je ne peux pas te dire que tu veux dire",
    "je veux dire que je n'ai pas besoin",
    "reheatreheatreheat",
    "synchronoussynchronous",
    "municipal municipal",
]


def _normalize_tokens(text: str) -> list[str]:
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9à-öø-ÿ\s]", " ", lowered)
    return [t for t in cleaned.split() if t]


def is_technical_request(user_text: str) -> bool:
    text = user_text.lower()
    return any(marker in text for marker in TECH_MARKERS)


def detect_quality_issues(user_text: str, answer: str, in_domain: bool = True) -> list[str]:
    text = (answer or "").strip()
    if not text:
        return ["empty"]

    lowered = " ".join(text.lower().split())
    issues: list[str] = []

    if any(p in lowered for p in BAD_PATTERNS):
        issues.append("known_bad_pattern")

    tokens = _normalize_tokens(text)
    if len(tokens) >= 14:
        uniq_ratio = len(set(tokens)) / max(1, len(tokens))
        if uniq_ratio < 0.5:
            issues.append("high_repetition")

    if re.search(r"\b(de\s+te\s+dire\s+que\s+tu|vous\s+avez\s+une\s+idee\s+de)\b", lowered):
        issues.append("unstable_phrase")

    if in_domain and any(k in lowered for k in OUT_DOMAIN_MARKERS):
        issues.append("out_of_domain_content")

    tech_query = is_technical_request(user_text)
    if in_domain and tech_query:
        if len(tokens) < 8:
            issues.append("too_short_for_technical")
        if not any(k in lowered for k in TECH_MARKERS):
            issues.append("missing_technical_signal")

    if len(tokens) > 220:
        issues.append("too_long")

    return issues
