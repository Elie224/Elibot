import re


INTENT_AUTOMATION_KEYWORDS = {
    "workflow",
    "automatisation",
    "automatiser",
    "api",
    "webhook",
    "n8n",
    "zapier",
    "plan d action",
    "orchestration",
    "pipeline",
    "executer",
    "exécuter",
}

INTENT_CODE_KEYWORDS = {
    "code",
    "python",
    "sql",
    "pandas",
    "script",
    "fonction",
    "debug",
    "erreur",
    "bug",
}

INTENT_TECH_KEYWORDS = {
    "data",
    "donnee",
    "dataset",
    "ml",
    "machine learning",
    "modele",
    "modèle",
    "ia",
    "analyse",
    "fastapi",
    "docker",
}

INTENT_OUT_DOMAIN_KEYWORDS = {
    "politique",
    "élection",
    "election",
    "médical",
    "medical",
    "diagnostic",
    "psychologie",
    "amour",
    "relation",
    "astrologie",
}

GREETING_KEYWORDS = {"bonjour", "salut", "hello", "bonsoir", "coucou", "bjr", "slt", "cc"}
GRATITUDE_KEYWORDS = {"merci", "thanks", "thx"}


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(k in text for k in keywords)


def classify_intent(user_text: str) -> dict:
    text = _normalize(user_text)
    if not text:
        return {"intent": "unknown", "confidence": 0.0, "reasons": ["empty"]}

    reasons: list[str] = []

    if text in GREETING_KEYWORDS:
        return {"intent": "greeting", "confidence": 0.99, "reasons": ["matched_greeting"]}

    if text in GRATITUDE_KEYWORDS or text.startswith("merci"):
        return {"intent": "gratitude", "confidence": 0.98, "reasons": ["matched_gratitude"]}

    if _contains_any(text, INTENT_OUT_DOMAIN_KEYWORDS):
        return {"intent": "out_of_domain", "confidence": 0.92, "reasons": ["matched_out_domain_keyword"]}

    automation = _contains_any(text, INTENT_AUTOMATION_KEYWORDS)
    code = _contains_any(text, INTENT_CODE_KEYWORDS)
    tech = _contains_any(text, INTENT_TECH_KEYWORDS)

    if automation:
        reasons.append("matched_automation_keyword")
    if code:
        reasons.append("matched_code_keyword")
    if tech:
        reasons.append("matched_tech_keyword")

    if automation and ("plan" in text or "étape" in text or "etape" in text or "execute" in text or "exécute" in text):
        return {"intent": "automation_request", "confidence": 0.95, "reasons": reasons}

    if automation:
        return {"intent": "automation_request", "confidence": 0.88, "reasons": reasons}

    if code:
        return {"intent": "code_request", "confidence": 0.9, "reasons": reasons}

    if tech or "?" in text:
        return {"intent": "technical_question", "confidence": 0.75 if tech else 0.55, "reasons": reasons or ["question_mark"]}

    return {"intent": "unknown", "confidence": 0.35, "reasons": ["no_strong_signal"]}
