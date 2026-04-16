from __future__ import annotations

import re


MAX_BULLETS = 8


def _clean(text: str) -> str:
    text = " ".join((text or "").split())
    return text.strip()


def _shorten(text: str, limit: int = 180) -> str:
    value = _clean(text)
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def build_dynamic_summary(
    history: list[tuple[str, str]],
    profile: dict[str, str] | None = None,
    previous_summary: str = "",
    max_chars: int = 1400,
) -> str:
    """Build an extractive summary for long conversations.

    The summary keeps durable user facts and recent technical intent so the model
    stays coherent when history is compressed.
    """
    profile = profile or {}
    lines: list[str] = []

    if previous_summary:
        lines.append("Resume precedent:")
        lines.append(_shorten(previous_summary, 260))

    if profile:
        fields = []
        for key in ("prenom", "ville", "sport", "objectif_minutes"):
            if key in profile:
                fields.append(f"{key}={profile[key]}")
        if fields:
            lines.append("Profil utilisateur: " + ", ".join(fields))

    # Keep user intents and last assistant commitments.
    bullets: list[str] = []
    for role, text in history:
        t = _clean(text)
        if not t:
            continue

        if role == "Utilisateur":
            if re.search(r"\b(pipeline|api|python|sql|pandas|workflow|dataset|modele|automatisation)\b", t.lower()):
                bullets.append("Demande technique: " + _shorten(t, 170))
            elif re.search(r"\b(je veux|objectif|besoin|aide|corrige|explique)\b", t.lower()):
                bullets.append("Intention utilisateur: " + _shorten(t, 170))
        else:
            if re.search(r"\b(plan|etape|solution|je peux|voici|utilise|pipeline)\b", t.lower()):
                bullets.append("Engagement assistant: " + _shorten(t, 170))

    # Ensure the very last exchange is preserved.
    if history:
        last_user = None
        last_assistant = None
        for role, text in reversed(history):
            if role == "Assistant" and last_assistant is None:
                last_assistant = _shorten(text, 170)
            elif role == "Utilisateur" and last_user is None:
                last_user = _shorten(text, 170)
            if last_user and last_assistant:
                break

        if last_user:
            bullets.append("Derniere demande: " + last_user)
        if last_assistant:
            bullets.append("Derniere reponse: " + last_assistant)

    # Deduplicate while preserving order.
    seen = set()
    uniq_bullets = []
    for b in bullets:
        k = b.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq_bullets.append(b)

    if uniq_bullets:
        lines.append("Points cles:")
        for b in uniq_bullets[:MAX_BULLETS]:
            lines.append("- " + b)

    summary = "\n".join(lines).strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return summary


def format_summary_context(summary: str) -> str:
    summary = _clean(summary)
    if not summary:
        return ""
    return "Resume conversation:\n" + summary
