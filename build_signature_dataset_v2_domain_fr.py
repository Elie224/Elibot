import argparse
import csv
import random
import re
from pathlib import Path


TECH_TOPICS = [
    "pipeline machine learning",
    "nettoyage de dataset CSV",
    "optimisation pandas",
    "validation de donnees",
    "API FastAPI",
    "orchestration ETL",
    "feature engineering",
    "evaluation de modele",
    "monitoring de modeles",
    "automatisation Python",
]

OUT_DOMAIN_PROMPTS = [
    "Donne-moi un conseil medical",
    "Parle-moi de politique internationale",
    "Aide-moi en psychologie relationnelle",
    "Donne ton avis sur une election",
    "Fais-moi une prediction astrologique",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Elibot v2 signature dataset focused on data/AI/automation")
    parser.add_argument("--rows", type=int, default=132000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-file", default="data/processed/chatbot_train_fr_signature_v2_domain.csv")
    return parser.parse_args()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _make_technical_pair(topic: str) -> tuple[str, str]:
    instruction = f"Explique un plan clair pour {topic}."
    response = (
        f"Voici une approche professionnelle pour {topic}: "
        "1) Clarifier l'objectif metier et les contraintes. "
        "2) Definir les entrees/sorties et les criteres de qualite. "
        "3) Implementer un prototype Python reproductible. "
        "4) Ajouter tests, logging et gestion d'erreurs. "
        "5) Mesurer les resultats et iterer avec une boucle d'amelioration continue."
    )
    return instruction, response


def _make_debug_pair(topic: str) -> tuple[str, str]:
    instruction = f"J'ai une erreur sur {topic}. Comment debugger proprement ?"
    response = (
        "Utilise cette methode: "
        "1) Reproduire le bug avec un cas minimal. "
        "2) Verifier versions, schema de donnees et types. "
        "3) Ajouter des logs avant/apres l'etape fautive. "
        "4) Ecrire un test qui echoue puis corriger. "
        "5) Verifier la correction sur plusieurs echantillons."
    )
    return instruction, response


def _make_reframe_pair(prompt: str) -> tuple[str, str]:
    response = (
        "Je suis specialise en analyse de donnees, IA appliquee et automatisation. "
        "Je ne traite pas ce sujet. "
        "Si tu veux, je peux t'aider sur un pipeline ML, une API FastAPI, du SQL, pandas ou l'analyse d'un dataset."
    )
    return prompt, response


def _make_clarification_pair(topic: str) -> tuple[str, str]:
    instruction = f"Peux-tu m'aider sur {topic} ?"
    response = (
        "Oui. Pour te proposer une solution precise, donne: "
        "1) le format des donnees en entree, "
        "2) le resultat attendu, "
        "3) les contraintes de temps/volume, "
        "4) un extrait d'erreur ou de code."
    )
    return instruction, response


def _build_history(topic: str) -> str:
    turns = [
        "Utilisateur: Je veux une solution technique fiable.",
        "Assistant: D'accord, je vais proposer une approche structuree.",
        f"Utilisateur: Le sujet est {topic}.",
        "Assistant: Parfait, je reste dans le cadre data/IA/automatisation.",
    ]
    return " ||| ".join(turns)


def make_rows(n: int, seed: int) -> list[dict]:
    random.seed(seed)
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    generators = ["technical", "debug", "clarify", "reframe"]
    tones = ["professionnel", "concis", "structure"]

    for case_id in range(1, n + 1):
        kind = random.choice(generators)
        topic = random.choice(TECH_TOPICS)
        tone = random.choice(tones)

        if kind == "technical":
            instruction, response = _make_technical_pair(topic)
        elif kind == "debug":
            instruction, response = _make_debug_pair(topic)
        elif kind == "clarify":
            instruction, response = _make_clarification_pair(topic)
        else:
            instruction, response = _make_reframe_pair(random.choice(OUT_DOMAIN_PROMPTS))

        instruction = f"Cas {case_id} [{tone}] {instruction}"
        response = f"Ton: {tone}. {response}"

        if len(instruction) < 20 or len(response) < 40:
            continue

        key = (_norm(instruction), _norm(response))
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": _build_history(topic),
                "source": "signature_v2_domain",
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    rows = make_rows(args.rows, args.seed)

    out = Path(args.out_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print({"rows": len(rows), "out_file": str(out)})


if __name__ == "__main__":
    main()
