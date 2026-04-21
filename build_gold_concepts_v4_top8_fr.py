import argparse
import csv
import random
from pathlib import Path

CASES = [
    {
        "topic": "classes_desequilibrees",
        "questions": [
            "quelle metrique pour classes desequilibrees ?",
            "metriques pour dataset desequilibre en classification",
            "comment evaluer une classe rare ?",
        ],
        "keywords": ["precision", "recall", "f1", "pr auc"],
        "definition": "Pour classes desequilibrees, il faut privilegier precision, recall, f1 et pr auc plutot que l'accuracy.",
    },
    {
        "topic": "valeurs_manquantes",
        "questions": [
            "comment gerer les valeurs manquantes ?",
            "quelle strategie d'imputation en ML ?",
            "missing values en pipeline: bonne pratique",
        ],
        "keywords": ["imputation", "median", "mode", "pipeline"],
        "definition": "Les valeurs manquantes se traitent par imputation (median ou mode) dans un pipeline reproductible.",
    },
    {
        "topic": "categorical_encoding",
        "questions": [
            "categorical encoding options",
            "comment encoder des variables categorielles ?",
            "one-hot ou target encoding ?",
        ],
        "keywords": ["one-hot", "target encoding", "cardinalite", "fuite"],
        "definition": "Pour variables categorielles, utiliser one-hot ou target encoding selon la cardinalite en evitant la fuite.",
    },
    {
        "topic": "random_forest",
        "questions": [
            "quand utiliser random forest ?",
            "cas d'usage random forest",
            "random forest en pratique",
        ],
        "keywords": ["tabulaire", "baseline", "non-lineaire", "interpretabilite"],
        "definition": "Random forest est adapte au tabulaire comme baseline non-lineaire avec bonne interpretabilite.",
    },
    {
        "topic": "monitoring_prod",
        "questions": [
            "monitoring d'un modele en prod",
            "quels KPI suivre en production ML ?",
            "surveillance d'un modele deploye",
        ],
        "keywords": ["drift", "latence", "erreur", "metriques"],
        "definition": "Le monitoring production suit drift, latence, taux d'erreur et metriques metier.",
    },
    {
        "topic": "data_drift",
        "questions": [
            "qu'est-ce que le data drift ?",
            "comment detecter un drift des donnees ?",
            "derive de distribution des features",
        ],
        "keywords": ["distribution", "features", "surveillance", "reentrainement"],
        "definition": "Le data drift est un changement de distribution des features et impose surveillance puis reentrainement.",
    },
    {
        "topic": "concept_drift",
        "questions": [
            "comment detecter concept drift ?",
            "definition operationnelle du concept drift",
            "evolution cible et derive conceptuelle",
        ],
        "keywords": ["cible", "performance", "fenetre temporelle", "alerte"],
        "definition": "Le concept drift touche la relation avec la cible et se suit via performance par fenetre temporelle avec alerte.",
    },
    {
        "topic": "mlops",
        "questions": [
            "MLOps c'est quoi en pratique ?",
            "comment mettre en place MLOps ?",
            "principes MLOps pour la production",
        ],
        "keywords": ["ci/cd", "versionning", "deploiement", "monitoring", "gouvernance"],
        "definition": "MLOps couvre ci/cd, versionning, deploiement, monitoring et gouvernance du cycle de vie modele.",
    },
]

STYLES = ["Reponds directement.", "Version claire et concise.", "Sans detour."]
CONTEXTS = [
    "Contexte: projet machine learning en production.",
    "Contexte: pipeline data avec API.",
    "Contexte: cas metier orienté qualite de prediction.",
]
PROFILES = [
    "Profil: engineer MLOps.",
    "Profil: junior data scientist.",
    "Profil: intermediaire.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gold v4 dataset for top 8 residual benchmark failures")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed/chatbot_train_fr_gold_concepts_v4.csv")
    return parser.parse_args()


def build_response(question: str, definition: str, keywords: list[str]) -> str:
    return (
        f"Definition: {definition}\n"
        f"Bonnes pratiques: appliquer un pipeline robuste, tester en validation, et monitorer en production.\n"
        f"Points cles: {', '.join(keywords)}.\n"
        f"Application: pour '{question}', donner une reponse operationnelle et orientee metier."
    )


def main() -> None:
    args = parse_args()
    rnd = random.Random(args.seed)

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    target = max(1, args.rows)
    max_attempts = target * 40
    attempts = 0

    while len(rows) < target and attempts < max_attempts:
        attempts += 1
        case = rnd.choice(CASES)
        question = rnd.choice(case["questions"])
        instruction = f"{question} {rnd.choice(STYLES)} {rnd.choice(CONTEXTS)} {rnd.choice(PROFILES)}"
        response = build_response(question, case["definition"], case["keywords"])

        key = (instruction.lower(), response.lower())
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: Je veux une reponse technique exploitable. ||| Assistant: Je reponds en mode expert structure.",
                "source": "gold_concepts_v4",
            }
        )

    i = 0
    while len(rows) < target:
        case = CASES[i % len(CASES)]
        question = case["questions"][i % len(case["questions"])]
        instruction = f"{question} {STYLES[i % len(STYLES)]} {CONTEXTS[i % len(CONTEXTS)]} {PROFILES[i % len(PROFILES)]}"
        rows.append(
            {
                "instruction": instruction,
                "response": build_response(question, case["definition"], case["keywords"]),
                "history": "Utilisateur: Je veux une reponse technique exploitable. ||| Assistant: Je reponds en mode expert structure.",
                "source": "gold_concepts_v4",
            }
        )
        i += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print({"rows": len(rows), "out": str(out)})


if __name__ == "__main__":
    main()
