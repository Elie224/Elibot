import argparse
import csv
import random
from pathlib import Path

CONCEPTS = {
    "machine learning": "Le machine learning permet a un modele d'apprendre a partir de donnees pour predire ou classer sans ecrire toutes les regles a la main.",
    "overfitting": "L'overfitting apparait quand le modele memorise trop le train et generalise mal sur des donnees nouvelles.",
    "underfitting": "L'underfitting apparait quand le modele est trop simple et n'apprend pas suffisamment le signal utile.",
    "precision recall": "Precision = qualite des positifs predits; recall = capacite a retrouver les vrais positifs.",
    "f1 score": "Le F1-score est la moyenne harmonique entre precision et recall, utile en classes desequilibrees.",
    "cross validation": "La cross-validation evalue un modele sur plusieurs decoupages pour obtenir une mesure plus robuste.",
    "data leakage": "La data leakage est une fuite d'information cible dans les variables d'entree, ce qui biaise l'evaluation.",
    "baseline": "Une baseline est un modele simple de reference pour verifier qu'un modele plus complexe apporte un gain reel.",
    "feature engineering": "Le feature engineering consiste a creer ou transformer des variables pour rendre le signal plus exploitable.",
    "fastapi model serving": "Pour servir un modele avec FastAPI: endpoint /predict, schemas Pydantic, logs, metriques, health check et versionnage.",
}

QUESTION_VARIANTS = {
    "machine learning": [
        "C'est quoi le machine learning ?",
        "Peux-tu definir le machine learning simplement ?",
        "Ca veut dire quoi machine learning ?",
    ],
    "overfitting": [
        "Explique l'overfitting simplement.",
        "C'est quoi l'overfitting ?",
        "Comment reconnaitre l'overfitting ?",
    ],
    "underfitting": [
        "C'est quoi l'underfitting ?",
        "Explique le sous-apprentissage.",
        "Comment detecter un underfitting ?",
    ],
    "precision recall": [
        "Quelle est la difference entre precision et recall ?",
        "Explique precision recall simplement.",
        "Quand privilegier precision ou recall ?",
    ],
    "f1 score": [
        "C'est quoi le F1-score ?",
        "A quoi sert le F1 score ?",
        "Quand utiliser F1 plutot qu'accuracy ?",
    ],
    "cross validation": [
        "C'est quoi la cross-validation ?",
        "Explique la validation croisee.",
        "A quoi sert le k-fold ?",
    ],
    "data leakage": [
        "C'est quoi la data leakage ?",
        "Explique la fuite de donnees.",
        "Comment eviter la data leakage ?",
    ],
    "baseline": [
        "C'est quoi une baseline en ML ?",
        "A quoi sert un modele baseline ?",
        "Pourquoi commencer par une baseline ?",
    ],
    "feature engineering": [
        "C'est quoi le feature engineering ?",
        "Comment faire un bon feature engineering ?",
        "A quoi sert l'ingenierie de variables ?",
    ],
    "fastapi model serving": [
        "Comment servir un modele ML avec FastAPI ?",
        "Architecture FastAPI pour un modele ML ?",
        "Quelles bonnes pratiques pour model serving FastAPI ?",
    ],
}

SUFFIXES = [
    "Reponds directement.",
    "Version claire et concise.",
    "Sans detour.",
]

CONTEXTS = [
    "dans un projet SaaS",
    "dans un cas e-commerce",
    "dans un contexte bancaire",
    "dans une startup data",
    "dans un projet de prevision",
]

PROFILES = [
    "debutant",
    "intermediaire",
    "junior data scientist",
    "engineer MLOps",
]

OBJECTIVES = [
    "ameliorer la qualite de prediction",
    "reduire les erreurs en production",
    "stabiliser le pipeline",
    "accelerer la mise en production",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build high-precision French concept QA dataset")
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed/chatbot_train_fr_gold_concepts.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rnd = random.Random(args.seed)

    rows = []
    seen = set()

    keys = list(CONCEPTS.keys())
    target_rows = max(1, args.rows)
    max_attempts = target_rows * 20
    attempts = 0

    while len(rows) < target_rows and attempts < max_attempts:
        attempts += 1
        key = rnd.choice(keys)
        q = rnd.choice(QUESTION_VARIANTS[key])
        suffix = rnd.choice(SUFFIXES)
        context = rnd.choice(CONTEXTS)
        profile = rnd.choice(PROFILES)
        objective = rnd.choice(OBJECTIVES)

        instruction = f"{q} {suffix} Contexte: {context}. Profil: {profile}.".strip()
        response = f"{CONCEPTS[key]} (Point cle; objectif: {objective})"

        dedupe_key = (instruction.lower(), response.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: Je veux une reponse technique precise. ||| Assistant: Je reponds de facon concise et exacte.",
                "source": "gold_concepts",
            }
        )

    # Deterministic fallback if dedupe saturation is reached.
    if len(rows) < target_rows:
        idx = 0
        while len(rows) < target_rows:
            key = keys[idx % len(keys)]
            q = QUESTION_VARIANTS[key][idx % len(QUESTION_VARIANTS[key])]
            suffix = SUFFIXES[idx % len(SUFFIXES)]
            context = CONTEXTS[idx % len(CONTEXTS)]
            profile = PROFILES[idx % len(PROFILES)]
            objective = OBJECTIVES[idx % len(OBJECTIVES)]
            instruction = f"{q} {suffix} Contexte: {context}. Profil: {profile}."
            response = f"{CONCEPTS[key]} (Point cle; objectif: {objective})"
            rows.append(
                {
                    "instruction": instruction,
                    "response": response,
                    "history": "Utilisateur: Je veux une reponse technique precise. ||| Assistant: Je reponds de facon concise et exacte.",
                    "source": "gold_concepts",
                }
            )
            idx += 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print({"rows": len(rows), "out": str(out_path)})


if __name__ == "__main__":
    main()
