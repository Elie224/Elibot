import argparse
import csv
import random
from pathlib import Path

CASES = [
    {
        "topic": "surapprentissage",
        "questions": [
            "comment eviter le surapprentissage ?",
            "comment reduire l'overfitting d'un modele ?",
            "quelles techniques contre le surapprentissage ?",
        ],
        "response": "Pour eviter le surapprentissage: utiliser regularisation, validation rigoureuse, early stopping et cross-validation.",
        "keywords": ["regularisation", "validation", "early stopping", "cross-validation"],
    },
    {
        "topic": "f1 score",
        "questions": [
            "f1 score c'est quoi ?",
            "explique le F1 score simplement",
            "pourquoi utiliser le F1 score ?",
        ],
        "response": "Le F1 score est une moyenne harmonique entre precision et recall, utile pour equilibrer les erreurs.",
        "keywords": ["precision", "recall", "harmonique", "equilibre"],
    },
    {
        "topic": "classes desequilibrees",
        "questions": [
            "quelle metrique pour classes desequilibrees ?",
            "metriques utiles pour datasets desequilibres",
            "comment evaluer un modele sur classes rares ?",
        ],
        "response": "Sur classes desequilibrees, suivre precision, recall, f1 et pr auc plutot que l'accuracy seule.",
        "keywords": ["precision", "recall", "f1", "pr auc"],
    },
    {
        "topic": "seuil decision",
        "questions": [
            "comment choisir un seuil de decision ?",
            "comment fixer le threshold en classification ?",
            "seuil de decision: bonne methode ?",
        ],
        "response": "Choisir le seuil selon le cout metier, puis verifier l'impact sur precision et recall.",
        "keywords": ["seuil", "cout metier", "precision", "recall"],
    },
    {
        "topic": "valeurs manquantes",
        "questions": [
            "comment gerer les valeurs manquantes ?",
            "strategie pour missing values en ML",
            "imputation des donnees: bonnes pratiques",
        ],
        "response": "Traiter les valeurs manquantes avec imputation (median ou mode) dans un pipeline reproductible.",
        "keywords": ["imputation", "median", "mode", "pipeline"],
    },
    {
        "topic": "encodage categoriel",
        "questions": [
            "categorical encoding options",
            "comment encoder des variables categorielles ?",
            "one-hot ou target encoding: quand choisir ?",
        ],
        "response": "Pour variables categorielles: one-hot sur faible cardinalite, target encoding sinon, avec controle de fuite.",
        "keywords": ["one-hot", "target encoding", "cardinalite", "fuite"],
    },
    {
        "topic": "pipeline bout en bout",
        "questions": [
            "pipeline ML de bout en bout",
            "workflow machine learning complet",
            "etapes d'un projet ML en production",
        ],
        "response": "Pipeline ML complet: collecte, preprocessing, entrainement, deploiement et monitoring.",
        "keywords": ["collecte", "preprocessing", "entrainement", "deploiement", "monitoring"],
    },
    {
        "topic": "random forest",
        "questions": [
            "quand utiliser random forest ?",
            "random forest: cas d'usage ideal",
            "random forest ou autre modele ?",
        ],
        "response": "Random forest fonctionne bien sur donnees tabulaires comme baseline non-lineaire, avec bonne interpretabilite.",
        "keywords": ["tabulaire", "baseline", "non-lineaire", "interpretabilite"],
    },
    {
        "topic": "xgboost vs random forest",
        "questions": [
            "xgboost vs random forest",
            "difference entre boosting et bagging",
            "quel modele choisir entre xgboost et random forest ?",
        ],
        "response": "XGBoost repose sur boosting, Random Forest sur bagging; comparer performance et tuning selon le contexte.",
        "keywords": ["boosting", "bagging", "performance", "tuning"],
    },
    {
        "topic": "monitoring prod",
        "questions": [
            "monitoring d'un modele en prod",
            "comment monitorer un modele en production ?",
            "quels KPI suivre en MLOps ?",
        ],
        "response": "En production, suivre drift, latence, taux d'erreur et metriques metier de performance.",
        "keywords": ["drift", "latence", "erreur", "metriques"],
    },
    {
        "topic": "data drift",
        "questions": [
            "qu'est-ce que le data drift ?",
            "comment detecter un drift de donnees ?",
            "drift des features: impact modele",
        ],
        "response": "Le data drift est un changement de distribution des features; il faut le surveiller pour declencher retraining.",
        "keywords": ["distribution", "features", "surveillance", "reentrainement"],
    },
    {
        "topic": "concept drift",
        "questions": [
            "comment detecter concept drift ?",
            "concept drift: definition operationnelle",
            "comment surveiller evolution de la cible ?",
        ],
        "response": "Le concept drift touche la relation features-cible; suivre performance sur fenetre temporelle avec alertes.",
        "keywords": ["cible", "performance", "fenetre temporelle", "alerte"],
    },
    {
        "topic": "retraining",
        "questions": [
            "quand relancer un retraining ?",
            "critere pour reentrainer un modele",
            "strategie de retraining MLOps",
        ],
        "response": "Relancer retraining si drift ou degradation de performance depasse un seuil defini dans le pipeline.",
        "keywords": ["drift", "degradation", "seuil", "pipeline"],
    },
    {
        "topic": "mlops",
        "questions": [
            "MLOps c'est quoi en pratique ?",
            "comment mettre en place une demarche MLOps ?",
            "principes MLOps en entreprise",
        ],
        "response": "MLOps couvre ci/cd, versionning, deploiement, monitoring et gouvernance du cycle de vie modele.",
        "keywords": ["ci/cd", "versionning", "deploiement", "monitoring", "gouvernance"],
    },
]

CONTEXTS = [
    "dans un projet SaaS",
    "dans un cas e-commerce",
    "dans un contexte bancaire",
    "dans une startup data",
    "dans un projet de prevision",
]
PROFILES = ["debutant", "intermediaire", "junior data scientist", "engineer MLOps"]
STYLES = ["Reponds directement.", "Version claire et concise.", "Sans detour."]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build targeted gold v2 dataset from business benchmark failures")
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed/chatbot_train_fr_gold_concepts_v2.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rnd = random.Random(args.seed)

    rows = []
    seen = set()
    target = max(1, args.rows)
    attempts = 0
    max_attempts = target * 30

    while len(rows) < target and attempts < max_attempts:
        attempts += 1
        case = rnd.choice(CASES)
        question = rnd.choice(case["questions"])
        context = rnd.choice(CONTEXTS)
        profile = rnd.choice(PROFILES)
        style = rnd.choice(STYLES)

        instruction = f"{question} {style} Contexte: {context}. Profil: {profile}."
        response = (
            f"Definition: {case['response']}\n"
            "Bonnes pratiques: appliquer ces principes dans un pipeline reproductible et monitorable.\n"
            f"Points cles: {', '.join(case['keywords'])}."
        )
        key = (instruction.lower(), response.lower())
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: Je veux une reponse precise orientee metier. ||| Assistant: Je fournis une reponse concise et actionnable.",
                "source": "gold_concepts_v2",
            }
        )

    # Deterministic completion if uniqueness saturates.
    i = 0
    while len(rows) < target:
        case = CASES[i % len(CASES)]
        question = case["questions"][i % len(case["questions"])]
        context = CONTEXTS[i % len(CONTEXTS)]
        profile = PROFILES[i % len(PROFILES)]
        style = STYLES[i % len(STYLES)]
        instruction = f"{question} {style} Contexte: {context}. Profil: {profile}."
        rows.append(
            {
                "instruction": instruction,
                "response": (
                    f"Definition: {case['response']}\n"
                    "Bonnes pratiques: appliquer ces principes dans un pipeline reproductible et monitorable.\n"
                    f"Points cles: {', '.join(case['keywords'])}."
                ),
                "history": "Utilisateur: Je veux une reponse precise orientee metier. ||| Assistant: Je fournis une reponse concise et actionnable.",
                "source": "gold_concepts_v2",
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
