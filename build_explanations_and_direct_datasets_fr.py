import argparse
import csv
import random
from pathlib import Path

CONCEPTS = [
    {
        "name": "machine learning",
        "definition": "Le machine learning est une branche de l'IA qui apprend des patterns a partir de donnees pour faire des predictions sans coder toutes les regles a la main.",
        "example": "Predire si un client va quitter un service a partir de son historique d'usage.",
        "best": "Demarrer par une baseline simple, choisir une metrique metier, et separer train/validation/test proprement.",
        "pitfalls": "Utiliser des informations futures (data leakage) ou evaluer sur des donnees non representatives.",
    },
    {
        "name": "overfitting",
        "definition": "L'overfitting apparait quand le modele memorise trop le train et generalise mal sur des donnees nouvelles.",
        "example": "Score train tres haut, score validation nettement plus bas.",
        "best": "Cross-validation, regularisation, simplification du modele et early stopping.",
        "pitfalls": "Tuner excessivement sur un seul split de validation.",
    },
    {
        "name": "cross-validation",
        "definition": "La cross-validation evalue un modele sur plusieurs decoupages des donnees pour obtenir une mesure plus robuste.",
        "example": "K-fold: le modele est entraine K fois en changeant le fold de validation.",
        "best": "Utiliser une version stratifiee si classes desequilibrees.",
        "pitfalls": "Faire le preprocessing avant split et introduire une fuite de donnees.",
    },
    {
        "name": "precision et recall",
        "definition": "La precision mesure la qualite des positifs predits, le recall mesure la capacite a retrouver les vrais positifs.",
        "example": "En detection de fraude, on prefere souvent un recall eleve pour rater moins de fraudes.",
        "best": "Choisir le compromis selon le cout metier des faux positifs/faux negatifs.",
        "pitfalls": "Comparer des modeles sans fixer ou analyser le seuil de decision.",
    },
    {
        "name": "feature engineering",
        "definition": "Le feature engineering consiste a creer ou transformer des variables pour rendre le signal plus exploitable par le modele.",
        "example": "Transformer une date en jour de semaine, mois et saison.",
        "best": "Valider les features avec le metier et tester leur impact incremental.",
        "pitfalls": "Encoder des informations indisponibles au moment de prediction.",
    },
    {
        "name": "data leakage",
        "definition": "La data leakage se produit quand une information liee a la cible fuit dans les variables explicatives.",
        "example": "Normaliser le dataset complet avant de separer train/test.",
        "best": "Construire un pipeline qui fit uniquement sur le train.",
        "pitfalls": "Se fier a une performance offline irrealiste qui s'ecroule en prod.",
    },
    {
        "name": "pipeline ML",
        "definition": "Un pipeline ML structure le cycle complet: cadrage, preparation data, entrainement, evaluation, deploiement et monitoring.",
        "example": "Pipeline de scoring client mis a jour chaque semaine avec suivi de derive.",
        "best": "Versionner donnees, code, modele et metriques.",
        "pitfalls": "Ne pas monitorer la derivee de donnees et la degradation de performance.",
    },
    {
        "name": "model serving avec FastAPI",
        "definition": "Le model serving expose un modele via API pour des predictions temps reel ou batch.",
        "example": "Endpoint /predict qui recoit un JSON et renvoie un score de risque.",
        "best": "Ajouter validation Pydantic, logs structures, metriques et health checks.",
        "pitfalls": "Deployer sans gestion d'erreurs ni versionnage du modele.",
    },
]

DIRECT_QUESTIONS = [
    "C'est quoi le machine learning ?",
    "Explique simplement l'overfitting.",
    "Quelle est la difference entre precision et recall ?",
    "A quoi sert la cross-validation ?",
    "Comment faire un pipeline ML propre ?",
    "Qu'est-ce que la data leakage ?",
    "Comment servir un modele ML avec FastAPI ?",
    "Comment choisir une metrique de classification ?",
    "C'est quoi une baseline en machine learning ?",
    "Comment eviter des reponses trop superficielles d'un chatbot ML ?",
]

DIRECT_ANSWERS = [
    "Le machine learning permet a un modele d'apprendre a partir de donnees pour predire ou classer sans ecrire toutes les regles a la main.",
    "L'overfitting, c'est quand le modele apprend trop bien le train mais echoue sur des donnees nouvelles. On le reduit avec regularisation, cross-validation et early stopping.",
    "Precision = qualite des positifs predits. Recall = capacite a retrouver les vrais positifs. Le bon compromis depend du cout metier des erreurs.",
    "La cross-validation donne une evaluation plus fiable en testant le modele sur plusieurs decoupages des donnees.",
    "Un pipeline ML propre suit: cadrage, data quality, features, train/val/test, entrainement, evaluation, deploiement, monitoring.",
    "La data leakage est une fuite d'information cible dans les features. Elle gonfle artificiellement les scores et casse la perf en production.",
    "Pour FastAPI: /health, /predict, schemas Pydantic, gestion d'erreurs centralisee, logs + metriques + versionnage du modele.",
    "En classification desequilibree, privilegie precision/recall/F1/PR-AUC plutot que l'accuracy seule.",
    "Une baseline est un modele simple de reference pour verifier qu'une solution plus complexe apporte un vrai gain.",
    "Pour des reponses moins superficielles: plus d'exemples detailles, format structure, et entrainement sur des reponses directes sans detour.",
]

AUDIENCES = ["debutant", "intermediaire", "junior data scientist", "chef de projet", "engineer MLOps"]
CONTEXTS = [
    "dans un projet SaaS",
    "dans un cas e-commerce",
    "dans un contexte bancaire",
    "dans une startup data",
    "dans un projet de prevision",
    "dans un service client automatise",
]
OBJECTIVES = [
    "ameliorer la qualite de prediction",
    "reduire les erreurs en production",
    "stabiliser le pipeline",
    "accelerer la mise en production",
    "mieux expliquer les decisions du modele",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build explanation and direct-answer French datasets")
    parser.add_argument("--explanations-rows", type=int, default=2000)
    parser.add_argument("--direct-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-explanations", default="data/processed/chatbot_train_fr_explanations_detailed.csv")
    parser.add_argument("--out-direct", default="data/processed/chatbot_train_fr_direct_answers.csv")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)


def build_explanations_rows(n: int, seed: int) -> list[dict[str, str]]:
    rnd = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    frames = [
        "Explique {name} de facon pedagogique pour un debutant.",
        "Donne une explication detaillee de {name} avec exemple concret.",
        "Je veux comprendre {name}: definition, cas pratique et erreurs a eviter.",
        "Peux-tu vulgariser {name} sans perdre la rigueur technique ?",
        "Explique {name} pour un profil {audience} {context}.",
        "Comment appliquer {name} pour {objective} ?",
    ]

    tries = 0
    while len(rows) < n and tries < n * 20:
        tries += 1
        c = rnd.choice(CONCEPTS)
        instruction = rnd.choice(frames).format(
            name=c["name"],
            audience=rnd.choice(AUDIENCES),
            context=rnd.choice(CONTEXTS),
            objective=rnd.choice(OBJECTIVES),
        )
        scenario = rnd.choice(CONTEXTS)
        angle = rnd.choice([
            "focalise sur l'implementation",
            "focalise sur la validation",
            "focalise sur la production",
            "focalise sur les erreurs frequentes",
        ])
        response = (
            f"Definition: {c['definition']}\n"
            f"Exemple: {c['example']}\n"
            f"Bonnes pratiques: {c['best']}\n"
            f"Pieges a eviter: {c['pitfalls']}\n"
            f"Contexte: {scenario}; Conseil: {angle}."
        )

        key = (instruction.strip().lower(), response.strip().lower())
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": (
                    f"Utilisateur: Mon profil est {rnd.choice(AUDIENCES)}. ||| "
                    f"Assistant: Je donne une reponse structuree orientee {rnd.choice(OBJECTIVES)}."
                ),
                "source": "explanations_detailed",
            }
        )

    return rows


def build_direct_rows(n: int, seed: int) -> list[dict[str, str]]:
    rnd = random.Random(seed + 1)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    variants = [
        "Reponds directement.",
        "Sans poser de question supplementaire.",
        "Version claire et concise.",
        "Version utile pour debuter vite.",
    ]

    tries = 0
    while len(rows) < n and tries < n * 30:
        tries += 1
        idx = rnd.randrange(len(DIRECT_QUESTIONS))
        audience = rnd.choice(AUDIENCES)
        context = rnd.choice(CONTEXTS)
        objective = rnd.choice(OBJECTIVES)
        instruction = f"{DIRECT_QUESTIONS[idx]} {rnd.choice(variants)} Contexte: {context}. Profil: {audience}."
        response = (
            f"{DIRECT_ANSWERS[idx]} "
            f"({rnd.choice(['Action immediate', 'Point cle', 'Conseil pratique'])}; objectif: {objective})"
        )

        key = (instruction.strip().lower(), response.strip().lower())
        if key in seen:
            continue
        seen.add(key)

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "Utilisateur: Reponds directement. ||| Assistant: Reponse immediate, sans detour.",
                "source": "direct_answers",
            }
        )

    return rows


def main() -> None:
    args = parse_args()

    explanations = build_explanations_rows(max(1, args.explanations_rows), args.seed)
    direct = build_direct_rows(max(1, args.direct_rows), args.seed)

    out_exp = Path(args.out_explanations)
    out_dir = Path(args.out_direct)

    _write_csv(out_exp, explanations)
    _write_csv(out_dir, direct)

    print(
        {
            "explanations_rows": len(explanations),
            "direct_rows": len(direct),
            "out_explanations": str(out_exp),
            "out_direct": str(out_dir),
        }
    )


if __name__ == "__main__":
    main()
