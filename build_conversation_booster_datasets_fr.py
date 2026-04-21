import argparse
import csv
from pathlib import Path


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)


def build_conversation_base_rows(target_rows: int) -> list[dict[str, str]]:
    starters = [
        "bonjour",
        "hello",
        "bonsoir",
        "merci pour ton aide",
        "comment tu t'appelles",
        "qui es tu",
        "que peux tu faire",
        "tu peux m'aider en machine learning",
        "je veux apprendre le machine learning",
        "explique moi ton domaine",
    ]

    tones = [
        "debutant",
        "intermediaire",
        "expert",
    ]

    focus = [
        "data preparation",
        "modelisation",
        "evaluation",
        "deploiement api",
        "monitoring",
        "automatisation",
        "debugging python",
    ]

    rows: list[dict[str, str]] = []
    idx = 0
    while len(rows) < target_rows:
        s = starters[idx % len(starters)]
        t = tones[idx % len(tones)]
        f = focus[idx % len(focus)]

        instruction = f"{s} et j'ai un niveau {t}, aide moi sur {f}"
        response = (
            "Accueil et cadrage:\n"
            "- Definition: je suis Elibot, assistant specialise en data, IA appliquee et automatisation.\n"
            "- Ce que je peux faire: expliquer un concept, proposer une methode, donner un plan concret.\n"
            f"- Axe recommande pour toi ({t}): commencer par {f} avec objectifs et contraintes.\n"
            "- Prochaine action: envoie ton cas d'usage et je te donne une reponse detaillee et operationnelle."
        )

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "",
                "source": "conversation_base_booster",
            }
        )
        idx += 1
    return rows


def build_simple_detailed_rows(target_rows: int) -> list[dict[str, str]]:
    questions = [
        ("c'est quoi le machine learning", "apprentissage automatique", "predire ou classer"),
        ("c'est quoi un pipeline ml", "chaine de traitement", "collecte a monitoring"),
        ("c'est quoi overfitting", "surapprentissage", "train tres bon et test faible"),
        ("c'est quoi data leakage", "fuite de donnees", "signal cible present avant prediction"),
        ("c'est quoi la cross validation", "evaluation robuste", "plusieurs splits"),
        ("c'est quoi precision recall", "metriques de classification", "faux positifs et faux negatifs"),
        ("c'est quoi feature engineering", "creation de variables", "ameliorer signal utile"),
        ("c'est quoi model registry", "gestion des versions", "rollback et traçabilite"),
        ("c'est quoi le data drift", "derive des features", "distribution qui change"),
        ("c'est quoi le concept drift", "derive relation cible", "performance qui degrade"),
    ]

    contexts = [
        "en version simple",
        "avec un exemple concret",
        "avec des remarques d'expert",
        "et les erreurs frequentes",
        "et ce que je dois faire en prod",
    ]

    rows: list[dict[str, str]] = []
    idx = 0
    while len(rows) < target_rows:
        q, definition, point = questions[idx % len(questions)]
        c = contexts[idx % len(contexts)]

        instruction = f"{q} {c}"
        response = (
            "Reponse detaillee:\n"
            f"- Definition: {definition}.\n"
            f"- Explication: idee cle = {point}.\n"
            "- Exemple: sur un projet reel, on valide sur des donnees hors entrainement puis on mesure l'impact metier.\n"
            "- Remarque d'expert: ne pas s'arreter a la theorie, verifier metrique, seuil de decision et robustesse en production.\n"
            "- Action concrete: je peux te proposer une checklist adaptee a ton cas en 5 etapes."
        )

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "",
                "source": "simple_detailed_booster",
            }
        )
        idx += 1
    return rows


def build_multiturn_rows(target_rows: int) -> list[dict[str, str]]:
    anchors = [
        ("c'est quoi le machine learning", "definition de base + exemples metier"),
        ("explique un pipeline ml", "etapes de collecte a monitoring"),
        ("comment deployer avec fastapi", "contrats, erreurs, observabilite"),
        ("comment gerer le drift", "surveillance, alerte, retraining"),
        ("comment ameliorer un modele", "analyse d'erreurs, features, seuil"),
    ]

    follow_ups = [
        "approfondis avec plus de details techniques",
        "donne des remarques d'expert",
        "donne un exemple concret en production",
        "quels controles faire chaque semaine",
        "quels pieges je dois eviter",
    ]

    rows: list[dict[str, str]] = []
    idx = 0
    while len(rows) < target_rows:
        anchor_q, anchor_a = anchors[idx % len(anchors)]
        follow = follow_ups[idx % len(follow_ups)]

        history = (
            f"Utilisateur: {anchor_q} ||| "
            f"Assistant: Reponse initiale: {anchor_a}. "
            "Je peux detailler architecture, risques, et plan d'execution."
        )
        instruction = follow
        response = (
            "Approfondissement contextualise:\n"
            "- Rappel du contexte: je reprends le sujet precedent pour garder le fil de la conversation.\n"
            "- Details techniques: definir entree/sortie, metrique, seuil, monitoring et plan de rollback.\n"
            "- Remarques d'expert: prioriser les risques qui coutent le plus en production.\n"
            "- Controle hebdo type: qualite data, drift, latence API, taux d'erreur et performance metier.\n"
            "- Suite: si tu veux, je te fais un plan d'action sur 7 jours avec priorites."
        )

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": history,
                "source": "multiturn_conversation_booster",
            }
        )
        idx += 1
    return rows


def build_goal_following_rows(target_rows: int) -> list[dict[str, str]]:
    scenarios = [
        "objectif: deployer un modele de churn via API en 7 jours",
        "objectif: reduire les faux positifs d'un detecteur de fraude",
        "objectif: mettre en place un monitoring drift en production",
        "objectif: passer d'un notebook a un pipeline CI/CD MLOps",
        "objectif: stabiliser les predictions et le seuil de decision",
    ]

    constraints = [
        "contrainte: petite equipe data",
        "contrainte: delai court",
        "contrainte: budget limite",
        "contrainte: compliance forte",
        "contrainte: latence API inferieure a 200ms",
    ]

    asks = [
        "fais un plan concret et ordonne",
        "decoupe en etapes actionnables",
        "donne les priorites semaine 1",
        "propose les KPI de suivi",
        "donne une checklist d'execution",
    ]

    rows: list[dict[str, str]] = []
    idx = 0
    while len(rows) < target_rows:
        s = scenarios[idx % len(scenarios)]
        c = constraints[idx % len(constraints)]
        a = asks[idx % len(asks)]

        instruction = f"{s}; {c}; {a}"
        response = (
            "Plan guide par objectif:\n"
            "1) Cadrage: reformuler objectif, metrique de succes et risques critiques.\n"
            "2) Donnees: verifier qualite, fuites et schema d'entree/sortie.\n"
            "3) Modele: baseline, tuning cible, seuil de decision aligne metier.\n"
            "4) Industrialisation: API robuste (validation, erreurs, logs, versionning).\n"
            "5) Exploitation: monitoring drift/performance, alertes, boucle de retraining.\n"
            "6) Suivi objectif: jalons hebdo, KPI, plan de rollback, prochaines actions."
        )

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "",
                "source": "goal_following_booster",
            }
        )
        idx += 1
    return rows


def build_style_signature_rows(target_rows: int) -> list[dict[str, str]]:
    prompts = [
        "explique le data drift de facon claire et pro",
        "donne une reponse experte sur precision recall",
        "comment servir un modele ml avec fastapi",
        "que faire contre le surapprentissage",
        "comment organiser un pipeline ml en production",
    ]

    tones = [
        "ton professionnel et chaleureux",
        "style pedagogique et structuré",
        "reponse concise puis detaillee",
        "format expert avec exemples",
    ]

    rows: list[dict[str, str]] = []
    idx = 0
    while len(rows) < target_rows:
        p = prompts[idx % len(prompts)]
        t = tones[idx % len(tones)]
        instruction = f"{p}; style attendu: {t}"
        response = (
            "Reponse signature Elibot:\n"
            "- Definition rapide: poser le concept en une phrase claire.\n"
            "- Explication utile: relier au cas terrain et au risque principal.\n"
            "- Exemple concret: montrer une situation production realiste.\n"
            "- Remarque d'expert: indiquer le piege frequent et le bon reflexe.\n"
            "- Action suivante: proposer une checklist courte et priorisee."
        )

        rows.append(
            {
                "instruction": instruction,
                "response": response,
                "history": "",
                "source": "style_signature_booster",
            }
        )
        idx += 1
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conversational booster datasets for Elibot.")
    parser.add_argument("--out-base", default="data/processed/chatbot_train_fr_conversation_base.csv")
    parser.add_argument("--out-simple", default="data/processed/chatbot_train_fr_simple_detailed.csv")
    parser.add_argument("--out-multiturn", default="data/processed/chatbot_train_fr_multiturn_contextual.csv")
    parser.add_argument("--out-goal", default="data/processed/chatbot_train_fr_goal_following.csv")
    parser.add_argument("--out-style", default="data/processed/chatbot_train_fr_style_signature.csv")
    parser.add_argument("--rows-base", type=int, default=700)
    parser.add_argument("--rows-simple", type=int, default=1600)
    parser.add_argument("--rows-multiturn", type=int, default=900)
    parser.add_argument("--rows-goal", type=int, default=800)
    parser.add_argument("--rows-style", type=int, default=700)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_rows = build_conversation_base_rows(args.rows_base)
    simple_rows = build_simple_detailed_rows(args.rows_simple)
    multiturn_rows = build_multiturn_rows(args.rows_multiturn)
    goal_rows = build_goal_following_rows(args.rows_goal)
    style_rows = build_style_signature_rows(args.rows_style)

    out_base = Path(args.out_base)
    out_simple = Path(args.out_simple)
    out_multiturn = Path(args.out_multiturn)
    out_goal = Path(args.out_goal)
    out_style = Path(args.out_style)

    write_rows(out_base, base_rows)
    write_rows(out_simple, simple_rows)
    write_rows(out_multiturn, multiturn_rows)
    write_rows(out_goal, goal_rows)
    write_rows(out_style, style_rows)

    print({
        "base_rows": len(base_rows),
        "simple_rows": len(simple_rows),
        "multiturn_rows": len(multiturn_rows),
        "goal_rows": len(goal_rows),
        "style_rows": len(style_rows),
        "out_base": str(out_base).replace("\\", "/"),
        "out_simple": str(out_simple).replace("\\", "/"),
        "out_multiturn": str(out_multiturn).replace("\\", "/"),
        "out_goal": str(out_goal).replace("\\", "/"),
        "out_style": str(out_style).replace("\\", "/"),
    })


if __name__ == "__main__":
    main()
