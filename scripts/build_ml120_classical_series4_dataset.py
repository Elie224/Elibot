import csv
from pathlib import Path

OUT_PATH = Path("data/processed/chatbot_train_fr_ml_120_classical_series4.csv")

SECTIONS = {
    "modeles_lineaires_stats": [
        "Comment diagnostiquer la non-linearite dans une regression",
        "Comment tester l independance des residus",
        "Comment tester la normalite des residus",
        "Comment interpreter un modele avec interactions",
        "Comment interpreter un modele avec transformations log",
        "Comment interpreter un modele avec transformations Box-Cox",
        "Comment choisir entre modele additif et modele lineaire",
        "Comment detecter un modele mal specifie",
        "Comment interpreter un modele quasi-Poisson",
        "Comment interpreter un modele binomial negatif",
        "Comment interpreter un modele Tobit",
        "Comment interpreter un modele Probit",
        "Comment interpreter un modele Logit multinomial",
        "Comment interpreter un modele ordinal logit",
        "Comment interpreter un modele ordinal probit",
        "Comment tester l homoscedasticite",
        "Comment tester l autocorrelation",
        "Comment tester la significativite globale d un modele",
        "Comment interpreter un modele avec regularisation",
        "Comment interpreter un modele avec contraintes",
    ],
    "feature_engineering_avance": [
        "Comment creer des features basees sur des ratios",
        "Comment creer des features basees sur des differences",
        "Comment creer des features basees sur des agregations",
        "Comment creer des features basees sur des fenetres glissantes",
        "Comment creer des features basees sur des groupes hierarchiques",
        "Comment creer des features basees sur des interactions non lineaires",
        "Comment creer des features basees sur des distances",
        "Comment creer des features basees sur des similarites",
        "Comment creer des features basees sur des statistiques robustes",
        "Comment creer des features basees sur des quantiles",
        "Comment creer des features basees sur des encodages supervises",
        "Comment creer des features basees sur des encodages bayesiens",
        "Comment creer des features basees sur des encodages leave-one-out",
        "Comment creer des features basees sur des encodages target-mean",
        "Comment creer des features basees sur des encodages ordinal-probabilistes",
        "Comment creer des features basees sur des embeddings tabulaires",
        "Comment creer des features basees sur des clusters",
        "Comment creer des features basees sur des modeles pre-entraines",
        "Comment creer des features basees sur des regles metier",
        "Comment creer des features basees sur des signaux faibles",
    ],
    "probabilistes_graphiques": [
        "Comment construire un reseau bayesien",
        "Comment estimer les probabilites conditionnelles",
        "Comment faire de l inference exacte dans un reseau bayesien",
        "Comment faire de l inference approximative",
        "Comment faire du sampling dans un modele probabiliste",
        "Comment construire un modele de Markov cache",
        "Comment estimer les parametres d un HMM",
        "Comment faire du Viterbi decoding",
        "Comment faire du forward-backward",
        "Comment construire un modele mixture gaussienne",
        "Comment choisir le nombre de composantes",
        "Comment interpreter les responsibilities",
        "Comment interpreter les covariances dans un GMM",
        "Comment detecter un mauvais clustering probabiliste",
        "Comment faire du clustering hierarchique probabiliste",
        "Comment faire du filtering bayesien",
        "Comment faire du smoothing bayesien",
        "Comment faire du particle filtering",
        "Comment faire du Kalman filtering",
        "Comment faire du Kalman smoothing",
    ],
    "causalite": [
        "Comment construire un DAG causal",
        "Comment identifier les variables confondantes",
        "Comment identifier les variables mediatrices",
        "Comment identifier les variables colliders",
        "Comment detecter un biais de selection",
        "Comment faire du backdoor adjustment",
        "Comment faire du frontdoor adjustment",
        "Comment faire du matching causal",
        "Comment faire du propensity score matching",
        "Comment faire du weighting causal",
        "Comment faire du double robust estimation",
        "Comment faire du difference-in-differences",
        "Comment faire du synthetic control",
        "Comment faire du regression discontinuity",
        "Comment faire du instrumental variables",
        "Comment tester la validite d un instrument",
        "Comment interpreter un effet causal local",
        "Comment interpreter un effet causal moyen",
        "Comment interpreter un effet causal conditionnel",
        "Comment detecter un DAG mal specifie",
    ],
    "series_temporelles_avancees": [
        "Comment detecter une rupture structurelle",
        "Comment detecter un changement de regime",
        "Comment modeliser une serie non stationnaire",
        "Comment modeliser une serie avec saisonnalite multiple",
        "Comment modeliser une serie avec tendance non lineaire",
        "Comment modeliser une serie avec volatilite",
        "Comment modeliser une serie avec dependance longue",
        "Comment modeliser une serie avec bruit heteroscedastique",
        "Comment modeliser une serie avec valeurs manquantes",
        "Comment modeliser une serie avec anomalies",
        "Comment faire du forecasting hierarchique",
        "Comment faire du forecasting multi-horizon",
        "Comment faire du forecasting quantile",
        "Comment faire du forecasting probabiliste",
        "Comment faire du backtesting avance",
        "Comment faire du walk-forward robust",
        "Comment faire du cross-validation temporel",
        "Comment faire du feature engineering temporel avance",
        "Comment faire du smoothing exponentiel",
        "Comment faire du Holt-Winters",
    ],
    "mlops_qualite_robustesse": [
        "Comment tester la robustesse d un modele tabulaire",
        "Comment tester la sensibilite aux perturbations",
        "Comment tester la stabilite entre deux versions",
        "Comment tester la coherence inter-modeles",
        "Comment tester la coherence inter-segments",
        "Comment tester la coherence inter-pays",
        "Comment tester la coherence inter-periodes",
        "Comment detecter un drift lent",
        "Comment detecter un drift brutal",
        "Comment detecter un drift saisonnier",
        "Comment detecter un drift structurel",
        "Comment detecter un drift sur les features",
        "Comment detecter un drift sur les labels",
        "Comment detecter un drift sur les residus",
        "Comment monitorer un modele tabulaire",
        "Comment monitorer un modele probabiliste",
        "Comment monitorer un modele de series temporelles",
        "Comment monitorer un modele multi-segments",
        "Comment monitorer un modele multi-pays",
        "Comment monitorer un modele soumis a regulation",
    ],
}


def make_response(section: str, question: str) -> str:
    intro = f"Question: {question}\n"

    if section == "modeles_lineaires_stats":
        body = (
            "Diagnostic statistique applique:\n"
            "- Hypotheses a verifier: linearite, homoscedasticite, independance, normalite residuelle si necessaire.\n"
            "- Outils: tests formels + diagnostics visuels + comparaison entre specifications.\n"
            "- Lecture metier: mesurer l impact des violations sur le risque de decision.\n"
            "- Remediation: transformation, robust regression, regularisation ou changement de famille de modele.\n"
            "- Sortie attendue: modele interpretable, stable et defendable en audit."
        )
    elif section == "feature_engineering_avance":
        body = (
            "Design de features expert:\n"
            "- Point de depart: hypotheses metier explicites par segment.\n"
            "- Construction: ratios, deltas, agregats, fenetres et encodages supervises anti-leakage.\n"
            "- Controle qualite: robustesse outliers, drift, et cout compute.\n"
            "- Validation: gain incremental par bloc de features, pas seulement global.\n"
            "- Production: parite train/serving et versioning des transformations."
        )
    elif section == "probabilistes_graphiques":
        body = (
            "Approche probabiliste operationnelle:\n"
            "- Structuration: expliciter dependances conditionnelles et variables latentes.\n"
            "- Inference: exacte si possible, approximee si contraintes de latence/cout.\n"
            "- Estimation: calibrer distributions et verifier coherence probabiliste.\n"
            "- Edge cases: sensibilite aux hypotheses fortes et aux donnees rares.\n"
            "- Decision: privilegier la calibration et l incertitude exploitable en production."
        )
    elif section == "causalite":
        body = (
            "Cadre causal praticable:\n"
            "- DAG: identifier confounders, mediators, colliders et voies interdites.\n"
            "- Identification: backdoor/frontdoor/IV selon hypotheses testables.\n"
            "- Estimation: matching, weighting, DiD, RDD selon design de donnees.\n"
            "- Validation: tests de sensibilite et verifications d assumptions.\n"
            "- Usage: traduire ATE/CATE en decision politique produit ou risque."
        )
    elif section == "series_temporelles_avancees":
        body = (
            "Plan time-series robuste:\n"
            "- Analyse initiale: regime shifts, saisonnalites multiples, volatilite, anomalies.\n"
            "- Modelisation: baseline naive puis modele structurel adapte au signal.\n"
            "- Evaluation: backtesting chronologique strict et multi-horizon.\n"
            "- Monitoring: drift temporel, rupture structurelle et degradation par horizon.\n"
            "- Action: ajuster granularite, fenetre d entrainement et cadence de retraining."
        )
    else:
        body = (
            "Industrialisation et robustesse:\n"
            "- Controle continu: stabilite version a version et coherence inter-segments/pays.\n"
            "- Detection drift: lent, brutal, saisonnier, structurel sur features/labels/residus.\n"
            "- Observabilite: dashboards decisionnels relies aux SLO metier.\n"
            "- Governance: trace complete modele-data-code et politiques de rollback.\n"
            "- Conformite: preuves d audit pour environnements regules."
        )

    checklist = (
        "\nChecklist execution:\n"
        "1) Definir la metrique metier primaire + metrique de risque.\n"
        "2) Evaluer par segment, pays, periode et distribution.\n"
        "3) Verifier reproductibilite et plan de rollback avant promotion."
    )

    return intro + body + checklist


def main() -> None:
    rows = []
    for section, questions in SECTIONS.items():
        for question in questions:
            rows.append(
                {
                    "instruction": question,
                    "response": make_response(section, question),
                    "history": "Utilisateur: je veux une reponse experte, sans deep learning, orientee production. ||| Assistant: je fournis un cadre methodologique, des diagnostics et une checklist d execution.",
                    "source": "ml_120_classical_series4",
                }
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "response", "history", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved={OUT_PATH}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
