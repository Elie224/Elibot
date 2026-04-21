import csv
from pathlib import Path

OUT_PATH = Path("data/processed/chatbot_train_fr_ml_120_classical_advanced.csv")

SECTIONS = {
    "stats_theorie": [
        "Qu est-ce qu un estimateur consistant",
        "Qu est-ce qu un estimateur efficace",
        "Qu est-ce que le theoreme central limite",
        "Qu est-ce que la loi des grands nombres",
        "Qu est-ce qu un intervalle de confiance robuste",
        "Qu est-ce qu un test d hypothese",
        "Qu est-ce que le test du chi-deux",
        "Qu est-ce que le test de Kolmogorov-Smirnov",
        "Qu est-ce que le test de Shapiro-Wilk",
        "Qu est-ce que la correlation de Spearman",
        "Qu est-ce que la correlation de Kendall",
        "Qu est-ce que la covariance",
        "Qu est-ce que la multicolinearite",
        "Comment detecter la multicolinearite",
        "Qu est-ce que le VIF",
        "Qu est-ce que l heteroscedasticite",
        "Qu est-ce que le test de Breusch-Pagan",
        "Qu est-ce que le test de Durbin-Watson",
        "Qu est-ce qu un modele lineaire generalise",
        "Qu est-ce qu un modele additif generalise",
    ],
    "regression_lineaire": [
        "Comment interpreter les coefficients d une regression lineaire",
        "Comment interpreter les coefficients d une regression logistique",
        "Comment gerer la multicolinearite dans une regression",
        "Comment choisir entre Ridge et Lasso",
        "Comment interpreter les odds ratios",
        "Comment interpreter les p-values dans une regression",
        "Comment interpreter les intervalles de confiance des coefficients",
        "Comment detecter un modele mal specifie",
        "Comment diagnostiquer les residus",
        "Comment interpreter un QQ-plot",
        "Comment gerer les interactions entre variables",
        "Comment gerer les polynomes dans une regression",
        "Comment choisir le degre d un polynome",
        "Comment interpreter un modele log-log",
        "Comment interpreter un modele log-lin",
        "Comment interpreter un modele lin-log",
        "Comment interpreter un modele Poisson",
        "Comment interpreter un modele binomial negatif",
        "Comment interpreter un modele ordinal",
        "Comment interpreter un modele multinomial",
    ],
    "feature_engineering": [
        "Comment creer des features temporelles",
        "Comment creer des features categorielles avancees",
        "Comment creer des features d interaction",
        "Comment creer des features polynomiales",
        "Comment creer des features basees sur des groupes",
        "Comment creer des features rolling window",
        "Comment creer des features lag",
        "Comment creer des features lead",
        "Comment encoder des categories rares",
        "Comment encoder des categories ordinales",
        "Comment encoder des categories hierarchiques",
        "Comment encoder des categories multi-labels",
        "Comment gerer les cardinalites elevees",
        "Comment faire du target encoding",
        "Comment eviter le leakage dans le target encoding",
        "Comment faire du weight of evidence",
        "Comment faire du binning optimal",
        "Comment faire du binning supervise",
        "Comment faire du binning non supervise",
        "Comment faire du hashing trick",
    ],
    "probabilistes_graphiques": [
        "Qu est-ce qu un modele bayesien",
        "Qu est-ce qu un reseau bayesien",
        "Qu est-ce qu un reseau de Markov",
        "Qu est-ce qu un modele de Markov cache",
        "Qu est-ce qu un modele AR",
        "Qu est-ce qu un modele MA",
        "Qu est-ce qu un modele ARMA",
        "Qu est-ce qu un modele ARIMA",
        "Qu est-ce qu un modele SARIMA",
        "Qu est-ce qu un modele VAR",
        "Qu est-ce qu un modele GARCH",
        "Qu est-ce qu un modele mixture gaussienne",
        "Qu est-ce qu un modele naif bayesien",
        "Qu est-ce qu un modele hierarchique bayesien",
        "Qu est-ce qu un modele latent variable",
        "Qu est-ce qu un modele generatif probabiliste",
        "Qu est-ce que l inference variationnelle",
        "Qu est-ce que MCMC",
        "Qu est-ce que Gibbs sampling",
        "Qu est-ce que Metropolis-Hastings",
    ],
    "series_temporelles": [
        "Comment detecter la stationnarite",
        "Comment appliquer le test ADF",
        "Comment appliquer le test KPSS",
        "Comment differencier une serie",
        "Comment detecter la saisonnalite",
        "Comment decomposer une serie",
        "Comment choisir un modele ARIMA",
        "Comment interpreter les ACF PACF",
        "Comment gerer les series irregulieres",
        "Comment gerer les series multi-variees",
        "Comment gerer les series avec ruptures",
        "Comment gerer les series avec outliers",
        "Comment faire du forecasting hierarchique",
        "Comment faire du forecasting probabiliste",
        "Comment faire du nowcasting",
        "Comment faire du backtesting",
        "Comment faire du walk-forward validation",
        "Comment faire du feature engineering temporel",
        "Comment faire du smoothing",
        "Comment faire du filtering",
    ],
    "fairness_robustesse": [
        "Comment detecter un biais algorithmique",
        "Comment mesurer la fairness",
        "Comment corriger un biais dans les donnees",
        "Comment corriger un biais dans un modele",
        "Comment tester la robustesse d un modele",
        "Comment tester la sensibilite d un modele",
        "Comment tester la stabilite d un modele",
        "Comment tester la coherence d un modele",
        "Comment tester la resilience aux perturbations",
        "Comment tester la resistance aux attaques adversariales",
        "Comment anonymiser un dataset",
        "Comment pseudonymiser un dataset",
        "Comment gerer les donnees sensibles",
        "Comment gerer les donnees personnelles",
        "Comment gerer les donnees manquantes sensibles",
        "Comment gerer les donnees biaisees",
        "Comment gerer les donnees non representatives",
        "Comment gerer les donnees protegees",
        "Comment gerer les donnees multi-pays",
        "Comment gerer les donnees multi-reglementaires",
    ],
}


def _response_template(section: str, question: str) -> str:
    intro = f"Question: {question}\n"

    if section == "stats_theorie":
        body = (
            "Reponse expert statistique:\n"
            "- Definition operationnelle: expliquer le concept et ses hypotheses.\n"
            "- Utilite ML: impact sur estimation, selection de modele et incertitude.\n"
            "- Verification pratique: tests, diagnostics, et seuils d alerte.\n"
            "- Pieges frequents: interpretation abusive des p-values, confusion correlation/causalite.\n"
            "- Decision: convertir le resultat statistique en action metier mesurable."
        )
    elif section == "regression_lineaire":
        body = (
            "Guide d interpretation:\n"
            "- Coefficients: sens, amplitude, et unite de variation.\n"
            "- Validite: residus, colinearite, specification, et robustesse.\n"
            "- Incertitude: intervalle de confiance, stabilite inter-echantillons.\n"
            "- Arbitrage: explicabilite vs performance selon contrainte metier.\n"
            "- Sortie attendue: recommandations actionnables par segment cible."
        )
    elif section == "feature_engineering":
        body = (
            "Pipeline feature engineering:\n"
            "- Conception: features temporelles, interactions et agregats de groupe.\n"
            "- Encodage: strategie selon cardinalite, rarete et risque de fuite.\n"
            "- Validation: gain incremental, robustesse hors echantillon, et cout compute.\n"
            "- Hygiene: versionner transformations et prevenir le leakage.\n"
            "- Production: garantir parite train/serving avec tests automatiques."
        )
    elif section == "probabilistes_graphiques":
        body = (
            "Lecture probabiliste:\n"
            "- Intuition: variable latente, dependances conditionnelles, incertitude explicite.\n"
            "- Inference: exacte vs approximee selon cout et precision cibles.\n"
            "- Cas d usage: tabulaire, temporel, risque, scoring et detection d anomalies.\n"
            "- Limites: sensibilite aux hypotheses et complexite de calibration.\n"
            "- Bon reflexe: evaluer calibration probabiliste avant passage en prod."
        )
    elif section == "series_temporelles":
        body = (
            "Cadre time series:\n"
            "- Pre-analyse: stationnarite, saisonnalite, ruptures et outliers.\n"
            "- Modelisation: baseline naive, ARIMA/SARIMA/VAR selon structure.\n"
            "- Validation: backtesting et walk-forward pour eviter l illusion offline.\n"
            "- Monitoring: drift temporel, degradation horizon, recalibration des intervalles.\n"
            "- Action: ajuster horizon, granularite et politique de retraining."
        )
    else:
        body = (
            "Playbook robustesse et conformite:\n"
            "- Detection: biais, instabilite et vulnerabilites adversariales.\n"
            "- Mitigation: reweighting, contraintes fairness, et governance data.\n"
            "- Confidentialite: anonymisation, pseudonymisation, minimisation des donnees.\n"
            "- Multi-pays: adapter regles et audits aux exigences locales.\n"
            "- Exploitation: definir controles continus avant et apres deploiement."
        )

    checklist = (
        "\nMini checklist execution:\n"
        "1) Fixer la metrique business prioritaire et le seuil de risque.\n"
        "2) Segmenter l analyse par population et periode.\n"
        "3) Verifier reproductibilite et tracabilite complete."
    )
    return intro + body + checklist


def main() -> None:
    rows = []
    for section, questions in SECTIONS.items():
        for question in questions:
            rows.append(
                {
                    "instruction": question,
                    "response": _response_template(section, question),
                    "history": "Utilisateur: je veux une reponse experte sans deep learning. ||| Assistant: je fournis une explication exploitable en production avec diagnostics et checklist.",
                    "source": "ml_120_classical_advanced",
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
