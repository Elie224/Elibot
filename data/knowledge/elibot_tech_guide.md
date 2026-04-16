# Elibot Knowledge Guide

## Domaine
Elibot repond uniquement sur l analyse de donnees, le machine learning, l IA appliquee, les API et l automatisation.

## Methode recommandee pour une question technique
1. Clarifier objectif, entree, sortie et contraintes.
2. Proposer un plan en etapes courtes.
3. Donner un exemple executable (Python, SQL, API).
4. Ajouter verification, logs et gestion d erreurs.
5. Terminer par une checklist de validation.

## Bonnes pratiques pandas
- Preferer vectorisation a apply ligne par ligne.
- Convertir les types explicites avant transformations.
- Isoler nettoyage, feature engineering et export en fonctions distinctes.
- Ajouter tests sur un petit echantillon.

## Bonnes pratiques pipeline ML
- Separation stricte train/validation/test.
- Pipeline unique pour preprocessing + modele.
- Metriques explicites selon le cas (classification/regression).
- Journaliser versions des donnees et hyperparametres.

## Bonnes pratiques API
- Valider les entrees et retourner des erreurs explicites.
- Ajouter timeout, retry et logs.
- Ne jamais exposer de secrets dans les reponses.
