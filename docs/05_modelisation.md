# Semaine 5 - Protocole de modélisation

## Partitions

1. dates triées chronologiquement ;
2. deux dernières dates réservées au test final ;
3. parmi les autres dates, les dernières 20 % forment la validation ;
4. aucune ligne d'une date ne traverse deux partitions.

Le test final n'est pas scoré pendant la semaine 5. Cette règle empêche d'ajuster le
pipeline sur le futur jeu d'évaluation.

## Modèles candidats

- moyenne globale : baseline minimale ;
- moyenne saisonnière mensuelle : baseline climatique ;
- régression linéaire standardisée : référence interprétable ;
- Random Forest compacte : non-linéarités et interactions, implémentée en NumPy ;
- gradient boosting de petits arbres : candidat de boosting local ;
- XGBoost natif : candidat avancé exécuté sur toute la partition d'entraînement réelle.

## Sélection préliminaire

Le candidat avec la plus faible MAE sur les dates de validation est retenu. Les
métriques publiées sont MAE, RMSE, R² et gain MAE par rapport à la baseline globale.
Le modèle sélectionné est ensuite réentraîné sur entraînement + validation, tandis
que les dates finales restent réservées.

Sur le run réel, la baseline saisonnière obtient la meilleure MAE (1,728 °C).
La Random Forest est le meilleur candidat Machine Learning (2,944 °C), mais ne
bat pas cette baseline. Elle est conservée comme candidat à analyser, sans
prétendre à un gain non observé.

## Prévention des fuites

- les coordonnées et la date ne sont pas utilisées directement comme identifiants ;
- les transformations sont ajustées sur l'entraînement ;
- le split est enregistré dans `splits.json` ;
- une assertion bloque tout chevauchement de dates ;
- la cible n'apparaît jamais dans la liste des features.
