# Fiche modèle préliminaire - semaine 5

## Modèle sélectionné

`linear_regression`, sélectionné sur la MAE des dates de validation.

| Métrique | Valeur |
|---|---:|
| MAE | 0.6741 °C |
| RMSE | 0.8474 °C |
| R² | 0.9415 |
| Gain MAE vs baseline | 78.08 % |

## Données

Jeu **synthétique de démonstration** : 17,718 lignes d'entraînement,
5,417 lignes de validation et 2,586 lignes
réservées. Ces métriques valident le pipeline logiciel ; elles ne mesurent pas encore
la performance scientifique sur Rabat.

## Partitions temporelles

- entraînement : 2019-06-24, 2019-08-11, 2020-06-10, 2020-07-28, 2021-06-29, 2021-08-16, 2022-06-16, 2022-07-18, 2022-08-19, 2023-06-03, 2023-07-21, 2023-08-22
- validation : 2024-06-21, 2024-07-23, 2024-08-24, 2025-06-08
- test final réservé : 2025-07-26, 2025-08-27

Le test final n'a pas été scoré.

## Usage prévu

Estimation de la température diurne de surface sur une grille 30 m, une fois le modèle
réentraîné et validé sur les observations réelles harmonisées.

## Usages interdits

Ne pas interpréter la sortie comme température de l'air, température ressentie,
diagnostic médical, mesure quotidienne satellitaire directe ou vérité terrain.

## Étape suivante

Semaine 6 : ouvrir le test final une seule fois, produire les résidus spatiaux,
analyser les erreurs par date et documenter l'incertitude.
