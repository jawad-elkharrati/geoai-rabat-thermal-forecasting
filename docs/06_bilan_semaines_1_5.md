# Bilan vérifié des semaines 1 à 5

## Semaine 1 - cadrage

- [x] périmètre, cible LST et période 2019-2025 définis ;
- [x] limite OSM de Rabat enregistrée ;
- [x] projection `EPSG:32629` et grille de 30 m fixées ;
- [x] sources, licences, risques et règles de qualité documentés.

## Semaine 2 - acquisition et faisabilité

- [x] scènes Landsat 8/9 réelles cataloguées ;
- [x] scène Sentinel-2 L2A et Copernicus DEM acquis ;
- [x] objets OpenStreetMap et météo de réanalyse acquis ;
- [x] identifiants, URL, dates et provenance enregistrés ;
- [x] faisabilité et volumes contrôlés sur une zone pilote.

## Semaine 3 - prétraitement

- [x] conversion Landsat ST_B10 en degrés Celsius ;
- [x] masque `QA_PIXEL` appliqué ;
- [x] masque SCL Sentinel-2 appliqué ;
- [x] reprojection et alignement sur une grille commune ;
- [x] NDVI, NDBI, imperméabilisation, distances, altitude et pente calculés.

## Semaine 4 - table pixel-date

- [x] 13 642 bâtiments et 8 899 voies OSM rasterisés à 10 m puis agrégés à 30 m ;
- [x] température et humidité ERA5-Land intégrées ;
- [x] vent et rayonnement ERA5 intégrés lorsque non exposés par ERA5-Land ;
- [x] table Parquet de 748 120 lignes, 14 dates et 53 690 pixels ;
- [x] 15 variables explicatives et une cible LST ;
- [x] aucun doublon pixel-date et aucune valeur manquante ;
- [x] dictionnaire, manifeste, empreintes et audit qualité générés.

## Semaine 5 - modélisation

- [x] baseline moyenne globale ;
- [x] baseline saisonnière mensuelle ;
- [x] régression linéaire ;
- [x] Random Forest ;
- [x] gradient boosting ;
- [x] XGBoost natif ;
- [x] séparation chronologique par dates complètes ;
- [x] modèle sérialisé, prédictions de validation et métriques ;
- [x] importance des variables par permutation ;
- [x] fiche du modèle réelle.

## Résultats de validation

| Modèle | MAE (°C) | RMSE (°C) | R² |
|---|---:|---:|---:|
| Baseline saisonnière mensuelle | 1,728 | 2,417 | -0,185 |
| Baseline moyenne globale | 2,611 | 3,246 | -1,138 |
| Random Forest | 2,944 | 3,472 | -1,446 |
| Gradient boosting | 3,110 | 3,656 | -1,711 |
| XGBoost | 4,510 | 5,247 | -4,586 |
| Régression linéaire | 5,005 | 5,396 | -4,908 |

La Random Forest est le meilleur candidat Machine Learning, mais elle ne bat pas
la baseline saisonnière. Ce résultat négatif est conservé honnêtement. Les deux
dates de test 2025 restent isolées et non scorées jusqu'à la semaine 6.

## Conclusion

Les tâches prévues pour les semaines 1 à 5 sont terminées à l'échelle de la zone
pilote. La semaine 6 doit maintenant analyser le décalage temporel, cartographier
les résidus et ouvrir une seule fois le test final.
