# Semaine 6 - Validation finale sur dates jamais vues

Le modèle Random Forest figé en semaine 5 a été évalué une seule fois sur les deux
dates de test réservées : le 15 juin 2025 et le 25 juillet 2025. Le marqueur
`artifacts/final/test_evaluation_marker.json` conserve la date d'ouverture et
l'empreinte du modèle. Aucun réentraînement n'a été effectué après cette ouverture.

## Résultats globaux

| Modèle | MAE (°C) | RMSE (°C) | R² | Biais (°C) |
|---|---:|---:|---:|---:|
| Random Forest figée | 1,416 | 1,831 | 0,237 | +0,738 |
| Baseline saisonnière mensuelle | 1,497 | 2,071 | 0,024 | -0,458 |

La Random Forest gagne donc le test final en MAE et en RMSE. Elle explique une
partie modeste de la variabilité spatiale et présente un biais chaud moyen de
0,74 °C. Ce résultat est meilleur que la baseline, mais reste celui d'un pilote
spatial limité et ne doit pas être interprété comme une validation opérationnelle
sur toute la commune.

## Résultats par date

| Date | Lignes | MAE RF (°C) | RMSE RF (°C) | R² RF |
|---|---:|---:|---:|---:|
| 2025-06-15 | 53 438 | 1,363 | 1,777 | 0,215 |
| 2025-07-25 | 53 381 | 1,469 | 1,884 | 0,142 |

Les cartes de résidus utilisent la convention `observé - prédit`. Les valeurs
positives indiquent donc une sous-estimation du modèle, et les négatives une
surestimation. Les deux cartes partagent la même échelle symétrique afin de rendre
leur comparaison honnête.

## Livrables

- métriques : `artifacts/final/test_metrics_final.json` ;
- prédictions pixel-date : `artifacts/final/test_predictions_final.parquet` ;
- résidus géoréférencés : `output/rasters/residual_test_*.tif` ;
- cartes de diagnostic : `reports/figures/final/residual_test_*.png`.
