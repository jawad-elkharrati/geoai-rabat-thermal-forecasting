# Model card finale - GeoAI Rabat

## Usage

- Cible : température diurne de surface terrestre (LST), en °C.
- Périmètre : zone pilote de Rabat, grille de 30 m en EPSG:32629.
- Usage prévu : preuve de concept, comparaison spatiale et cartes J+1/J+2.
- Usage interdit : température de l'air, ressenti humain ou alerte sanitaire officielle.

## Modèle

- Modèle figé : `random_forest`.
- Variables : 15 variables explicatives.
- Sélection : dates complètes de validation, sans utiliser le test final.
- Test final ouvert une seule fois : 2025-06-15, 2025-07-25.

## Résultats sur 106,819 observations de test

| Méthode | MAE (°C) | RMSE (°C) | R² | Biais (°C) |
|---|---:|---:|---:|---:|
| Random Forest figée | 1.416 | 1.831 | 0.237 | +0.738 |
| Baseline saisonnière | 1.497 | 2.071 | 0.024 | -0.458 |

La Random Forest bat la baseline sur le test final en MAE et RMSE. Son R² reste
modeste et son biais est positif : le modèle doit être étendu et surveillé avant
toute mise en production.

## Limites et surveillance

- zone pilote et seulement 14 dates Landsat ;
- variables Sentinel-2 et urbaines considérées statiques pour J+1/J+2 ;
- météo future extraite au point central de la zone ;
- vérifier la dérive, le biais et la qualité des entrées à chaque nouvelle campagne ;
- conserver le modèle, la réponse météo, la date de génération et les cartes.
