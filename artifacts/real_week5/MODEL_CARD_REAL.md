# Model card - semaine 5 réelle

- Meilleur modèle candidat : **random_forest**
- Meilleur résultat global : **monthly_seasonal_baseline**
- MAE validation : **2.944 °C**
- RMSE validation : **3.472 °C**
- R² validation : **-1.446**
- Validation : dates complètes, sans mélange de pixels entre partitions.
- Test final : deux dates 2025 réservées et non scorées.
- Données : observations Landsat réelles, ERA5-Land et variables urbaines OSM.
- Gain du candidat contre la meilleure baseline : **-70.40 %**.
- Limite : entraînement réalisé sur la zone pilote, pas encore toute la commune.
