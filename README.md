# GeoAI Rabat

Pipeline géospatial pour cartographier et prévoir la température de surface terrestre à Rabat à partir de données satellitaires, urbaines et météorologiques.

## Objectif

Le projet construit une grille de 30 mètres sur une zone pilote de Rabat, rassemble les variables utiles pour chaque pixel et chaque date, puis entraîne plusieurs modèles de régression. Le modèle retenu produit des cartes de température de surface à J+1 et J+2.

La variable cible est la **LST** (*Land Surface Temperature*), c’est-à-dire la température de la surface observée par satellite. Elle ne doit pas être confondue avec la température de l’air mesurée par une station météo.

## Données utilisées

- Landsat Collection 2 Level-2 pour la LST et le masque qualité ;
- Sentinel-2 pour les indices de végétation et de bâti ;
- Copernicus DEM pour l’altitude et la pente ;
- OpenStreetMap pour les densités de bâtiments et de routes ;
- ERA5-Land et Open-Meteo pour les variables météorologiques.

Toutes les couches sont reprojetées en `EPSG:32629` et alignées sur la même grille avant la construction de la table pixel-date.

## Résultats du pilote

| Indicateur | Valeur |
|---|---:|
| Observations pixel-date | 748 120 |
| Pixels uniques | 53 690 |
| Dates Landsat | 14 |
| Variables explicatives | 15 |
| Observations du test final | 106 819 |
| MAE du Random Forest | 1,416 °C |
| RMSE du Random Forest | 1,831 °C |
| R² du Random Forest | 0,237 |

La séparation est temporelle : les dates de test ne sont jamais utilisées pendant l’entraînement ni pendant le choix du modèle. Sur ce pilote, le Random Forest obtient une MAE de 1,416 °C sur le test final, contre 1,497 °C pour la baseline saisonnière.

## Organisation du dépôt

```text
configs/                 paramètres du pipeline
src/geoai_rabat/         collecte, préparation, modélisation et prévision
tests/                   tests unitaires et d’intégration
data/raw/samples/        petits échantillons reproductibles
data/processed/          dictionnaire et échantillon CSV de démonstration
artifacts/               métriques, schémas et fiches modèles
docs/                    notes méthodologiques détaillées
```

Les données satellitaires brutes, les rasters intermédiaires, les tables Parquet, les modèles sérialisés et les cartes générées ne sont pas versionnés. Ils sont volumineux et sont reconstruits localement par le pipeline.

## Installation

Python 3.10 ou une version plus récente est nécessaire.

```powershell
git clone https://github.com/jawad-elkharrati/geoai-rabat-thermal-forecasting.git
cd geoai-rabat-thermal-forecasting
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-real.txt
```

## Exécution

Pour lancer le pipeline complet sous Windows :

```powershell
.\scripts\run_complete.ps1
```

Les principales vérifications peuvent aussi être exécutées séparément :

```powershell
python -m pytest -m "not integration"
python -m geoai_rabat.cli verify-real --config configs/rabat_real_pilot.json
python -m geoai_rabat.cli verify-final --config configs/rabat_real_pilot.json
python -m geoai_rabat.cli verify-delivery --config configs/rabat_real_pilot.json
```

La première exécution du pipeline réel télécharge les données nécessaires et peut prendre du temps selon la connexion et la machine utilisée.
Les deux tests marqués `integration` contrôlent la livraison locale complète et nécessitent les données et modèles générés. Ils peuvent être lancés après le pipeline avec `python -m pytest -m integration`.

## Livrables

- [Fiche du modèle final](artifacts/final/MODEL_CARD_FINAL.md)
- [Résultats du test final](artifacts/final/test_metrics_final.json)
- [Documentation méthodologique](docs/)

Après l’exécution, les prévisions GeoTIFF sont écrites dans `output/rasters/` et les cartes PNG dans `reports/figures/`.

## Limites

Il s’agit d’une étude pilote sur une partie de Rabat. Les variables Sentinel-2 et urbaines sont considérées comme statiques, tandis que la météo prévisionnelle est extraite au point central de la zone. Les cartes produites servent à l’analyse géospatiale et ne constituent pas un système officiel d’alerte sanitaire.
