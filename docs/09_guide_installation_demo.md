# Semaine 8 - Guide d'installation, exécution et démonstration

## Prérequis

- Windows 10/11 ou Linux avec Python 3.10 à 3.12 ;
- environ 8 Go d'espace disque pour conserver les données réelles et les sorties ;
- accès Internet uniquement pour actualiser les données ou les prévisions météo.

## Installation PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-real.txt
```

## Exécution complète

```powershell
.\scripts\run_complete.ps1
```

La commande vérifie d'abord les données réelles des semaines 1 à 5, réutilise
l'évaluation finale déjà verrouillée, actualise les prévisions J+1/J+2, génère les
deux rapports, contrôle toute la livraison et exécute les tests automatisés.

Les étapes peuvent aussi être lancées séparément :

```powershell
.\.venv\Scripts\python.exe -m geoai_rabat.cli evaluate-final --config configs/rabat_real_pilot.json
.\.venv\Scripts\python.exe -m geoai_rabat.cli forecast-two-days --config configs/rabat_real_pilot.json
.\.venv\Scripts\python.exe -m geoai_rabat.cli verify-final --config configs/rabat_real_pilot.json
.\.venv\Scripts\python.exe -m geoai_rabat.cli verify-delivery --config configs/rabat_real_pilot.json
.\.venv\Scripts\python.exe -m geoai_rabat.cli report-final --config configs/rabat_real_pilot.json
.\.venv\Scripts\python.exe -m geoai_rabat.cli report-defense --config configs/rabat_real_pilot.json
.\.venv\Scripts\python.exe -m pytest
```

## Scénario de démonstration (5 minutes)

1. Afficher `artifacts/real_week5/dataset_audit_real.json` pour prouver le volume,
   les dates et l'origine réelle des données.
2. Afficher `artifacts/final/test_metrics_final.json` et expliquer que les deux dates
   2025 n'ont été ouvertes qu'après le gel du modèle.
3. Comparer les deux cartes `residual_test_*.png` avec leur échelle commune.
4. Lancer `forecast-two-days`, puis ouvrir les deux nouvelles cartes et le CSV des
   points chauds.
5. Terminer avec `verify-final` et `pytest` pour montrer la reproductibilité.

## Principaux livrables

- table pixel-date : `data/processed/real/pixel_date_real.parquet` ;
- modèle figé : `artifacts/real_week5/model_real_week5.pkl` ;
- métriques et audit : `artifacts/real_week5/` et `artifacts/final/` ;
- rasters : `output/rasters/` ;
- rapport : `output/pdf/rapport_final_geoai_rabat.pdf` ;
- guide de soutenance : `output/pdf/guide_complet_soutenance_geoai_rabat.pdf` ;
- présentation : `output/presentation/presentation_finale_geoai_rabat.pptx`.

## Reproductibilité et traçabilité

Les manifestes enregistrent les URL, horodatages et empreintes SHA-256 des sources.
Les partitions sont faites par dates entières. Le modèle utilisé pour le test et la
prévision est celui figé en semaine 5. Les GeoTIFF conservent la projection, la
résolution, l'emprise et la valeur nodata. Pour une nouvelle campagne de données,
conserver les anciens artefacts dans un répertoire versionné avant de relancer
`run-real`.
