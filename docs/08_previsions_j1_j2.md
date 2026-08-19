# Semaine 7 - Prévisions automatisées J+1 et J+2

La commande `forecast-two-days` interroge l'API Open-Meteo pour Rabat, extrait les
variables météorologiques à 11 h locales, reconstruit les 15 variables explicatives
dans l'ordre du modèle figé, puis produit une carte GeoTIFF et une carte PNG pour
chacun des deux jours suivants.

L'exécution de livraison a généré les prévisions des 11 et 12 août 2026 sur 53 690
pixels valides, avec une grille commune de 30 m en EPSG:32629.

| Date | Air (°C) | HR (%) | Vent (m/s) | Rayonnement (W/m²) | LST moyenne (°C) |
|---|---:|---:|---:|---:|---:|
| 2026-08-11 | 23,6 | 71 | 1,50 | 561 | 38,10 |
| 2026-08-12 | 23,9 | 75 | 2,16 | 566 | 38,31 |

Les deux PNG utilisent une échelle colorimétrique identique (36,66 à 40,58 °C).
Le fichier `forecast_hotspots_top100.csv` fournit, pour chaque date, les 100 pixels
les plus chauds avec identifiant, coordonnées projetées et température prévue.

## Limites d'usage

- la sortie estime la température de surface diurne, pas la température de l'air ;
- la météo provient d'un point central et non d'un champ météorologique local ;
- l'occupation du sol et les autres variables statiques sont supposées inchangées ;
- les cartes sont une démonstration pilote et non une alerte sanitaire officielle ;
- une exploitation régulière doit journaliser la version du modèle et archiver la
  réponse météo brute.
