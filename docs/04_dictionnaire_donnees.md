# Semaine 4 - Dictionnaire de la table pixel-date

La version machine lisible est générée dans
`data/processed/data_dictionary.csv`.

| Variable | Type | Unité | Rôle | Source |
|---|---|---|---|---|
| `pixel_id` | string | - | identifiant | grille |
| `date` | date | - | groupe temporel | Landsat |
| `x_m`, `y_m` | float | m | géolocalisation | grille UTM |
| `lon`, `lat` | float | degré | géolocalisation | grille WGS84 |
| `elevation_m` | float | m | feature statique | Copernicus DEM |
| `slope_deg` | float | degré | feature statique | DEM dérivé |
| `ndvi` | float | indice | feature | Sentinel-2 |
| `ndbi` | float | indice | feature | Sentinel-2 |
| `building_density` | float | 0-1 | feature | OSM |
| `road_density` | float | 0-1 | feature | OSM |
| `impervious_fraction` | float | 0-1 | feature | Sentinel-2/OSM |
| `distance_to_water_m` | float | m | feature | Sentinel-2, masque MNDWI |
| `distance_to_green_m` | float | m | feature | Sentinel-2, masque NDVI |
| `air_temperature_c` | float | °C | feature météo | ERA5-Land |
| `relative_humidity_pct` | float | % | feature météo | ERA5-Land |
| `wind_speed_ms` | float | m/s | feature météo | ERA5 |
| `shortwave_radiation_wm2` | float | W/m² | feature météo | ERA5 |
| `day_of_year_sin/cos` | float | - | feature temporelle | date |
| `lst_c` | float | °C | cible | Landsat ST |

## Conventions

- les densités OSM sont rasterisées à 10 m puis agrégées sur la grille de 30 m ;
- les distances sont calculées dans `EPSG:32629`, jamais en degrés ;
- `NDVI = (NIR - Red) / (NIR + Red)` ;
- `NDBI = (SWIR - NIR) / (SWIR + NIR)` ;
- les valeurs manquantes de la cible sont supprimées, jamais imputées ;
- toute imputation de feature est apprise uniquement sur l'entraînement.

## Pipeline des couches réelles

Le module `geoai_rabat.geoprocessing` construit une grille alignée sur un multiple de
30 m à partir de la limite acquise, reprojette les rasters par méthode explicite,
rasterise les objets OSM et calcule les distances en mètres. Les variables continues
utilisent `bilinear` ou `average`; les masques QA et classes utilisent `nearest`.
