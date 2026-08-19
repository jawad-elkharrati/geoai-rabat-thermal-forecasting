# Semaine 1-2 - Inventaire des sources

Le registre machine lisible est dans `data/raw/source_inventory.csv`.

## Landsat 8/9 Collection 2 Level-2

- rôle : cible LST et masque `QA_PIXEL` ;
- collection STAC : `landsat-c2l2-st` ;
- facteur LST officiel : `0.00341802`, offset `149.0 K` ;
- accès : LandsatLook STAC puis Cloud Optimized GeoTIFF ;
- référence : https://www.usgs.gov/landsat-missions/spatiotemporal-asset-catalog-stac

## Sentinel-2 Level-2A

- rôle : NDVI, NDWI/MNDWI, NDBI et état récent de la surface ;
- bandes : B02, B03, B04, B08, B11 et SCL ;
- les bandes à 10 et 20 m sont agrégées sur la grille cible de 30 m ;
- référence : https://documentation.dataspace.copernicus.eu/Data/SentinelMissions/Sentinel2.html

## OpenStreetMap

- rôle : limite, bâtiments, routes, parcs et eau ;
- limite : Nominatim, objets thématiques : Overpass ;
- attribution requise : © OpenStreetMap contributors ;
- termes : https://www.openstreetmap.org/copyright

## Copernicus DEM GLO-30

- rôle : altitude et pente ;
- résolution native proche de 30 m ;
- référence : https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM

## ERA5-Land

- rôle : historique cohérent de température, humidité, vent et rayonnement ;
- résolution : environ 0,1 degré, horaire ;
- acquisition réelle : Open-Meteo Historical API avec `models=era5_land` ;
- complément : ERA5 pour le vent et le rayonnement non exposés par cette passerelle ;
- référence : https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land

## Open-Meteo

- rôle : prévisions futures J+1/J+2 et échantillon de faisabilité ;
- variables : `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`,
  `shortwave_radiation` ;
- référence : https://open-meteo.com/en/docs

## Traçabilité attendue pour chaque fichier

`source_id`, URL exacte, identifiant de scène, date d'acquisition, date de
téléchargement UTC, licence/termes, taille, SHA-256, emprise, CRS, résolution,
pourcentage de données valides et étape de transformation.
