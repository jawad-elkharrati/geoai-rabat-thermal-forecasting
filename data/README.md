# Données

Ce dossier suit une séparation stricte :

- `raw/` : métadonnées et échantillons bruts immuables ;
- `interim/` : couches nettoyées ou alignées ;
- `processed/` : table pixel-date prête pour le Machine Learning.

Les gros rasters et les sorties générées ne sont pas versionnés. Le fichier
`raw/source_inventory.csv` est le registre de traçabilité des sources.

Le mode démonstration génère des données **synthétiques, spatialement structurées et
clairement identifiées**. Il sert à valider le pipeline en attendant l'acquisition du
volume historique réel.
