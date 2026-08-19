# Semaine 1 - Cadrage et décisions

## Question traitée

Comment transformer des données géospatiales et météorologiques ouvertes en
prédictions fiables de la température diurne de surface, à l'échelle d'un pixel de
30 m, pour la commune de Rabat ?

## Décisions validées

| Élément | Décision |
|---|---|
| Zone | Commune de Rabat, Maroc |
| Limite de référence | Relation OpenStreetMap `2799215`, figée avec date d'acquisition |
| Emprise de recherche | `[-6.93, 33.89, -6.73, 34.06]` en WGS84 |
| Projection de travail | WGS 84 / UTM zone 29N, `EPSG:32629` |
| Résolution | 30 m |
| Unité d'observation | une cellule de grille pour une date Landsat |
| Cible | Landsat Collection 2 Level-2 Surface Temperature, en °C |
| Période | 2019-2025 |
| Heure d'intérêt | passage diurne Landsat ; météo agrégée autour du passage |
| Validation | dates complètes, jamais lignes aléatoires |
| Dates finales | deux dernières dates exploitables réservées et non évaluées en semaine 5 |

La géométrie acquise couvre environ **115,77 km²**. À 30 m, cela correspond à
environ **128 628 cellules** avant les ajustements de bord, ce qui confirme
l'estimation de 120 000 à 140 000 cellules du rapport.

## Définition de la cible

Conversion Landsat :

```text
LST_K = DN * 0.00341802 + 149.0
LST_C = LST_K - 273.15
```

La LST décrit la surface (toiture, route, sol, végétation). Elle ne doit pas être
présentée comme la température de l'air ni comme une mesure du confort thermique.

## Critères de qualité

- pixels `QA_PIXEL` affectés par remplissage, nuage dilaté, cirrus, nuage, ombre ou
  neige exclus ;
- scènes avec plus de 35 % de couverture nuageuse écartées au catalogue ;
- cible conservée entre 10 et 70 °C pour les contrôles initiaux ;
- toutes les couches alignées sur la même origine, résolution, forme et projection ;
- chaque source enregistrée avec URL, licence, acquisition et empreinte de fichier.

## Hors périmètre avant stabilisation

Deep Learning, tableau de bord, service temps réel et orchestration cloud complète.
Ces extensions ne doivent pas retarder la construction du dataset et la validation
du modèle tabulaire.
