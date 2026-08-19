# Semaine 2 - Rapport de faisabilité

## Conclusion

La preuve de concept est faisable avec les sources ouvertes prévues. Le principal
risque n'est pas algorithmique : il vient du volume, de la disponibilité de la LST
sur certaines scènes et des nuages.

## Vérifications réalisées dans le dépôt

- limite de Rabat acquise via Nominatim : relation OSM `2799215`, géométrie
  `MultiPolygon`, avec licence ODbL enregistrée ;
- météo horaire Open-Meteo acquise du 1 au 7 juillet 2024 ;
- catalogue Landsat ST interrogé sur l'été 2024 : 20 scènes retournées dans
  l'échantillon, avec identifiants, dates, couverture nuageuse et clés d'assets ;
- implémentation et tests du masque QA Landsat ;
- conversion DN vers °C ;
- grille de référence et identifiants de pixels stables ;
- pipeline de démonstration produisant plusieurs années et dates nuageuses ;
- contrôle de schéma, valeurs manquantes et séparation temporelle.

Les réponses brutes et leurs empreintes SHA-256 sont conservées sous
`data/raw/samples/`. Leur manifeste permet de prouver la date et le résultat de
l'acquisition.

## Ordres de grandeur

| Élément | Estimation |
|---|---:|
| Cellules 30 m | 120 000 à 140 000 |
| Dates exploitables | 50 à 100 |
| Lignes après qualité | 3 à 8 millions |
| Variables explicatives | 15 à 30 |

Le calcul sur la limite OSM réellement acquise donne 115,77 km², soit environ
128 628 cellules de 900 m². L'ordre de grandeur du rapport est donc confirmé.

Le format Parquet, le typage `float32` et le traitement date par date limitent la
mémoire. Les premiers essais peuvent utiliser un échantillonnage spatial stratifié,
mais jamais un échantillonnage qui mélange les dates de validation.

## Risques et parades

| Risque | Parade |
|---|---|
| Nuages / lacunes LST | bits QA, seuil de scène, plusieurs années |
| Résolutions différentes | grille 30 m unique, règles d'agrégation explicites |
| Fuite temporelle | partitions par dates complètes |
| Variables urbaines datées | enregistrer la date du snapshot OSM/Sentinel |
| Météo trop grossière | ne pas lui attribuer une variabilité spatiale artificielle |
| Volume local | Parquet, colonnes `float32`, batchs par date |
| Accès API | cache brut, manifeste, reprise et mode démonstration |

## Limite importante

Le dataset de démonstration est synthétique. Ses métriques prouvent que le code et le
protocole fonctionnent ; elles ne constituent pas une validation scientifique sur
des observations réelles de Rabat.
