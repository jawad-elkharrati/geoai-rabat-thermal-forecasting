# Audit final des semaines 1 à 8

Date de contrôle : 10 août 2026.

| Semaine | Exigence contrôlée | Statut | Preuve |
|---|---|---|---|
| 1 | Zone, cible, résolution et projection | Conforme | `docs/01_cadrage.md`, grille EPSG:32629 à 30 m |
| 2 | Sources satellitaires, urbaines, relief et météo | Conforme | inventaires et manifestes dans `data/raw/real/` et `data/processed/real/` |
| 3 | Masques qualité, reprojection et alignement | Conforme | 15 rasters alignés selon `verify-real` |
| 4 | Table pixel-date et dictionnaire | Conforme | 748 120 lignes, 14 dates, 53 690 pixels, 15 variables |
| 5 | Baselines et modèles candidats | Conforme | 6 méthodes évaluées, Random Forest ML retenue et figée |
| 6 | Test sur dates jamais vues et analyse des résidus | Conforme | 106 819 lignes, 2 dates, métriques et 2 cartes de résidus |
| 7 | Prévisions automatiques J+1/J+2 | Conforme | 2 GeoTIFF, 2 PNG, météo archivée et Top 100 hotspots/date |
| 8 | Tests, documentation, rapport et présentation | Conforme | 16 tests réussis, 10 documents, PDF final, guide de soutenance et PPTX |

## Contrôles exécutés

```text
pytest : 16 passed
verify-real : status=ok, aligned_rasters=15
verify-final : status=ok, test_dates_scored=2, forecast_rasters=2
slides_test.py : Test passed. No overflow detected.
rapport PDF : 6 pages rendues et inspectées visuellement
présentation : 9 diapositives rendues et inspectées visuellement
guide de soutenance : 15 pages et 22 questions-réponses vérifiées
```

Les avertissements pytest restants proviennent d'une dépréciation future dans la
dépendance `planetary_computer` et d'un refus d'écriture du cache `.pytest_cache` ;
ils ne modifient ni les résultats ni les fichiers livrés.

## Décision

Le projet demandé est complet à l'échelle pilote définie dans le cahier des
charges. Les limites scientifiques et opérationnelles sont documentées ; aucune
affirmation de couverture de toute la commune ou d'alerte sanitaire n'est faite.
