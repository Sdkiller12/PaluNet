# PaluNet — Rapport de tests et observations

**Modèle évalué :** v1.0.0 (MobileNetV2, ONNX, 10,18 Mo)
**Date du rapport :** 8 octobre 2026
**Environnement :** Python 3.11.9, Windows 11, Docker Desktop 29.7.2

---

## 1. Synthèse

| Volet | Résultat | Critère | Statut |
| --- | --- | --- | --- |
| Tests automatisés (`pytest`) | 84 passés, 0 échec, 0 ignoré | 100 % verts | ✅ |
| Couverture de code (`src/`) | 98,2 % | ≥ 70 % | ✅ |
| Accuracy (test) | 0,9623 | ≥ 0,95 | ✅ |
| Rappel / sensibilité (test) | 0,9694 | ≥ 0,96 | ✅ |
| AUC (test) | 0,9932 | ≥ 0,97 | ✅ |
| Non-régression du modèle | 4/4 tests passés | aucune baisse > 0,005 | ✅ |
| Latence API P95 (charge) | 67 ms | ≤ 500 ms | ✅ |
| Échecs sous charge | 0 / 2241 requêtes | 0 | ✅ |
| Validation externe (A6/A7) | non réalisée | ≥ 200 images d'un labo partenaire | ⏸ reportée |

Tous les critères d'acceptation du cahier des charges sont atteints, sauf la validation
externe, reportée volontairement (voir §8). Les performances du modèle sont donc des
**résultats de validation interne** : jeu NIH, découpage par patient.

---

## 2. Stratégie de test

| Niveau | Emplacement | Objet | Dépendances | Exécution |
| --- | --- | --- | --- | --- |
| Unitaire | `tests/unit/` | Fonctions isolées : prétraitement, split, métriques, inférence, artefacts, logs | numpy, PIL | CI, à chaque push |
| Intégration | `tests/integration/` | API FastAPI de bout en bout (`TestClient` + `MockPredictor`) | FastAPI | CI, à chaque push |
| Non-régression | `tests/model/` | Vrai modèle ONNX sur le jeu de test, comparé à la baseline | modèle, baseline, `test.npz` | Local, si les artefacts existent |
| Charge | `tests/load/` | Latence et robustesse de l'API conteneurisée | API démarrée, Locust | Manuel |
| Évaluation | `src/models/evaluate.py` | Métriques finales et écriture de la baseline | TensorFlow / ONNX | Pipeline d'entraînement |

Aucun test de la CI ne dépend de TensorFlow ni du vrai modèle : l'API est testée avec un
prédicteur factice déterministe et des cellules synthétiques (`tests/conftest.py`).

Commandes :

```bash
pytest --cov                                   # unitaires + intégration + non-régression
locust -f tests/load/locustfile.py --host http://localhost:8000 \
       --headless -u 20 -r 5 -t 2m --csv reports/load
```

---

## 3. Tests unitaires (54 tests)

| Fichier | Tests | Ce qui est vérifié |
| --- | --- | --- |
| `test_preprocessing.py` | 15 | Forme, type et plage de sortie. **Égalité exacte** entre le cache d'entraînement et le prétraitement de l'API. Valeurs de référence figées. PNG et JPEG acceptés. Autres formats, données corrompues et PNG tronqués refusés. RGBA composité sur fond noir, niveaux de gris convertis en RGB. Contrôle qualité (image trop petite ou uniforme). Le format réel prime sur l'extension. |
| `test_split.py` | 6 | Extraction de l'identifiant patient. **Aucun patient dans deux splits.** Proportions et équilibre des classes. Reproductibilité selon la graine. Fractions dont la somme ≠ 1 refusées. Aller-retour du manifeste. |
| `test_metrics.py` | 7 | AUC parfaite, aléatoire et inversée. AUC égale à la définition par paires. AUC `nan` sur une seule classe. Matrice de confusion. Le seuil choisi atteint le rappel cible avec la meilleure précision. Contrôles de baseline et d'acceptation. |
| `test_inference.py` | 13 | `class_map` figée et validée. Probabilité positive correcte quel que soit l'encodage. Décision avec le **seuil calibré, pas 0,5**. Confiance relative à la classe prédite. `OnnxPredictor` avec une fausse session (avec ou sans carte d'activation). Rendu de la heatmap. |
| `test_artifacts.py` | 8 | Chemins versionnés des artefacts. Version semver obligatoire. Fiche `.json` à côté du `.onnx`. Écriture et lecture JSON (dossiers créés, accents conservés). Fusion des métadonnées. |
| `test_logger.py` | 5 | Logs JSON (message, niveau, champs supplémentaires, exceptions). Un seul handler après reconfiguration. Niveau via `LOG_LEVEL`. Journal des prédictions à rotation quotidienne avec rétention. |

---

## 4. Tests d'intégration de l'API (26 tests)

| Thème | Comportement vérifié |
| --- | --- |
| Santé | `/health` respecte son contrat. Si le modèle manque, `/health` et `/v1/*` répondent 503. |
| Prédiction | Contrat de réponse. Confiance relative à la classe prédite. Heatmap à la demande. JPEG accepté. Avertissements qualité renvoyés. |
| Erreurs | Fichier invalide → 400. Format non supporté → 400. Fichier trop gros → 413. Fichier manquant → 422. |
| Sécurité | Clé API exigée si configurée. Plusieurs clés acceptées. `/health` reste public. Quota **par clé** : la 4ᵉ requête à 3/minute renvoie 429, une autre clé passe. |
| Traitement par lot | Échec partiel : résultats et erreurs renvoyés ensemble. Plus de 32 fichiers → 400. |
| Observabilité | Journal anonymisé (ni nom de fichier ni IP). `/metrics` expose le total, le taux d'erreur, le P95 et la version. |
| Chargement du modèle | Lecture de la fiche et surcharges par variables d'environnement. Refus si `class_map` inversée, seuil absent ou hors ]0, 1[, ou fiche manquante. Mode mock. |

---

## 5. Couverture de code

| Module | Couverture |
| --- | --- |
| `src/api/dependencies.py` | 96 % |
| `src/api/main.py` | 95 % |
| `src/api/service.py` | 99 % |
| `src/data/preprocessing.py` | 96 % |
| `src/data/split.py` | 97 % |
| `src/models/inference.py` | 97 % |
| `src/models/metrics.py` | 98 % |
| `src/models/artifacts.py` | 100 % (76 % avant ajout de `test_artifacts.py`) |
| `src/utils/logger.py` | 100 % (80 % avant ajout de `test_logger.py`) |
| 12 autres fichiers | 100 % |
| **Total** | **98,2 %** (726 instructions, 13 non couvertes) |

Sont exclus de la mesure les scripts hors ligne qui dépendent de TensorFlow ou des données
réelles : `download`, `prepare`, `train`, `model`, `export`, `evaluate`,
`calibrate_threshold`. Ils sont validés par l'exécution du pipeline et par les tests de
non-régression.

---

## 6. Évaluation du modèle

### 6.1 Données

Le jeu est le NIH Malaria Cell Images (27 558 images). Il est découpé **par patient** et
stratifié, pour qu'aucun patient ne serve à la fois à l'entraînement et à l'évaluation.

| Split | Images | Patients | Saines | Parasitées |
| --- | --- | --- | --- | --- |
| Entraînement | 19 195 | 141 | 9 627 | 9 568 |
| Validation | 4 173 | 30 | 2 085 | 2 088 |
| Test | 4 190 | 30 | 2 067 | 2 123 |

### 6.2 Entraînement

| Run | Données | Époques (phase 1 + 2) | LR phase 2 | Val. accuracy @0,5 | Val. rappel @0,5 | Val. AUC | Durée |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (essai rapide) | 2 000 / 500 | 1 + 1 | 1e-5 | 0,9000 | 0,8538 | 0,9680 | 31 s |
| 2 | complet | 15 + 10 | 1e-5 | 0,9535 | 0,9377 | 0,9876 | 19 min |
| **3 (retenu)** | complet | 15 + 30 max | 2e-5 | **0,9674** | **0,9646** | **0,9937** | 31 min |

Le transfert d'apprentissage se fait en deux phases : tête seule, puis dégel des 100
dernières couches. Pour le run 3, l'arrêt anticipé (`val_loss`, patience 5) a stoppé la
phase 2 après 15 époques sur 30.

### 6.3 Calibration du seuil

La règle est : précision maximale sous la contrainte rappel ≥ 0,98, sur le jeu de
validation. Le seuil retenu est **0,3221**. Sur la validation, il donne un rappel de
0,9804, une précision de 0,9508 et une spécificité de 0,9492.

Extrait de la courbe sur la validation :

| Seuil | Rappel | Précision | Spécificité |
| --- | --- | --- | --- |
| 0,2 | 0,9866 | 0,9196 | 0,9137 |
| 0,3 | 0,9808 | 0,9473 | 0,9453 |
| 0,4 | 0,9732 | 0,9589 | 0,9583 |
| 0,5 | 0,9646 | 0,9701 | 0,9703 |
| 0,7 | 0,9392 | 0,9844 | 0,9851 |

### 6.4 Résultats sur le jeu de test (4 190 images, seuil 0,3221)

| Métrique | Valeur | IC 95 % (binomial, indicatif) |
| --- | --- | --- |
| Accuracy | 0,9623 | ≈ [0,957 ; 0,968] |
| Rappel (sensibilité) | 0,9694 | ≈ [0,962 ; 0,977] |
| Précision | 0,9568 | |
| Spécificité | 0,9550 | |
| F1 | 0,9630 | |
| AUC | 0,9932 | |

Matrice de confusion :

| | Prédit parasitée | Prédit saine |
| --- | --- | --- |
| **Réel parasitée** (2 123) | VP = 2 058 | FN = 65 |
| **Réel saine** (2 067) | FP = 93 | VN = 1 974 |

La baseline est enregistrée dans `metrics_baseline/1.0.0.json` (`acceptance_passed: true`).

### 6.5 Non-régression (4 tests)

- Aucune baisse d'accuracy, de rappel ou d'AUC au-delà de 0,005 par rapport à la baseline.
- Pas d'inversion de classe : score moyen > 0,8 sur les parasitées et < 0,2 sur les saines.
- Taille de l'ONNX < 20 Mo (10,18 Mo mesurés).
- `class_map` de la fiche identique à la configuration, seuil dans ]0, 1[.

---

## 7. Test de charge

### 7.1 Conditions

- Conteneur Docker de production, limité à **2 vCPU / 2 Go** (`docker-compose.yml`).
- Vrai modèle ONNX v1.0.0, 1 worker uvicorn.
- 200 vraies images NIH tirées au hasard.
- 20 utilisateurs simulés, montée de 5 utilisateurs par seconde, durée 2 minutes.
- Chaque utilisateur attend 0,5 à 1,5 s entre deux requêtes, avec 10 prédictions pour 1 appel `/health`.
- Quota relevé à `RATE_LIMIT=100000/minute` pour ce test. Sinon, la limite de 60/minute
  par clé produit des 429.

### 7.2 Résultats

| Endpoint | Requêtes | Échecs | Médiane | P95 | P99 | Max | Débit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `POST /v1/predict` | 2 062 | 0 | 60 ms | **67 ms** | 72 ms | 173 ms | 17,4 req/s |
| `GET /health` | 179 | 0 | 5 ms | 8 ms | 12 ms | 30 ms | 1,5 req/s |

SLO respecté : P95 de 67 ms pour une cible ≤ 500 ms, zéro échec. Les CSV bruts sont dans
`reports/load_*.csv`.

---

## 8. Observations

### Performances du modèle

1. **Tous les critères d'acceptation sont atteints**, mais avec des marges différentes.
   L'AUC (0,993) dépasse nettement le minimum de 0,97. En revanche, le rappel (0,969) n'est
   qu'à 0,009 au-dessus du minimum de 0,96, soit à peine plus que l'incertitude statistique
   (± 0,007).
2. **Le rappel baisse entre la validation et le test** : 0,980 en validation contre 0,969
   en test, avec le même seuil. La cible de 0,98 a été calibrée sur la validation et ne se
   transfère pas entièrement au test. Cet écart est attendu, car les deux jeux ne comptent
   que 30 patients chacun et la variabilité entre patients pèse lourd.
3. **65 cellules parasitées sont manquées** (3,1 % des cellules infectées). C'est l'erreur
   la plus grave en contexte clinique. Le diagnostic réel porte sur un frottis de
   nombreuses cellules, pas sur une seule : l'impact au niveau du patient reste à mesurer.
4. **Le seuil de 0,32 privilégie la sensibilité.** À 0,5, la spécificité serait d'environ
   0,970 en validation, contre 0,949 au seuil retenu. Le coût est d'environ 2 points de
   faux positifs en plus (93 FP en test). Ce compromis est voulu : un faux positif entraîne
   une relecture, un faux négatif un paludisme non traité.
5. **Pas de surapprentissage visible** : en fin d'entraînement, l'accuracy est de 0,967 en
   entraînement et de 0,964 en validation.
6. **L'entraînement long a été déterminant.** Entre les runs 2 et 3, le learning rate de la
   phase 2 doublé et la phase 2 allongée font passer l'AUC de validation de 0,988 à 0,994 et
   le rappel à 0,5 de 0,938 à 0,965. L'arrêt anticipé a coupé la phase 2 à 15 époques.

### API et infrastructure

7. **La latence a une marge importante** : le P95 (67 ms) est environ 7 fois plus faible
   que la cible. La distribution est resserrée (médiane 60 ms, P99 72 ms). Le seul pic
   (173 ms) correspond probablement à la montée en charge initiale.
8. **Le débit mesuré n'est pas un débit maximal.** Les 17 req/s découlent du temps d'attente
   des 20 utilisateurs simulés, pas d'une saturation du serveur. La capacité maximale d'une
   instance n'a pas été recherchée.
9. **Le quota par défaut ne convient pas à un usage partagé.** À 60 requêtes par minute et
   par clé, une seule clé partagée par plusieurs postes atteint vite la limite. Il faut une
   clé par poste, ou un quota adapté au volume réel.
10. **Les tests garantissent la cohérence entre entraînement et production** : prétraitement
    identique au bit près, encodage de classe figé, détection d'une inversion de classe.
    C'est la principale source d'erreur silencieuse dans ce type de système.

### Limites

11. **Aucune validation externe.** Toutes les images viennent du même jeu (NIH, Bangladesh,
    même protocole de coloration et d'acquisition). Les performances sur des frottis de
    laboratoires locaux, avec d'autres microscopes et d'autres colorations, restent
    inconnues. La validation A6/A7 est reportée : elle demande le jeu d'un labo partenaire
    (≥ 200 images), l'avis d'un comité d'éthique et une déclaration à l'ARTCI. Elle est
    indispensable avant tout déploiement réel.
12. **Les intervalles de confiance sont optimistes.** Ils supposent des cellules
    indépendantes, alors que les cellules d'un même patient sont corrélées. Un IC calculé
    par patient (30 patients en test) serait plus large.
13. **Les tests d'intégration utilisent des images synthétiques et un prédicteur factice.**
    Ils valident le comportement de l'API, pas la qualité des prédictions. Celle-ci relève
    des tests de non-régression et de l'évaluation.
14. **Le test de charge a tourné sur une seule machine** (Windows, Docker Desktop), avec le
    client et le serveur sur le même hôte. Les latences réseau réelles ne sont pas incluses.

---

## 9. Recommandations

1. Mener la validation externe A6/A7 avant tout usage clinique, et recalibrer le seuil si
   le rappel externe passe sous 0,96.
2. Évaluer au niveau du patient (frottis complet), en plus du niveau cellule.
3. Calculer des intervalles de confiance par bootstrap sur les patients.
4. Ajouter un test de capacité maximale (montée progressive jusqu'à violer le SLO) pour
   dimensionner le nombre d'instances.
5. Revoir le quota par clé avec les utilisateurs cibles avant la mise en production.
6. Suivre le rappel en production avec le seuil d'alerte de dérive prévu (0,94, E5).
