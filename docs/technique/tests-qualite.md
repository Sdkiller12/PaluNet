# Tests et qualité (`tests/`)

```bash
pytest --cov                     # unitaires + intégration (+ non-régression si le modèle existe)
pytest tests/unit -q             # unitaires seuls
pytest -k batch                  # filtrer par nom
```

Configuration dans [`pyproject.toml`](../../pyproject.toml) : `testpaths = ["tests"]`,
`pythonpath = ["."]` (imports `src.*` sans installation), `-q`, avertissements de dépréciation masqués.

État au 5 octobre 2026 (Python 3.11.9, Windows) : **67 tests passés, 4 ignorés** (non-régression
du modèle, faute de modèle entraîné), **couverture 94 %** (seuil exigé : 70 %).

## 1. Organisation

| Dossier | Type | Dépendances | Exécution |
| --- | --- | --- | --- |
| `tests/unit/` | Unitaires | numpy, PIL | Toujours, en CI |
| `tests/integration/` | API de bout en bout via `TestClient`, avec `MockPredictor` | FastAPI | Toujours, en CI |
| `tests/model/` | Non-régression du vrai modèle ONNX | modèle + baseline + `test.npz` | Ignorés automatiquement si absents |
| `tests/load/` | Charge (Locust) | API démarrée | Manuel |

Aucun test ne requiert TensorFlow : la CI installe seulement `requirements-dev.txt`.

## 2. Fixtures — `tests/conftest.py`

| Élément | Rôle |
| --- | --- |
| `make_cell(infected, size=140, seed=0)` | Cellule **synthétique** : disque rose sur fond noir, bruit gaussien, et 3 taches violettes si infectée. Le `MockPredictor` la classe correctement |
| `to_bytes(img, fmt="PNG")` | Image PIL → octets |
| `cfg` | `load_config()` |
| `infected_png`, `healthy_png` | Octets PNG des deux cellules types |
| `mock_service(cfg, threshold=0.5)` | `ModelService` avec `MockPredictor`, version `1.0.0` |
| `make_client(cfg, **overrides)` | `TestClient` sur `create_app(Settings(...), service=mock_service)`, sans interface web. Les `overrides` modifient `Settings` (clés, quota, taille max…) |
| `client` | Client standard avec un quota de 1000/minute |

## 3. Couverture fonctionnelle

### Unitaires

| Fichier | Ce qui est vérifié |
| --- | --- |
| `test_preprocessing.py` | Forme, type et plage de `preprocess` ; **égalité exacte** entre le cache uint8 et le prétraitement de l'API ; valeurs de référence figées (détecte tout changement involontaire) ; PNG et JPEG acceptés ; autres formats, données corrompues et PNG tronqué refusés ; composition RGBA sur fond noir ; niveaux de gris → RGB ; IQA (petite image, image uniforme) ; le format réel prime sur l'extension |
| `test_split.py` | Extraction de l'identifiant patient ; **aucun patient dans deux splits** ; proportions et équilibre des classes ; reproductibilité selon la graine ; fractions dont la somme ≠ 1 refusées ; aller-retour du manifeste |
| `test_metrics.py` | AUC parfaite / aléatoire / inversée ; AUC égale à la définition par paires ; AUC `nan` sur une seule classe ; matrice de confusion ; `choose_threshold` atteint la cible de rappel avec la meilleure précision ; contrôles baseline et acceptation |
| `test_inference.py` | `class_map` figée ; validation de `class_map` ; `positive_probability` avec encodage normal et inversé ; `decide` (positif, négatif, confiance de la classe prédite, seuil calibré et non 0,5) ; `OnnxPredictor` avec fausse session (avec / sans `cam`, sortie `probability` obligatoire) ; `MockPredictor` sépare les cellules synthétiques ; signe de la carte selon la classe ; rendu PNG de la heatmap |

### Intégration (`tests/integration/test_api.py`)

| Thème | Tests |
| --- | --- |
| Santé | Contrat `/health` ; 503 et `/v1` en 503 si le modèle manque, liveness toujours 200 |
| Prédiction | Contrat de réponse ; `confidence` relative à la classe prédite ; heatmap par formulaire et par paramètre de requête ; JPEG accepté ; avertissements qualité |
| Erreurs | Fichiers invalides (400) ; format non supporté (400) ; trop gros (413) ; fichier manquant (422) |
| Sécurité | Clé API exigée si configurée, plusieurs clés, `/health` public ; quota **par clé** (la 4ᵉ requête à 3/minute → 429, une autre clé passe) |
| Lot | Échec partiel (résultats + erreurs) ; plus de 32 fichiers → 400 |
| Observabilité | Journal anonymisé (ni nom de fichier ni IP) ; `/metrics` (total, taux d'erreur, p95, version) |
| Chargement | Lecture de la fiche et surcharges env ; refus si `class_map` inversée, seuil absent ou hors ]0,1[ ; fiche manquante ; mode mock ; lecture des variables d'environnement |

### Non-régression du modèle (`tests/model/test_model_regression.py`, E1/E4)

Exécutés seulement si `models/model_v<v>.onnx`, sa fiche, `metrics_baseline/<v>.json` et
`data/processed/test.npz` existent :

- pas de régression d'`accuracy`, `recall` ou `auc` au-delà de la tolérance (0,005) ;
- **pas d'inversion silencieuse de classe** : score moyen > 0,8 sur les parasitées, < 0,2 sur les saines ;
- taille de l'ONNX < 20 Mo ;
- `class_map` de la fiche identique à la config, seuil dans ]0, 1[.

## 4. Couverture de code (E2)

- Mesurée sur `src/`, seuil minimal **70 %** (`fail_under`), lignes manquantes affichées.
- Exclus : les scripts offline dépendant de TensorFlow ou de données réelles (`download`,
  `prepare`, `train`, `model`, `export`, `evaluate`, `calibrate_threshold`). Ils sont validés par
  l'exécution du pipeline et par les tests de non-régression du modèle.

## 5. Test de charge (`tests/load/locustfile.py`, E3)

```bash
locust -f tests/load/locustfile.py --host http://localhost:8000 \
       --headless -u 20 -r 5 -t 2m --csv reports/load
```

- Utilisateur simulé `PredictUser` : attente 0,5 à 1,5 s, 10 appels `/v1/predict` pour 1 appel `/health`.
- Images : jusqu'à 200 PNG de `LOAD_IMAGES_DIR` (défaut `data/raw/cell_images`), sinon 20 images aléatoires.
- `LOAD_API_KEY=k1,k2,…` : une clé par utilisateur, distribuées en rotation.
- À la fin, `enforce_slo` impose **P95 ≤ 500 ms et zéro échec** sur `/v1/predict`, sinon code
  de sortie 1.

Conditions de mesure : conteneur limité à 2 vCPU / 2 Go (`docker compose`), et `RATE_LIMIT`
relevé ou une clé par utilisateur. Sinon, les 429 font échouer le test : 20 utilisateurs
dépassent vite 60 requêtes par minute sur une seule clé.

## 6. Écrire un nouveau test

- Test d'API : utiliser `client` ou `make_client(cfg, <surcharges>)` ; ne jamais dépendre d'un
  vrai modèle.
- Image de test : `to_bytes(make_cell(True))`.
- Faux modèle ONNX : `OnnxPredictor(session=...)` avec un objet qui expose `get_inputs`,
  `get_outputs` et `run` (voir `test_inference.py`).
- Toute modification de `preprocessing.py` doit garder `test_preprocessing_reference_values_are_stable`
  vert, ou s'accompagner d'un réentraînement et d'une nouvelle version de modèle.
