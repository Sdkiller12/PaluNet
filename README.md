# PaluNet — Détection automatique du paludisme

Classification binaire de cellules sanguines (**parasitée / saine**) par transfer learning
MobileNetV2, servie par une API FastAPI + ONNX Runtime, avec une interface web de démonstration.

> ⚠️ **Outil d'aide à la décision — ne remplace pas un diagnostic médical certifié.**
> V1 = démonstration uniquement (production gelée jusqu'à validation clinique).

Référence : [cahier des charges V3.1](cdc-optimisé-détection-automatique-du-paludisme.md).
Documentation technique détaillée : [docs/technique/](docs/technique/README.md).

## Installation

Python **3.10 ou 3.11** (TensorFlow 2.15 ne supporte pas les versions plus récentes).

```bash
py -3.11 -m venv .venv                     # Windows (ou python3.11 -m venv .venv)
.venv\Scripts\activate                     # Windows (ou source .venv/bin/activate)
pip install -r requirements-train.txt -r requirements-dev.txt
```

| Fichier | Contenu |
| --- | --- |
| `requirements.txt` | API / inférence uniquement (image Docker, sans TensorFlow) |
| `requirements-train.txt` | + TensorFlow, tf2onnx, scikit-learn (pipeline offline) |
| `requirements-dev.txt` | + pytest, couverture, Locust |

## Pipeline d'entraînement (reproductible)

Toutes les commandes se lancent depuis la racine du projet. Les hyperparamètres, l'encodage de
classe et la version du modèle sont dans [`config/config.yaml`](config/config.yaml).

```bash
# 1. Données (A1) — télécharge le dataset NIH, ou réutilise une archive / un dossier local
python -m src.data.download                                   # depuis le NIH
python -m src.data.download --source "Malaria Cell Images Dataset.zip"

# 2. Split par patient 70/15/15 + cache prétraité (A2, A4, A6)
python -m src.data.prepare

# 3. Entraînement 2 phases (B1–B4)  — smoke test : --limit 2000 --epochs1 1 --epochs2 1
python -m src.models.train

# 4. Export ONNX (+ quantification int8 si > 20 Mo) (B5)
python -m src.models.export

# 5. Calibration du seuil sur la validation : rappel ≥ 98 %, précision maximale (B6)
python -m src.models.calibrate_threshold

# 6. Évaluation sur le test (+ jeu externe) et création de la baseline (B7, E4)
python -m src.models.evaluate --write-baseline
```

Artefacts (hors Git) :

| Fichier | Rôle |
| --- | --- |
| `models/model_v<version>.onnx` | Modèle servi (sorties `probability` et `cam` pour Grad-CAM) |
| `models/model_v<version>.json` | Fiche modèle : version, class_map, **seuil calibré**, détails de calibration, sha256 |
| `metrics_baseline/<version>.json` | Métriques de référence pour la non-régression (versionné dans Git) |
| `logs/training_v<version>_phase{1,2}.csv`, `logs/experiments.csv` | Suivi d'expériences (E8) |

### Points de conception importants

- **Split par patient.** Le dataset NIH contient de nombreuses cellules par patient (~200 patients).
  Un split au niveau image mettrait un même patient dans le train et le test, ce qui gonfle les
  métriques. `src/data/split.py` répartit des patients entiers, en équilibrant les classes.
- **Prétraitement unique** (`src/data/preprocessing.py`, sans TensorFlow) : utilisé pour construire
  le cache d'entraînement **et** par l'API. La mise à l'échelle [-1, 1] propre à MobileNetV2 est une
  couche du modèle, donc incluse dans l'ONNX.
- **Encodage de classe figé** (B9) : `class_map: {0: Uninfected, 1: Parasitized}`, donc la sortie
  sigmoïde vaut P(Parasitized). L'API refuse de démarrer si la fiche modèle ne correspond pas à la config.
- **Seuil jamais à 0.5 par défaut** : sans seuil calibré, l'API reste non prête (`/health` = 503).
- **Calibration et évaluation sur l'artefact ONNX** réellement déployé, pas sur le modèle Keras.
- **Grad-CAM sans TensorFlow à l'inférence** : la tête GAP → Dense → Dense permet un gradient
  analytique. La carte est donc calculée dans le graphe ONNX (voir `build_inference_model`).

## API

```bash
uvicorn src.api.main:app --port 8000                        # modèle réel
MOCK_MODEL=true uvicorn src.api.main:app --port 8000        # sans modèle (développement)
```

- Interface de démonstration : http://localhost:8000/
- Documentation OpenAPI : http://localhost:8000/docs

| Route | Description |
| --- | --- |
| `POST /v1/predict` | `file` (PNG/JPEG ≤ 5 Mo), `include_heatmap` optionnel |
| `POST /v1/predict/batch` | `files` (≤ 32) ; renvoie `results` + `errors` par fichier |
| `GET /health` | Readiness : 200 + `{status, model_version, decision_threshold, class_map}`, 503 si modèle non chargé |
| `GET /health/live` | Liveness |
| `GET /metrics` | Latences p50/p95/p99, taux d'erreur, version, seuil (par instance) |

```bash
curl -H "X-API-Key: $API_KEY" -F file=@cellule.png http://localhost:8000/v1/predict
```

```json
{
  "prediction": "Parasitized",
  "confidence": 0.9421,
  "probability_parasitized": 0.9421,
  "model_version": "1.0.0",
  "processing_time_ms": 21,
  "heatmap_base64": null,
  "quality_warnings": []
}
```

`confidence` = probabilité de la classe **prédite** ; `probability_parasitized` = probabilité
d'infection, quelle que soit la prédiction. Codes d'erreur : 400 (format / image corrompue), 401 (clé
API), 413 (> 5 Mo), 429 (quota), 503 (modèle non chargé), 500 (erreur interne).

### Variables d'environnement

| Variable | Défaut | Rôle |
| --- | --- | --- |
| `MODEL_PATH` | `models/model_v<config>.onnx` | Artefact ONNX (la fiche `.json` doit être à côté) |
| `MODEL_VERSION` | fiche modèle | Surcharge de la version affichée |
| `DECISION_THRESHOLD` | fiche modèle | Surcharge du seuil calibré |
| `API_KEY` | *(vide = auth désactivée)* | Clé(s) acceptée(s), séparées par des virgules. **Obligatoire hors réseau local** |
| `RATE_LIMIT` | `60/minute` | Quota par clé API (par IP si auth désactivée) |
| `RATE_LIMIT_STORAGE` | `memory://` | `redis://…` pour partager le quota entre plusieurs instances |
| `PORT`, `LOG_LEVEL` | `8000`, `INFO` | |
| `PREDICTION_LOG_FILE` | *(stdout seul)* | Journal des prédictions, rotation quotidienne |
| `LOG_RETENTION_DAYS` | `365` | Purge automatique au-delà de 12 mois |
| `MOCK_MODEL` | `false` | Modèle factice (développement front/back uniquement) |
| `INTRA_OP_THREADS` | auto | Threads ONNX Runtime (aligner sur les vCPU du conteneur) |

## Docker

```bash
cp .env.example .env          # puis définir API_KEY
docker compose up --build     # 2 vCPU / 2 Go, healthcheck sur /health
```

Le modèle (`models/model_v*.onnx` et `.json`) doit exister avant le build, ou être monté en volume.

## Tests

```bash
pytest --cov                              # unitaires + intégration (+ non-régression si le modèle existe)
locust -f tests/load/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 2m
```

Le test de charge échoue si le P95 de `/v1/predict` dépasse 500 ms. Le lancer contre le conteneur
limité à 2 vCPU / 2 Go, avec `RATE_LIMIT` relevé ou une clé par utilisateur simulé.

## Structure

```
config/          config.yaml (hyperparamètres, class_map, version)
src/data/        download, split (par patient), preprocessing (partagé), prepare
src/models/      model, train, export, calibrate_threshold, evaluate, inference, explain, metrics
src/api/         main, routes/, service, dependencies (auth, quota), schemas, settings
web_app/         interface HTML/JS
tests/           unit/, integration/, model/ (non-régression), load/ (Locust)
docs/            plan-derive.md (E5), gouvernance-donnees.md (A7, loi 2013-450)
```
