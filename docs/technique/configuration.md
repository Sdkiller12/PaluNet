# Configuration

PaluNet a deux sources de configuration :

1. [`config/config.yaml`](../../config/config.yaml) : paramètres du **modèle et du pipeline**
   (hyperparamètres, encodage de classe, version). Lu par l'entraînement **et** par l'API.
2. **Variables d'environnement** : paramètres d'**exploitation** de l'API (clé, quota, chemin du
   modèle…). Lues par [`src/api/settings.py`](../../src/api/settings.py).

## 1. `config/config.yaml`

### Référence des clés

| Clé | Valeur par défaut | Utilisée par | Rôle |
| --- | --- | --- | --- |
| `seed` | `42` | split, train | Graine de reproductibilité (split par patient, sous-échantillonnage, init. Keras) |
| `class_map` | `{0: Uninfected, 1: Parasitized}` | partout | **Encodage figé (B9).** La sigmoïde vaut P(`class_map[1]`) |
| `positive_class` | `Parasitized` | partout | Classe « malade » : rappel prioritaire, `probability_parasitized` |
| `data.raw_dir` | `data/raw/cell_images` | download, prepare | Dossier contenant `Parasitized/` et `Uninfected/` |
| `data.processed_dir` | `data/processed` | prepare, train, export, calibrate, evaluate | Cache `.npz` et manifeste |
| `data.external_dir` | `data/external` | prepare | Jeu externe (A6), même structure que `raw_dir` |
| `data.nih_url` | URL NIH | download | Source officielle du dataset |
| `data.split.{train,val,test}` | `0.70 / 0.15 / 0.15` | prepare | Fractions du split par patient (somme = 1) |
| `preprocessing.image_size` | `128` | prepare, export, API | Côté de l'image carrée en entrée du modèle |
| `preprocessing.min_side_px` | `32` | API (IQA) | En dessous : avertissement « image trop petite » |
| `preprocessing.min_std` | `0.01` | API (IQA) | Écart-type des niveaux de gris en dessous duquel l'image est « quasi uniforme » |
| `augmentation.rotation_deg` | `20` | train | Rotation aléatoire ±20° |
| `augmentation.translation` | `0.10` | train | Translation ±10 % en hauteur et largeur |
| `augmentation.zoom` | `0.15` | train | Zoom ±15 % |
| `augmentation.horizontal_flip` | `true` | train | Retournement horizontal aléatoire |
| `model.version` | `"1.0.0"` | partout | Version semver de l'artefact (B8) |
| `model.backbone` | `mobilenetv2` | (documentaire) | Non lu par le code : MobileNetV2 est codé en dur |
| `model.dense_units` | `128` | model | Neurones de la couche dense de la tête |
| `model.dropout` | `0.3` | model | Taux de dropout de la tête |
| `training.batch_size` | `64` | train | Taille de batch |
| `training.phase1` | `epochs 15, lr 1e-4` | train | Phase 1 : backbone gelé |
| `training.phase2` | `epochs 10, lr 1e-5, unfreeze_last_layers 30` | train | Phase 2 : fine-tuning des 30 dernières couches |
| `training.early_stopping` | `monitor val_loss, patience 4` | train | Arrêt anticipé avec restauration des meilleurs poids |
| `calibration.target_recall` | `0.98` | calibrate | Rappel minimal visé sur la validation (B6) |
| `export.opset` | `13` | export | Opset ONNX |
| `export.max_size_mb` | `20` | export, test modèle | Au-delà : quantification int8 dynamique |
| `acceptance.accuracy` | `0.95` | evaluate | Critère d'acceptation absolu (test) |
| `acceptance.recall` | `0.96` | evaluate | Idem |
| `acceptance.auc` | `0.97` | evaluate | Idem |
| `acceptance.tolerance` | `0.005` | evaluate, test modèle | Baisse tolérée par rapport à la baseline |
| `acceptance.drift_alert_recall` | `0.94` | evaluate `--images-dir` | Seuil d'alerte de dérive (E5) |

### Changer la configuration

- Un changement d'hyperparamètre impose de relancer le pipeline d'entraînement complet.
- Un nouveau modèle doit recevoir une **nouvelle version** (`model.version`), sinon les artefacts
  de la version précédente sont écrasés. Voir [plan-derive.md](../plan-derive.md) pour la
  convention semver.
- **Ne pas inverser `class_map`** sur un modèle existant : l'API refuse de démarrer si la fiche
  modèle ne correspond pas (contrôle B9). Le code reste cohérent si on inverse puis réentraîne,
  car toute l'interprétation passe par `positive_probability()`.

## 2. Module `src/config.py`

| Élément | Description |
| --- | --- |
| `ROOT_DIR` | Racine du projet (parent de `src/`). Base de tous les chemins relatifs |
| `DEFAULT_CONFIG_PATH` | `ROOT_DIR / "config" / "config.yaml"` |
| `load_config(path=None)` | Lit le YAML (chemin explicite, sinon `$CONFIG_PATH`, sinon défaut). Normalise et valide `class_map`. **Résultat mis en cache** (`lru_cache`, 4 entrées) : le dict retourné est partagé, ne pas le modifier |
| `parse_class_map(raw)` | Convertit les clés en `int` et les valeurs en `str` (YAML et JSON peuvent donner des clés `str`) |
| `validate_class_map(class_map, positive_class)` | Lève `ValueError` si les clés ne sont pas exactement `{0, 1}`, si les noms sont dupliqués, ou si `positive_class` est absente |
| `label_of(class_name, class_map)` | Nom de classe → indice. `KeyError` si inconnu |
| `positive_probability(sigmoid_out, class_map, positive_class)` | Sortie sigmoïde → probabilité de la classe positive. Renvoie `sigmoid_out` si la classe positive a l'indice 1, sinon `1 - sigmoid_out`. Fonctionne sur un scalaire ou un tableau numpy. **Seul endroit** où l'orientation de la sortie est interprétée |
| `resolve_path(p)` | Chemin absolu inchangé ; chemin relatif → `ROOT_DIR / p` |

Exemple :

```python
from src.config import load_config, positive_probability
cfg = load_config()
p = positive_probability(0.93, cfg["class_map"], cfg["positive_class"])  # 0.93 = P(Parasitized)
```

## 3. Variables d'environnement de l'API

Lues une seule fois au démarrage par `Settings.from_env()`.

| Variable | Défaut | Type | Rôle |
| --- | --- | --- | --- |
| `CONFIG_PATH` | `config/config.yaml` | chemin | Autre fichier de configuration (lu par `load_config`) |
| `MODEL_PATH` | `models/model_v<model.version>.onnx` | chemin | Artefact ONNX. La fiche `.json` de même nom doit être à côté |
| `MODEL_VERSION` | fiche modèle | str | Surcharge la version affichée |
| `DECISION_THRESHOLD` | fiche modèle | float ]0,1[ | Surcharge le seuil calibré. Vide = ignoré |
| `MOCK_MODEL` | `false` | bool | Prédicteur factice (développement uniquement). Seuil 0,5 non calibré |
| `API_KEY` | vide | liste CSV | Clés acceptées dans `X-API-Key`. **Vide = authentification désactivée** |
| `RATE_LIMIT` | `60/minute` | syntaxe `limits` | Quota par clé (par IP si pas de clé) sur chaque route `/v1/*` |
| `RATE_LIMIT_STORAGE` | `memory://` | URI | Stockage du quota. `redis://…` pour le partager entre instances |
| `MAX_UPLOAD_MB` | `5` | float | Taille maximale d'un fichier (→ 413) |
| `MAX_BATCH_SIZE` | `32` | int | Nombre maximal de fichiers par lot (→ 400) |
| `LOG_LEVEL` | `INFO` | str | Niveau du logger racine |
| `PREDICTION_LOG_FILE` | vide | chemin | Fichier du journal de prédictions (rotation quotidienne UTC) |
| `LOG_RETENTION_DAYS` | `365` | int | Nombre de fichiers journaliers conservés |
| `INTRA_OP_THREADS` | auto | int | Threads ONNX Runtime. Aligner sur les vCPU du conteneur |
| `CORS_ORIGINS` | vide | liste CSV | Origines autorisées. Vide = pas de middleware CORS |
| `SERVE_WEB_APP` | `true` | bool | Monte `web_app/` sur `/` |
| `PORT` | `8000` | int | Utilisé par le `CMD` Docker et le healthcheck uniquement |

Booléens : `1`, `true`, `yes`, `on` (insensible à la casse) valent vrai ; toute autre valeur vaut faux.

### Priorité des sources pour le modèle servi

| Paramètre | 1ʳᵉ source | 2ᵉ source | Absent |
| --- | --- | --- | --- |
| Version | `MODEL_VERSION` | `model_version` de la fiche | `None` |
| Seuil | `DECISION_THRESHOLD` | `decision_threshold` de la fiche | **Erreur** : API non prête (503) |
| `class_map` | — | fiche, comparée à `config.yaml` | **Erreur** si différente |

### Classe `Settings`

`Settings` est une dataclass **gelée** (`frozen=True`). Les tests la construisent directement
(`Settings(model_path="unused.onnx", serve_web_app=False, rate_limit="1000/minute")`) au lieu
de passer par l'environnement. La propriété `max_upload_bytes` convertit `max_upload_mb` en octets.
