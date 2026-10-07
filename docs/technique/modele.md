# Modèle (`src/models/`)

Ce document couvre le cycle de vie du modèle : architecture, entraînement, export ONNX,
calibration du seuil, évaluation, puis inférence et explicabilité.

| Module | Dépend de TensorFlow | Utilisé par l'API |
| --- | --- | --- |
| `model.py` | oui | non |
| `train.py` | oui | non |
| `export.py` | oui | non |
| `calibrate_threshold.py` | non (ONNX Runtime) | non |
| `evaluate.py` | non (ONNX Runtime) | non |
| `inference.py` | non | **oui** |
| `explain.py` | non | **oui** |
| `metrics.py` | non (numpy pur) | non |
| `artifacts.py` | non | **oui** |

## 1. Artefacts versionnés — `artifacts.py` (B8)

`ModelPaths(version)` centralise tous les chemins d'une version. Le constructeur refuse une
version qui n'est pas au format semver `X.Y.Z` (`ValueError`).

| Propriété | Chemin | Versionné dans Git |
| --- | --- | --- |
| `keras` | `models/keras/model_v<v>.keras` | non |
| `onnx` | `models/model_v<v>.onnx` | non |
| `metadata` | `models/model_v<v>.json` (fiche modèle) | non |
| `training_log` | `logs/training_v<v>.csv` (base des noms `_phase1` / `_phase2`) | non |
| `baseline` | `metrics_baseline/<v>.json` | **oui** |

Fonctions utilitaires :

- `metadata_path_for(onnx_path)` : même nom que l'ONNX, extension `.json`. L'API l'utilise pour
  trouver la fiche d'un `MODEL_PATH` quelconque.
- `read_json(path)`, `write_json(path, data)` (UTF-8, indentation 2, crée le dossier parent).
- `update_metadata(path, **fields)` : fusionne des champs dans une fiche existante (utilisé par
  la calibration).

### Fiche modèle (`model_v<v>.json`)

Créée par `export.py`, complétée par `calibrate_threshold.py` :

```json
{
  "model_version": "1.0.0",
  "created_at": "2026-09-29T18:00:00+00:00",
  "class_map": {"0": "Uninfected", "1": "Parasitized"},
  "positive_class": "Parasitized",
  "sigmoid_output": "P(class_map[1]) = P(Parasitized)",
  "input": {"name": "image", "shape": [null, 128, 128, 3], "dtype": "float32",
            "preprocessing": "RGB, resize bilinéaire, /255 -> [0,1] (src/data/preprocessing.py)"},
  "outputs": ["probability", "cam"],
  "size_mb": 9.1,
  "quantized_int8": false,
  "sha256": "…",
  "decision_threshold": 0.3127,
  "calibration": {
    "dataset": "validation",
    "calibrated_at": "…",
    "rule": "précision maximale sous contrainte rappel >= cible",
    "target_recall": 0.98,
    "val_recall": 0.9803, "val_precision": 0.95, "val_specificity": 0.94, "val_auc": 0.99,
    "val_size": 4100,
    "pr_curve": [{"threshold": 0.1, "recall": 0.99, "precision": 0.9, "specificity": 0.88}, "…"]
  }
}
```

Les valeurs numériques ci-dessus sont des exemples. `decision_threshold` vaut `null` tant que la
calibration n'a pas été faite ; l'API refuse alors de se déclarer prête.

## 2. Architecture — `model.py` (3.2)

```
image (N,128,128,3) float32 [0,1]
  → Rescaling(×2, −1)            "to_mobilenet_range"   → [-1, 1]
  → MobileNetV2 (ImageNet, sans tête)   dernière couche "out_relu" (N,4,4,1280)
  → GlobalAveragePooling2D       "gap"                   (N,1280)
  → Dense(128, ReLU)             "head_dense"
  → Dropout(0.3)                 "head_dropout"
  → Dense(1, sigmoïde)           "probability"           (N,1) = P(Parasitized)
```

| Élément | Rôle |
| --- | --- |
| `build_model(cfg, weights="imagenet")` | Construit le modèle `palunet`. Le backbone est créé avec `input_tensor=x` : ses couches sont « à plat » dans le modèle final, ce qui rend `out_relu` accessible pour Grad-CAM. Backbone entièrement gelé à la sortie |
| `NON_BACKBONE` | Noms des couches qui ne font pas partie du backbone (entrée, rescaling, tête) |
| `backbone_layers(model)` | Couches du modèle hors `NON_BACKBONE` |
| `set_backbone_trainable(model, trainable_last)` | Gèle tout le backbone sauf ses `trainable_last` dernières couches. Les `BatchNormalization` restent **toujours gelées** : avec de petits batchs, les dégeler détruirait les statistiques ImageNet |
| `build_inference_model(model)` | Modèle d'export à deux sorties : `probability` (N,1) et `cam` (N,h,w) |

### Grad-CAM analytique, sans gradient à l'inférence

Grad-CAM pondère chaque canal `k` de la dernière carte de features `A` par
`α_k = moyenne_{i,j} ∂z/∂A[i,j,k]`, où `z` est le logit de la classe 1.

La tête est `z = w₂ · ReLU(W₁ᵀ · GAP(A) + b₁) + b₂`. Le gradient est donc :

```
∂z/∂A[i,j,k] = Σ_u  w₂[u] · 1[h_u > 0] · W₁[k,u] / (H·W)
```

Il ne dépend pas de `(i, j)` : c'est directement `α_k`. `cam_fn` calcule cette expression avec
de simples produits matriciels, puis `cam = Σ_k α_k · A[:, :, :, k]`. Le graphe ONNX produit donc
la carte Grad-CAM **brute** (avant ReLU, orientée vers la classe 1) sans calcul de gradient, et
ONNX Runtime suffit à l'inférence.

La carte finale est `ReLU(cam)` si la classe prédite est 1, `ReLU(−cam)` si c'est la classe 0
(voir `explain.class_activation`). À 128 px d'entrée, la résolution de la carte est 4 × 4,
ensuite agrandie en 128 × 128.

> Si la tête change (couche supplémentaire, autre activation), la formule n'est plus valable et
> `build_inference_model` doit être réécrit.

## 3. Entraînement — `train.py` (B1–B4)

```bash
python -m src.models.train                                    # complet
python -m src.models.train --limit 2000 --epochs1 1 --epochs2 1  # smoke test
python -m src.models.train --deterministic                    # opérations déterministes (plus lent)
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--version` | `model.version` | Version des artefacts produits |
| `--epochs1`, `--epochs2` | 15, 10 | Époques max par phase |
| `--batch-size` | 64 | Taille de batch |
| `--limit` | aucun | Sous-échantillon du train ; la validation est réduite à `max(limit // 4, 64)` |
| `--deterministic` | non | `tf.config.experimental.enable_op_determinism()` |

### Déroulement

1. `keras.utils.set_random_seed(seed)`.
2. Chargement de `train.npz` et `val.npz`. **Le test n'est jamais chargé** (A4).
3. `make_dataset` : `tf.data` avec mélange (train uniquement), batch, conversion `/255`
   (identique à `uint8_to_model_input`), augmentation (train uniquement), `prefetch`.
4. **Phase 1** — extraction de features : backbone gelé, Adam lr 1e-4.
5. **Phase 2** — fine-tuning : dégel des 30 dernières couches du backbone (hors BatchNorm),
   recompilation, Adam lr 1e-5.
6. Chaque phase : `EarlyStopping(val_loss, patience=4, restore_best_weights=True)` et
   `CSVLogger` vers `logs/training_v<v>_phase{1,2}.csv`.
7. Sauvegarde `models/keras/model_v<v>.keras`.
8. Métriques de validation **au seuil 0,5, à titre indicatif** (le seuil réel est calibré plus
   tard) et ajout d'une ligne dans `logs/experiments.csv` (E8).

Augmentation (`augmentation(cfg)`) : `RandomFlip("horizontal")` (si activé), `RandomRotation`
(±20°), `RandomTranslation` (±10 %), `RandomZoom` (±15 %). Remplissage constant (noir), cohérent
avec le fond des images NIH.

Métriques Keras suivies : `accuracy`, `auc`, `recall` et `precision` de la classe d'indice 1.

Colonnes de `logs/experiments.csv` : `timestamp, model_version, train_size, val_size, epochs1,
epochs2, batch_size, lr1, lr2, val_accuracy@0.5, val_recall@0.5, val_auc, duration_s`.

## 4. Export ONNX — `export.py` (B5)

```bash
python -m src.models.export [--version 1.0.0] [--force-quantize]
```

1. Charge le `.keras` et construit le modèle d'inférence (`probability` + `cam`).
2. Conversion `tf2onnx` (opset 13), entrée nommée `image`, forme `(None, 128, 128, 3)`.
3. **Vérification d'équivalence** Keras / ONNX Runtime sur 64 images de validation : écart
   absolu max de probabilité ≤ 1e-4, sinon arrêt.
4. Si l'ONNX dépasse `export.max_size_mb` (20 Mo) ou si `--force-quantize` : la version fp32 est
   renommée `model_v<v>_fp32.onnx`, puis une quantification dynamique int8 (`QUInt8`) produit
   le nouvel `model_v<v>.onnx`. Nouvelle vérification avec une tolérance de 0,05.
5. Écriture de la fiche modèle (`decision_threshold: null`, sha256 de l'artefact final).

Après l'export, il **faut** lancer la calibration : sans elle, l'API reste non prête.

## 5. Moteur d'inférence — `inference.py`

Partagé par l'API, la calibration et l'évaluation. Ainsi, le seuil est calibré et les métriques
sont mesurées sur l'artefact **réellement déployé** (le `.onnx`), pas sur le modèle Keras.

| Élément | Rôle |
| --- | --- |
| `RawOutput(sigmoid, cam)` | `sigmoid` : `(N,)` = P(classe 1). `cam` : `(N,h,w)` brut ou `None` |
| `Predictor` (Protocol) | Interface : attribut `has_cam`, méthode `run(batch, with_cam)` |
| `OnnxPredictor(model_path, intra_op_threads, session)` | Session ONNX Runtime sur `CPUExecutionProvider`. Le paramètre `session` permet d'injecter une fausse session dans les tests |
| `OnnxPredictor._pick(outputs, name, rank)` | Trouve une sortie par nom exact, par préfixe `name:` (noms générés par tf2onnx), ou à défaut par rang (2 pour `probability`, 3 pour `cam`). Sans sortie `probability` : `ValueError` |
| `OnnxPredictor.run(batch, with_cam)` | Ne demande la sortie `cam` que si elle est voulue et disponible (gain de temps) |
| `OnnxPredictor.predict_uint8(x, batch_size=256)` | Inférence par lots sur un cache uint8 (évaluation hors ligne) |
| `MockPredictor` | Prédicteur factice déterministe (voir ci-dessous) |
| `Decision` | `prediction`, `confidence`, `probability_parasitized`, `predicted_index` |
| `decide(sigmoid, threshold, class_map, positive_class)` | Règle de décision |

### Règle de décision

```
p_pos = positive_probability(sigmoid)          # probabilité de Parasitized
si p_pos >= seuil : prediction = Parasitized, confidence = p_pos
sinon            : prediction = Uninfected,  confidence = 1 − p_pos
probability_parasitized = p_pos  (toujours)
```

`confidence` est la probabilité de la classe **prédite**, pas celle d'une classe fixe. Avec un
seuil bas (ex. 0,3), une cellule à `p_pos = 0,35` est déclarée parasitée avec une confiance de
0,35 : c'est voulu, le rappel est prioritaire.

### `MockPredictor`

Heuristique **sans valeur médicale**, pour développer le front et le back contre le contrat figé :

- pixels de cellule : niveau de gris moyen > 0,05 (le fond noir est ignoré) ;
- pixels « violets » : bleu > rouge à l'intérieur de la cellule ;
- `sigmoid = 1 / (1 + exp(−200 · (fraction_violette − 0,02)))` ;
- `cam` : fraction violette moyenne sur une grille 4 × 4, moins 0,01. Suppose une image de côté
  multiple de 4 (128 px).

## 6. Explicabilité — `explain.py` (C9 / D6)

Post-traitement numpy de la sortie `cam`.

| Fonction | Rôle |
| --- | --- |
| `class_activation(cam_raw, predicted_index)` | Garde la partie qui soutient la classe prédite (`cam` si classe 1, `−cam` si classe 0), applique ReLU, normalise par le maximum dans `[0, 1]` |
| `render_heatmap(activation, size, max_alpha=0.6)` | Agrandissement bilinéaire en `size × size`, palette type « jet » (bleu → cyan → jaune → rouge), canal alpha proportionnel à l'activation (max 60 %) |
| `heatmap_base64(cam_raw, predicted_index, size)` | PNG RGBA encodé en base64 : champ `heatmap_base64` de l'API, à superposer à l'image |

## 7. Métriques — `metrics.py`

numpy pur, utilisable partout. Convention : `y_true` vrai pour la classe positive, `scores` =
probabilité de la classe positive, décision positive si `score >= seuil`.

| Fonction | Rôle |
| --- | --- |
| `roc_auc(y_true, scores)` | AUC par la statistique de Mann-Whitney, rangs moyens pour les ex-aequo. `nan` si une seule classe est présente |
| `confusion(y_true, y_pred)` | `{tp, fn, fp, tn}` |
| `classification_metrics(y_true, scores, threshold)` | `n, threshold, accuracy, recall, precision, specificity, f1, auc, confusion_matrix`. Divisions par zéro → 0 |
| `choose_threshold(y_true, scores, target_recall)` | Calibration (B6), voir ci-dessous |
| `check_against_baseline(metrics, baseline, tolerance)` | Liste des régressions sur `accuracy`, `recall`, `auc` (métrique < baseline − tolérance) |
| `check_acceptance(metrics, acceptance)` | Liste des critères absolus non atteints (`accuracy` ≥ 0,95, `recall` ≥ 0,96, `auc` ≥ 0,97) |

### Algorithme de `choose_threshold`

1. Trie les scores par ordre décroissant et cumule TP et FP.
2. Les seuils candidats sont les **valeurs distinctes** des scores (on garde la dernière position
   de chaque valeur, pour traiter correctement les ex-aequo).
3. Pour chaque candidat : rappel = TP / P, précision = TP / (TP + FP).
4. Parmi les candidats où rappel ≥ cible, retient la **précision maximale** ; à précision égale,
   le **seuil le plus élevé**.
5. `ValueError` s'il n'y a aucun positif. La cible est toujours atteignable en théorie (le plus
   petit score donne un rappel de 1).

## 8. Calibration — `calibrate_threshold.py` (B6)

```bash
python -m src.models.calibrate_threshold [--version 1.0.0] [--target-recall 0.98]
```

1. Charge `val.npz` (**jamais le test**) et score avec l'`OnnxPredictor` de l'artefact exporté.
2. `choose_threshold` avec la cible de rappel.
3. Arrondit le seuil **vers le bas** à 4 décimales : un seuil plus bas ne peut que conserver ou
   augmenter le rappel.
4. Écrit `decision_threshold` et un bloc `calibration` (règle, cible, rappel / précision /
   spécificité / AUC de validation, taille, points de la courbe précision-rappel aux seuils
   0,1 … 0,9) dans la fiche modèle.

La variable `DECISION_THRESHOLD` de l'API peut surcharger ce seuil en exploitation.

## 9. Évaluation — `evaluate.py` (B7 / E4 / E5)

```bash
python -m src.models.evaluate                     # test (+ externe si présent)
python -m src.models.evaluate --write-baseline    # fige metrics_baseline/<v>.json
python -m src.models.evaluate --check-baseline    # non-régression
python -m src.models.evaluate --images-dir audit/2026-10   # audit de dérive mensuel
```

| Fonction | Rôle |
| --- | --- |
| `score_arrays(predictor, x_uint8, y, cfg)` | → `(y_pos, p_pos)` orientés vers la classe positive |
| `load_npz(name, cfg)` | `(x, y)` d'un cache, ou `None` s'il n'existe pas |
| `load_folder(folder, cfg)` | Charge et prétraite un dossier `Parasitized/` + `Uninfected/` (images illisibles ignorées) |

Comportement :

- Arrêt si la fiche modèle n'a pas de seuil calibré.
- Évalue `test` (obligatoire) et `external` (si présent) avec le **seuil calibré**, affiche le
  JSON des métriques, avertit pour chaque critère d'acceptation non atteint.
- `--write-baseline` : écrit `metrics_baseline/<v>.json` (version, date, seuil, tolérance,
  métriques test arrondies, taille du test, métriques externes, `acceptance_passed`).
- `--check-baseline` : compare le test à la baseline.
- `--images-dir` : évalue un dossier étiqueté et compare le rappel à `drift_alert_recall`.

| Code de sortie | Signification |
| --- | --- |
| `0` | OK |
| `1` | Régression par rapport à la baseline (`--check-baseline`) |
| `2` | Alerte de dérive : rappel < 0,94 (`--images-dir`) |

Ces codes permettent d'intégrer l'évaluation à un pipeline CI ou à une tâche planifiée.
