# Artefacts modèle (hors Git)

- `model_v<version>.onnx` : modèle servi par l'API
- `model_v<version>.json` : fiche modèle (class_map, seuil calibré, sha256, calibration)
- `keras/model_v<version>.keras` : modèle d'entraînement (non déployé)

Générés par le pipeline décrit dans le README racine. À stocker dans un espace d'artefacts
(stockage objet, registre de modèles), jamais dans Git.
