# Points d'attention, limites et pistes d'amélioration

Constats issus de l'analyse du code au 5 octobre 2026 (commit `3c981d0`). Ils ne remettent pas
en cause le fonctionnement de la V1 de démonstration, mais doivent être traités ou acceptés
avant un usage réel.

Gravité : **Haute** = à corriger avant production ; **Moyenne** = risque d'exploitation ;
**Basse** = confort ou maintenabilité.

## 1. Robustesse et sécurité de l'API

| # | Gravité | Constat | Effet | Piste |
| --- | --- | --- | --- | --- |
| 1 | Moyenne | `lifespan` n'intercepte que `ModelLoadError`. Une erreur d'ONNX Runtime (fichier corrompu) ou le `ValueError` d'`OnnxPredictor` (sortie `probability` absente) n'est pas convertie | Le processus s'arrête au démarrage au lieu de rester vivant avec `/health` = 503 | Dans `from_settings`, encapsuler la création d'`OnnxPredictor` et relever `ModelLoadError` |
| 2 | Moyenne | Le `sha256` de la fiche modèle n'est pas vérifié au chargement | Un artefact modifié ou incomplet peut être servi sans alerte | Comparer le hash au démarrage (coût : une lecture du fichier) |
| 3 | Moyenne | Quand l'authentification est désactivée, la clé de quota est quand même `key:<X-API-Key>` si l'en-tête est présent | Un client contourne le quota par IP en envoyant un en-tête différent à chaque requête | Utiliser la clé API seulement si l'authentification est active (`settings.api_keys` non vide) |
| 4 | Basse | Un lot de 32 images compte pour une seule requête dans le quota | Débit effectif jusqu'à 32 fois le quota annoncé | Quota distinct pour `/batch` ou coût proportionnel au nombre d'images |
| 5 | Basse | `/metrics` est public et expose version et seuil | Information utile à un attaquant, sans données patient | Restreindre au réseau interne ou derrière la clé API |
| 6 | Basse | La clé API de l'interface web est stockée en clair dans `localStorage` | Lisible par tout script de la page ou par un autre utilisateur du poste | Acceptable en démonstration ; prévoir une authentification par session ensuite |

## 2. Observabilité

| # | Gravité | Constat | Piste |
| --- | --- | --- | --- |
| 7 | Basse | `error_rate` compte tous les codes ≥ 400 (401, 413, 429 inclus) | Exposer séparément les 4xx et les 5xx pour l'alerting |
| 8 | Basse | Métriques en mémoire, par processus, remises à zéro au redémarrage, format JSON propriétaire | Export Prometheus si une supervision centralisée est mise en place |
| 9 | Basse | Percentiles par la méthode du rang le plus proche, sur les 1 000 dernières requêtes | Suffisant pour une instance ; documenter la fenêtre dans les tableaux de bord |

## 3. Modèle et données

| # | Gravité | Constat | Effet | Piste |
| --- | --- | --- | --- | --- |
| 10 | Haute | Le jeu externe (A6, ≥ 200 images de laboratoire) n'est pas encore fourni | Performance hors NIH inconnue ; c'est le principal risque clinique | Bloquant pour la recette (voir [gouvernance-donnees.md](../gouvernance-donnees.md)) |
| 11 | Moyenne | Redimensionnement en 128 × 128 **sans conserver le ratio** et sans normalisation de couleur | Sensible aux changements de microscope, de grossissement ou de coloration | Surveiller la dérive (E5) ; envisager une normalisation de coloration si les audits le montrent |
| 12 | Moyenne | La formule Grad-CAM de `build_inference_model` suppose exactement la tête GAP → Dense ReLU → Dense | Toute modification de la tête rend la carte fausse, sans erreur | Ajouter un test comparant la carte analytique à un Grad-CAM par `GradientTape` sur le modèle Keras |
| 13 | Basse | La carte Grad-CAM a une résolution native de 4 × 4 (agrandie en 128 × 128) | Localisation grossière : indique une zone, pas un parasite précis | À mentionner aux utilisateurs |
| 14 | Basse | `train.py` charge tout le cache en mémoire (environ 1,4 Go pour le dataset complet) | Échec possible sur une petite machine | `--limit` pour les essais ; `tf.data` depuis le disque si nécessaire |
| 15 | Basse | En cas de quantification, `model_v<v>_fp32.onnx` reste dans `models/` et est copié dans l'image Docker | Image plus lourde | Ajouter `models/*_fp32.onnx` au `.dockerignore` |

## 4. Configuration et maintenabilité

| # | Gravité | Constat | Piste |
| --- | --- | --- | --- |
| 16 | Basse | `load_config()` est mis en cache sur l'argument `path` : `CONFIG_PATH` n'est lu qu'au premier appel | Documenté ici ; appeler `load_config.cache_clear()` dans les tests qui changent `CONFIG_PATH` |
| 17 | Basse | `model.backbone` dans `config.yaml` n'est lu par aucun code (MobileNetV2 est codé en dur) | Le supprimer, ou l'utiliser pour choisir le backbone |
| 18 | Basse | `docker-compose.yml` fixe `MODEL_PATH` par défaut à `model_v1.0.0.onnx` | À mettre à jour à chaque version, ou dériver de la config |
| 19 | Basse | `MockPredictor` suppose un côté d'image multiple de 4 pour la carte `cam` | Sans effet avec 128 px ; à adapter si `image_size` change |
| 20 | Basse | La récupération de l'artefact modèle en CI/CD n'est pas automatisée | Brancher un stockage d'artefacts ou un registre de modèles dans le job `docker` |
| 21 | Basse | Les tests de non-régression du modèle sont toujours ignorés en CI (pas de modèle) | Les exécuter dans un job dédié qui télécharge l'artefact et le `test.npz` |

## 5. Points forts à préserver

- Split **par patient**, testé, qui évite les métriques gonflées.
- Prétraitement **unique** partagé entre entraînement et API, avec tests de valeurs de référence.
- Encodage de classe figé et vérifié au démarrage (B9) ; interprétation centralisée dans
  `positive_probability`.
- Seuil **calibré sur la validation**, sur l'artefact ONNX réellement servi, jamais 0,5 par défaut.
- API **non prête** plutôt que fausse : absence de modèle, de fiche, de seuil ou incohérence → 503.
- Journalisation anonymisée testée (pas de nom de fichier, pas d'IP).
- Grad-CAM calculé dans le graphe ONNX : pas de TensorFlow en production.
