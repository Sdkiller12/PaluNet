# Prompt — Vidéo motion design de présentation de PaluNet (5 min)

> À copier tel quel dans l'outil ou l'agent de génération vidéo qui a accès au dossier du projet
> `Palunet/`.

---

## Rôle et objectif

Tu es un motion designer et un réalisateur de vidéos techniques. Tu as accès au dossier complet
du projet **PaluNet**. Produis une **vidéo de motion design de 5 minutes exactement (300 s)**, en
**français**, qui présente le projet en détail : le problème, les données, le modèle, la
calibration, l'explicabilité, l'API, l'interface, la qualité, l'éthique et les limites.

La vidéo doit convaincre deux publics à la fois :
- **un jury technique** (enseignants, ingénieurs ML) qui attend de la rigueur ;
- **des acteurs de santé ivoiriens** (laboratoires, techniciens, PNLP) qui attendent de la clarté.

Ton : sérieux, pédagogique, rythmé, sans sensationnalisme. Aucun chiffre ne doit être inventé.

---

## 1. Sources à lire avant de commencer

Lis ces fichiers en entier. Ils sont la seule source de vérité pour le contenu.

| Fichier | Ce que tu y trouves |
| --- | --- |
| `cdc-optimisé-détection-automatique-du-paludisme.md` | Cahier des charges V3.1 : contexte, objectifs chiffrés, modules A à E, architecture, arbitrages, risques |
| `README.md` | Pipeline en 6 commandes, choix de conception, API, Docker, tests |
| `config/config.yaml` | Hyperparamètres, `class_map`, cibles d'acceptation |
| `models/model_v1.0.0.json` | Fiche modèle : seuil calibré, courbe PR, taille, sha256 |
| `logs/experiments.csv` | Paramètres et résultats réels de l'entraînement |
| `logs/training_v1.0.0_phase1.csv`, `logs/training_v1.0.0_phase2.csv` | Courbes d'apprentissage par epoch (à animer) |
| `data/processed/split_summary.json` | Répartition des images et des patients par jeu |
| `metrics_baseline/1.0.0.json` | Métriques **sur le jeu de test**, si le fichier existe |
| `docs/technique/*.md` | Documentation détaillée : modèle, pipeline, API, interface, déploiement, tests, points d'attention |
| `docs/gouvernance-donnees.md` | Loi ivoirienne n° 2013-450, ARTCI, anonymisation |
| `docs/plan-derive.md` | Audit mensuel, seuils d'alerte, réentraînement |
| `docs/PaluNet-documentation-technique.pdf`, `docs/normes-modele-et-images.pdf` | Compléments à consulter |
| `src/` | Code Python : `data/`, `models/`, `api/` (extraits de code à montrer) |
| `web_app/` | Interface React (pages `Home`, `Login`, `Dashboard`, `AdminDashboard`, `Analyzer`) |
| `Malaria Cell Images Dataset/` ou `data/raw/cell_images/` | Vraies images de cellules (`Parasitized/`, `Uninfected/`) |

---

## 2. Faits vérifiés à utiliser

Ces chiffres viennent des fichiers du projet. Utilise-les tels quels.

**Données (dataset NIH, frottis minces colorés au Giemsa)**
- 27 558 images de cellules uniques, 2 classes équilibrées, issues de 201 patients.
- Split **par patient** 70 / 15 / 15, avec une graine fixe (`seed: 42`) :
  - train : 19 195 images, 141 patients ;
  - validation : 4 173 images, 30 patients ;
  - test : 4 190 images, 30 patients.
- Raison : un même patient ne doit pas apparaître à la fois dans le train et dans le test,
  sinon les métriques sont gonflées.
- Prétraitement : 128 × 128 px, RGB, valeurs ramenées dans [0, 1]. Le même code sert à
  l'entraînement et à l'API (`src/data/preprocessing.py`).
- Contrôle qualité de l'image (IQA) : rejet des images de moins de 32 px ou quasi uniformes.
- Augmentation (train uniquement) : rotation ±20°, translation 10 %, zoom 15 %, flip horizontal.

**Modèle**
- Transfer learning **MobileNetV2** pré-entraîné sur ImageNet.
- Tête : GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.3) → Dense(1, sigmoïde).
- La sortie sigmoïde vaut P(Parasitée). L'encodage est figé : `0 = Uninfected`, `1 = Parasitized`.
- Entraînement en 2 phases : extraction de caractéristiques (backbone gelé), puis fine-tuning
  des dernières couches avec un taux d'apprentissage plus faible, et early stopping sur la
  validation. Prends les valeurs exactes du run dans `logs/experiments.csv` : 15 epochs puis 10,
  lr 1e-4 puis 1e-5, batch 64, durée 1 143 s (environ 19 min).

**Calibration du seuil (point fort à mettre en avant)**
- Au seuil standard de 0,5 (validation) : accuracy 95,35 %, rappel 93,77 %, AUC 0,9876.
- Règle : précision maximale sous la contrainte rappel ≥ 98 %.
- Seuil calibré : **0,156**. Sur la validation : rappel 98,04 %, précision 88,58 %,
  spécificité 87,34 %.
- Message clé : en santé, rater un malade (faux négatif) coûte plus cher qu'une fausse alerte.
  Le seuil n'est donc jamais 0,5 par défaut. Sans seuil calibré, l'API refuse de se dire prête.
- Utilise la courbe PR de `models/model_v1.0.0.json` (seuils 0,1 à 0,9) pour animer le compromis
  rappel / précision.

**Métriques sur le jeu de test**
- Si `metrics_baseline/1.0.0.json` existe, affiche ses valeurs et compare-les aux cibles du CDC :
  accuracy ≥ 95 %, rappel ≥ 96 %, AUC ≥ 0,97.
- Si le fichier n'existe pas, ne mets **aucun** chiffre de test à l'écran. Affiche seulement les
  métriques de validation, avec la mention « mesuré sur le jeu de validation ».

**Export et inférence**
- Export **ONNX** (opset 13) : 10,18 Mo, sous la limite de 20 Mo, donc sans quantification.
- Écart maximal Keras / ONNX : 5,48 × 10⁻⁶. Le modèle servi est fidèle au modèle entraîné.
- ONNX Runtime à l'inférence : pas de TensorFlow, pas de GPU, CPU seulement.
- Grad-CAM calculé directement dans le graphe ONNX (sortie `cam`). Sa résolution native est
  4 × 4, agrandie en 128 × 128 : la carte indique une zone, pas un parasite précis.

**API (FastAPI)**
- `POST /v1/predict` : une image PNG/JPEG de 5 Mo maximum, avec l'option `include_heatmap`.
- `POST /v1/predict/batch` : jusqu'à 32 images, avec les résultats et les erreurs par fichier.
- `GET /health` (readiness, 503 si le modèle n'est pas chargé), `GET /health/live`, `GET /metrics`
  (latences p50 / p95 / p99).
- Réponse : `prediction`, `confidence` (probabilité de la classe prédite),
  `probability_parasitized` (toujours la probabilité d'infection), `model_version`,
  `processing_time_ms`, `heatmap_base64`, `quality_warnings`.
- Sécurité : clé `X-API-Key`, quota de 60 requêtes par minute (`slowapi`).
- Codes d'erreur : 400, 401, 413, 429, 503, 500.
- SLO : latence P95 < 500 ms, vérifiée par un test de charge Locust.

**Qualité et déploiement**
- 63 tests pytest : unitaires (prétraitement, split, métriques, inférence), intégration API et
  non-régression du modèle. Objectif de couverture : ≥ 70 %. S'y ajoute un test de charge Locust.
- CI/CD GitHub Actions : tests à chaque push, build Docker et smoke test sur `main`.
- Docker : 2 vCPU / 2 Go par instance, healthcheck sur `/health`, configuration par variables
  d'environnement.
- Suivi d'expériences en CSV. Versioning sémantique du modèle (`1.0.0`).

**Éthique et conformité (Côte d'Ivoire)**
- Loi n° 2013-450 sur les données personnelles, autorité de contrôle : ARTCI.
- Aucune image conservée : traitement en mémoire uniquement.
- Logs anonymisés, sans nom de fichier, sans IP et sans image. Rétention de 12 mois, puis purge
  automatique.
- Accord du laboratoire et du comité d'éthique obligatoire avant toute image réelle de patient.
- Plan de dérive : audit mensuel sur au moins 200 images, alerte si le rappel passe sous 94 %,
  réentraînement documenté.

**Limites (à dire honnêtement)**
- Outil d'**aide à la décision**. Il ne remplace pas un diagnostic médical certifié. La V1 sert
  uniquement à la démonstration.
- Le jeu externe d'un laboratoire ivoirien (≥ 200 images) n'est pas encore fourni : la
  performance hors du dataset NIH reste inconnue.
- Cellules uniques en frottis mince. Les frottis complets, la goutte épaisse, la détection de
  l'espèce et le comptage de la parasitémie sont hors du périmètre de la V1.

---

## 3. Direction artistique

- **Format** : 1920 × 1080, 30 fps, 16:9, durée de 300 s exactement. Exports : MP4 H.264 et
  sous-titres `.srt`.
- **Palette** (reprise de l'interface `web_app/`) :
  - fond sombre `#0F172A` (bleu nuit) ;
  - fond clair `#F8FAFC` et surfaces `#E2E8F0` ;
  - accent principal `#0EA5E9` (bleu ciel), pour les données, le code et les flux ;
  - classe « Parasitée » `#EF4444` (rouge), classe « Saine » en vert (par exemple `#22C55E`) ;
  - surlignage `#FEF08A`.
- **Typographie** : Inter (titres en semi-bold ou bold, corps en regular). Code en JetBrains Mono
  ou Fira Code.
- **Motif récurrent** : une cellule (globule rouge stylisé, disque biconcave) qui sert de fil
  conducteur. Elle se divise, devient un pixel, un tenseur, un nœud du réseau, puis un point
  de la courbe.
- **Style d'animation** : flat design et isométrie légère, transitions par morphing de formes,
  easing doux (`easeInOutCubic`), apparitions décalées (stagger de 80 à 120 ms), compteurs
  numériques animés pour les chiffres.
- **Images réelles** : utilise de vraies images de cellules du dataset pour les scènes sur les
  données et le Grad-CAM. Ne montre **jamais** leurs noms de fichiers à l'écran : les noms NIH
  (`C100P61ThinF_…`) contiennent un identifiant patient.
- **Voix off** : française, posée, environ 140 mots par minute. Un texte est proposé par scène
  ci-dessous ; tu peux l'ajuster de ±10 % pour tenir le timing.
- **Musique** : électronique ambient ou corporate tech, discrète (−18 dB sous la voix), montée
  légère sur le hook et l'outro. Sound design subtil : clics, whoosh, bips de validation.
- **Sous-titres** : incrustés en bas, 2 lignes maximum, et fournis aussi en `.srt`.
- **Accessibilité** : contraste ≥ 4,5:1 pour le texte. Le rouge et le vert sont toujours
  accompagnés d'une icône ou d'un libellé (daltonisme).

---

## 4. Storyboard détaillé (14 scènes, 300 s)

### Scène 1 — Hook · 0:00 → 0:15 (15 s)
- **Visuel** : fond noir. Une goutte de sang tombe et s'étale en frottis. Zoom microscope
  jusqu'à une cellule, où apparaît un parasite. Titre « PaluNet » en surimpression, avec le
  sous-titre « Détection automatique du paludisme par deep learning ».
- **Voix off** : « Une goutte de sang, un microscope, et des centaines de cellules à examiner,
  une par une. C'est encore ainsi qu'on diagnostique le paludisme. Et si l'intelligence
  artificielle pouvait aider ? »

### Scène 2 — Contexte en Côte d'Ivoire · 0:15 → 0:40 (25 s)
- **Visuel** : carte de la Côte d'Ivoire qui se dessine au trait, en bleu ciel. Pictogrammes :
  microscopiste, horloge, fatigue, file d'attente de lames. Trois mots-clés animés : « Lent »,
  « Dépendant de l'expert », « Sujet à la fatigue ».
- **Voix off** : « En Côte d'Ivoire, le paludisme reste endémique. L'examen microscopique du
  frottis sanguin est la référence : il est fiable, mais lent. Il dépend de l'expertise de
  l'observateur et de sa fatigue. Le nombre d'analyses par jour est limité, alors que le besoin
  est immense. »

### Scène 3 — La solution PaluNet · 0:40 → 1:00 (20 s)
- **Visuel** : schéma d'architecture animé, inspiré du diagramme mermaid de la section 3.1 du
  CDC. À gauche, le bloc « Pipeline hors-ligne » (données → entraînement → calibration →
  export). À droite, le bloc « En ligne » (interface → API FastAPI → ONNX Runtime). Les flux
  sont des particules bleues.
- **Voix off** : « PaluNet classe automatiquement une cellule sanguine comme parasitée ou saine.
  Ce n'est pas qu'un modèle : c'est un système complet et déployable. Il comprend un pipeline de
  données reproductible, un réseau de neurones, une API d'inférence et une interface web. »

### Scène 4 — Les données · 1:00 → 1:30 (30 s)
- **Visuel** : une mosaïque de vraies cellules s'assemble, rouges à gauche (parasitées), vertes
  à droite (saines). Un compteur monte jusqu'à 27 558. Les cellules se regroupent ensuite en
  201 « dossiers patients » colorés, répartis en trois colonnes : Train 70 % (141 patients),
  Validation 15 % (30), Test 15 % (30). Encart « Pourquoi par patient ? » : une animation montre
  qu'un même patient dans le train et dans le test gonfle le score.
- **Voix off** : « Le modèle apprend sur le dataset public du NIH : 27 558 images de cellules,
  issues de 201 patients, équilibrées entre cellules parasitées et cellules saines. Point
  crucial : la séparation se fait par patient, pas par image. Les cellules d'un même patient
  restent ensemble. Ainsi, le test mesure une vraie capacité à généraliser, et non une mémoire
  déguisée. »

### Scène 5 — Prétraitement et augmentation · 1:30 → 1:50 (20 s)
- **Visuel** : une cellule passe dans une « machine ». Elle est redimensionnée en 128 × 128
  (une grille de pixels apparaît), ses valeurs sont normalisées entre 0 et 1, puis un contrôle
  qualité affiche un tampon « OK ». Ensuite, 4 copies tournent, glissent, zooment et se
  retournent (augmentation).
- **Voix off** : « Chaque image est redimensionnée en 128 par 128 pixels et normalisée, puis
  passe un contrôle qualité. Pendant l'entraînement, on crée des variantes : rotations,
  décalages, zooms, retournements. Le modèle devient plus robuste. Et ce prétraitement est le
  même code à l'entraînement et en production. »

### Scène 6 — Le modèle MobileNetV2 · 1:50 → 2:25 (35 s)
- **Visuel** : réseau MobileNetV2 en blocs isométriques empilés. Étiquette « Pré-entraîné sur
  ImageNet », avec des icônes d'objets du quotidien qui s'effacent. Une tête de classification
  se construit bloc par bloc : GAP → Dense 128 → Dropout 0,3 → Sigmoïde. **Phase 1** : le
  backbone est grisé avec un cadenas, seule la tête s'allume. **Phase 2** : les dernières couches
  se déverrouillent. En bas, les courbes réelles de `logs/training_v1.0.0_phase*.csv` (loss et
  AUC) se tracent en direct.
- **Voix off** : « Plutôt que de partir de zéro, PaluNet utilise le transfer learning avec
  MobileNetV2. Ce réseau léger a déjà appris à voir sur des millions d'images. On lui ajoute une
  tête de classification. Première phase : le réseau est gelé, seule la tête apprend. Deuxième
  phase : on libère les dernières couches pour les affiner sur nos cellules, avec un taux
  d'apprentissage plus faible. L'early stopping arrête l'entraînement avant le surapprentissage. »

### Scène 7 — La calibration du seuil · 2:25 → 2:55 (30 s)
- **Visuel** : un curseur de seuil sur une jauge de 0 à 1, positionné d'abord sur 0,5. À côté,
  deux grands compteurs : Rappel 93,77 % et Précision. Une silhouette de patient rouge passe à
  travers le filtre (faux négatif) et un panneau d'alerte s'affiche. Le curseur glisse ensuite
  jusqu'à **0,156** : le rappel monte à 98,04 % et la précision descend à 88,58 %. La courbe PR
  de la fiche modèle s'anime avec un point qui glisse. Message final à l'écran : « Mieux vaut une
  fausse alerte qu'un malade oublié. »
- **Voix off** : « En médecine, toutes les erreurs ne se valent pas. Rater une cellule infectée
  est bien plus grave qu'une fausse alerte. Au seuil classique de 0,5, le modèle détecte
  93,8 % des cellules parasitées. Nous calibrons donc le seuil sur le jeu de validation, pour
  garantir au moins 98 % de rappel. Le seuil retenu, 0,156, détecte 98 % des cellules
  infectées. Sans seuil calibré, l'API refuse de démarrer. »

### Scène 8 — L'explicabilité Grad-CAM · 2:55 → 3:15 (20 s)
- **Visuel** : une vraie cellule parasitée en grand. Une carte de chaleur Grad-CAM (dégradé du
  bleu au rouge) se superpose progressivement sur la zone du parasite. Une loupe et la mention
  « Où le modèle a regardé ».
- **Voix off** : « Un médecin ne fait pas confiance à une boîte noire. Avec Grad-CAM, PaluNet
  montre la zone de la cellule qui a motivé sa décision. Cette carte est calculée directement
  dans le modèle exporté, sans TensorFlow en production. Elle indique une région : c'est un
  repère visuel, pas une localisation précise du parasite. »

### Scène 9 — Export ONNX et API · 3:15 → 3:40 (25 s)
- **Visuel** : le modèle Keras se compresse en un fichier « model_v1.0.0.onnx — 10,2 Mo ». Une
  jauge le montre sous la limite de 20 Mo, et un badge affiche « Écart Keras/ONNX : 0,0000055 ».
  Une requête `curl` se tape dans un terminal stylisé, puis la réponse JSON apparaît champ par
  champ (`prediction`, `confidence`, `probability_parasitized`, `processing_time_ms`). En
  arrière-plan, un chronomètre affiche « P95 < 500 ms ». Des icônes clé et bouclier illustrent
  la clé API et le quota de 60 requêtes par minute.
- **Voix off** : « Le modèle est exporté au format ONNX : 10 mégaoctets, et il tourne sur un
  simple processeur, sans carte graphique. Il est servi par une API FastAPI : on envoie une
  image, on reçoit la classe prédite, la probabilité d'infection et, en option, la carte de
  chaleur, en moins d'une demi-seconde. L'accès est protégé par une clé API et un quota de
  requêtes. »

### Scène 10 — L'interface web · 3:40 → 4:00 (20 s)
- **Visuel** : capture ou recréation fidèle de l'interface `web_app/` (pages Home, Analyzer,
  Dashboard), dans un mockup d'écran d'ordinateur. Un curseur glisse une image dans la zone de
  dépôt et clique sur « Analyser ». Le résultat s'affiche : « Parasitée », une barre de
  confiance et la superposition Grad-CAM. Le bandeau d'avertissement médical reste visible en
  permanence.
- **Voix off** : « Côté utilisateur, tout passe par une interface web simple. Le technicien
  dépose une image et lance l'analyse. Il lit le résultat, le score de confiance et la carte de
  chaleur. Le statut d'outil d'aide à la décision est rappelé sur chaque page. »

### Scène 11 — Qualité, tests et déploiement · 4:00 → 4:20 (20 s)
- **Visuel** : un tableau de bord de coches vertes qui s'allument en cascade : 63 tests pytest,
  couverture ≥ 70 %, test de charge Locust, non-régression du modèle. Ensuite, un pipeline
  GitHub Actions (push → tests → build Docker → smoke test) et un conteneur Docker labellisé
  « 2 vCPU / 2 Go » avec un cœur qui bat (healthcheck `/health`).
- **Voix off** : « La fiabilité se prouve. PaluNet compte 63 tests automatisés, un test de
  charge et des tests de non-régression du modèle. À chaque modification, l'intégration continue
  relance les tests et construit l'image Docker. Le déploiement est reproductible, avec des
  ressources maîtrisées. »

### Scène 12 — Éthique et conformité · 4:20 → 4:35 (15 s)
- **Visuel** : un cadenas se referme sur une image qui se dissout (aucune image conservée). Un
  journal de logs dont les champs sensibles sont barrés. Badge « Loi n° 2013-450 · ARTCI ».
  Calendrier « Audit mensuel » avec une alerte « Rappel < 94 % ».
- **Voix off** : « Les données de santé sont sensibles. PaluNet ne conserve aucune image, et
  ses journaux sont anonymisés, conformément à la loi ivoirienne sur les données personnelles.
  Un audit mensuel surveille la dérive du modèle. »

### Scène 13 — Limites et feuille de route · 4:35 → 4:50 (15 s)
- **Visuel** : une route avec des jalons. « V1 : démonstration », cochée. Jalons suivants :
  « Jeu externe ivoirien ≥ 200 images », « Validation clinique », « Goutte épaisse et frottis
  complets », « Espèces et parasitémie ».
- **Voix off** : « Cette première version est une démonstration. Prochaines étapes : valider le
  modèle sur des images de laboratoires ivoiriens, puis s'étendre aux frottis complets, à la
  goutte épaisse et à l'identification des espèces. »

### Scène 14 — Outro · 4:50 → 5:00 (10 s)
- **Visuel** : la cellule-fil conducteur revient et se transforme en logo « PaluNet ». Sous le
  logo : « Aide à la décision — ne remplace pas un diagnostic médical certifié ». Fondu au noir.
- **Voix off** : « PaluNet : l'intelligence artificielle au service des laboratoires, pour un
  diagnostic plus rapide, sans remplacer l'œil de l'expert. »

---

## 5. Contraintes impératives

1. Durée totale de 300 s, avec une tolérance de ±2 s. Respecte le timing de chaque scène.
2. Aucun chiffre absent de la section 2 ou des fichiers sources. En cas de doute, ne l'affiche
   pas.
3. L'avertissement « Outil d'aide à la décision — ne remplace pas un diagnostic médical
   certifié » apparaît au moins dans les scènes 3, 10 et 14.
4. Aucun nom de fichier d'image NIH, aucun identifiant patient et aucune clé API réelle à
   l'écran.
5. Les métriques de validation sont toujours étiquetées « validation ». Les métriques de test
   ne sont affichées que si `metrics_baseline/1.0.0.json` existe.
6. Pas de logos d'institutions (NIH, ARTCI, ministère…) sans autorisation : utilise du texte ou
   des icônes génériques.
7. Les extraits de code montrés viennent du vrai code de `src/` (par exemple
   `group_stratified_split` dans `src/data/split.py`, ou `build_model` dans
   `src/models/model.py`), simplifiés si nécessaire, sans en inventer.

---

## 6. Mise en œuvre recommandée

Si tu génères la vidéo par le code, utilise **Remotion** (React + TypeScript), cohérent avec le
frontend React du projet :
- une composition `PaluNetVideo` (1920 × 1080, 30 fps, 9 000 frames) ;
- un composant par scène, enchaîné avec `<Series>`, et des données lues depuis les JSON et CSV
  du projet (courbe PR, courbes d'apprentissage, split) plutôt que codées en dur ;
- une voix off générée par TTS en français (voix masculine ou féminine posée), synchronisée par
  scène ;
- rendu final : `npx remotion render PaluNetVideo out/palunet.mp4`.

Avec un outil no-code (After Effects, Jitter, Canva, HeyGen…), suis le même storyboard, scène
par scène.

---

## 7. Livrables attendus

1. `palunet.mp4` : 1920 × 1080, 30 fps, 5:00.
2. `palunet.srt` : sous-titres français synchronisés.
3. `voix-off.md` : texte final de la voix off, minuté par scène.
4. Le projet source (Remotion ou équivalent), avec les assets utilisés.
5. Une checklist de vérification : durée, chiffres contrôlés contre les sources, avertissement
   présent, aucun identifiant patient à l'écran.
