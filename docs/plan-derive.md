# Plan de suivi de la dérive du modèle (E5)

## Audit mensuel

| Étape | Responsable | Détail |
| --- | --- | --- |
| 1. Échantillon | Data Engineer | ≥ 200 images étiquetées : moitié interne (NIH, hors train), moitié laboratoire partenaire (nouvelles acquisitions), rangées dans `audit/AAAA-MM/Parasitized` et `audit/AAAA-MM/Uninfected` |
| 2. Évaluation | ML Engineer | `python -m src.models.evaluate --images-dir audit/AAAA-MM` |
| 3. Décision | PO + ML Engineer | Voir seuils ci-dessous |
| 4. Traçabilité | ML Engineer | Résultat JSON archivé dans `metrics_baseline/audits/AAAA-MM.json` |

## Seuils d'alerte

| Rappel mesuré | Action |
| --- | --- |
| ≥ 96 % | RAS |
| 94 % – 96 % | Surveillance renforcée : audit à 15 jours, analyse des faux négatifs |
| **< 94 %** | **Alerte** (la commande sort en code 2) : déclenchement du réentraînement |

Suivre aussi la précision et la spécificité : une chute brutale signale souvent un
changement d'acquisition (microscope, coloration, caméra) plutôt qu'une dérive des parasites.

## Processus de réentraînement

1. Ajouter les nouvelles images étiquetées (avec accord éthique, voir `gouvernance-donnees.md`).
2. Incrémenter `model.version` dans `config/config.yaml` (semver : MINOR pour un réentraînement, MAJOR si le contrat ou l'encodage change).
3. Relancer le pipeline complet (README, section Entraînement) : prepare → train → export → calibrate → evaluate `--write-baseline`.
4. Comparer au modèle en service sur le même jeu de test + jeu externe. Déployer seulement si les critères d'acceptation sont atteints.
5. Déployer via `MODEL_PATH` et conserver l'ancienne version pour un retour arrière immédiat.
