# Gouvernance des données et conformité

Cadre : **Loi ivoirienne n° 2013-450** relative à la protection des données à caractère personnel — autorité de contrôle : **ARTCI**.

## Avant toute image réelle de patient (A7) — bloquant

- [ ] Accord écrit du laboratoire partenaire (référence : ______, date : ______)
- [ ] Avis favorable du comité d'éthique (référence : ______, date : ______)
- [ ] Formalités ARTCI vérifiées par une relecture juridique
- [ ] Procédure d'anonymisation à la source validée (voir ci-dessous)

## Anonymisation à la source

- Les noms de fichiers ne doivent contenir **aucun** identifiant patient. Renommer en `lab<ID-lot>_<n>.png` avant transfert.
  *Attention : les noms NIH (`C100P61ThinF_...`) contiennent un identifiant patient. C'est
  pourquoi l'API ne journalise jamais le nom des fichiers.*
- Supprimer les métadonnées EXIF (réenregistrement PNG sans métadonnées).
- La table de correspondance lot ↔ patient reste au laboratoire, jamais dans le projet.

## Jeu externe (A6)

- ≥ 200 images étiquetées, dans `data/external/Parasitized` et `data/external/Uninfected` (hors Git).
- Étiquetage par un microscopiste qualifié ; noter le protocole (microscope, coloration, grossissement).
- Pour le split, le préfixe avant `_` identifie le lot : garder un lot entier dans le même jeu.

## Dans l'API

| Exigence | Mise en œuvre |
| --- | --- |
| Aucune image conservée | Traitement en mémoire uniquement, rien n'est écrit sur disque |
| Logs anonymisés (C5) | Horodatage, identifiant aléatoire, classe, score, seuil, version, latence. Pas de nom de fichier, pas d'IP, pas d'image |
| Rétention 12 mois | `PREDICTION_LOG_FILE` : rotation quotidienne, 365 fichiers conservés (`LOG_RETENTION_DAYS`), purge automatique |
| Secrets | `API_KEY` via variables d'environnement / secrets CI, jamais dans le dépôt |
