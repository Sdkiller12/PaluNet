# Interface web de démonstration (`web_app/`)

Page statique sans framework ni étape de build, servie par FastAPI sur `/` (même origine que
l'API, donc pas de CORS nécessaire). Désactivable avec `SERVE_WEB_APP=false`.

| Fichier | Rôle |
| --- | --- |
| `index.html` | Structure : bandeau d'avertissement, zone de dépôt, options, résultat, erreurs |
| `app.js` | Logique : santé de l'API, sélection et contrôle du fichier, appel `/v1/predict`, rendu |
| `styles.css` | Mise en forme (carte, barre de confiance, superposition Grad-CAM) |

## 1. Parcours utilisateur

1. Au chargement, `checkHealth()` appelle `GET /health` et affiche :
   - « API opérationnelle — modèle vX, seuil de décision Y » (statut `ok`) ;
   - ou « API indisponible : <raison> » (statut `ko`).
   Si `heatmap_available` est faux, la case Grad-CAM est désactivée.
2. L'utilisateur dépose une image (glisser-déposer) ou la choisit.
3. `selectFile()` contrôle côté client le type MIME (`image/png`, `image/jpeg`) et la taille
   (≤ 5 Mo). Le serveur refait ces contrôles : ceux du client servent seulement à répondre plus vite.
4. Clic sur « Analyser » : `submit()` envoie un `FormData` (`file`, `include_heatmap`) avec
   l'en-tête `X-API-Key` s'il est renseigné. Le bouton est désactivé pendant l'analyse.
5. `render()` affiche le résultat.

## 2. Affichage du résultat

| Élément | Source | Remarque |
| --- | --- | --- |
| Classe prédite | `prediction` | Libellé français : « Parasitée » (rouge) / « Saine » (vert) |
| Libellé du score | `prediction` | « Confiance que la cellule est « parasitée » » : le libellé précise la classe concernée (exigence D2) |
| Barre et pourcentage | `confidence` | Format français (`94,2 %`) |
| Probabilité d'infection | `probability_parasitized` | Toujours affichée, quelle que soit la classe |
| Avertissements | `quality_warnings` | Préfixés « Qualité d'image : » |
| Carte Grad-CAM | `heatmap_base64` | Image PNG RGBA superposée à l'aperçu |
| Méta | `model_version`, `processing_time_ms` | |

L'aperçu de l'image est créé localement (`URL.createObjectURL`) : l'image n'est pas renvoyée par
le serveur.

## 3. Messages d'erreur (D3)

`describeError(status, detail)` traduit les codes HTTP en messages compréhensibles :

| Code | Message |
| --- | --- |
| 400 | « Image refusée : <détail du serveur> » |
| 401 | « Accès refusé : clé API manquante ou invalide » |
| 413 | « Fichier trop volumineux : la taille maximale est de 5 Mo. » |
| 429 | « Trop de requêtes : merci de patienter une minute… » |
| 503 | « Le service d'analyse n'est pas prêt (modèle non chargé)… » |
| autre | « Erreur inattendue du serveur (<code>)… » |
| réseau | « Impossible de joindre l'API. Vérifiez votre connexion. » |

## 4. Clé API

Le champ « Clé API » (repliable) est mémorisé dans `localStorage` (`palunet_api_key`). Les
accès à `localStorage` sont protégés par `try/catch` (navigation privée, stockage bloqué).

> La clé est stockée en clair dans le navigateur. Acceptable pour une démonstration sur poste
> maîtrisé ; à revoir avant tout usage multi-utilisateur (authentification par session, par exemple).

## 5. Exigences d'interface couvertes

| Exigence | Mise en œuvre |
| --- | --- |
| D5 — avertissement visible | Bandeau `.disclaimer` en haut de page |
| D2 — score non ambigu | Libellé dynamique du score selon la classe prédite |
| D3 — erreurs compréhensibles | `describeError()` |
| D6 — explicabilité | Case Grad-CAM, superposition de `heatmap_base64` |
| Confidentialité | Pied de page : aucune image conservée, journaux anonymisés |

## 6. Modifier l'interface

- `API_BASE = ""` : l'API est sur la même origine. Pour servir l'interface ailleurs, renseigner
  l'URL de l'API ici **et** `CORS_ORIGINS` côté serveur.
- `MAX_BYTES` doit rester aligné sur `MAX_UPLOAD_MB`.
- Les libellés de classes sont dans `LABELS_FR`. Une nouvelle classe s'affiche sous son nom
  anglais tant qu'elle n'y est pas ajoutée.
