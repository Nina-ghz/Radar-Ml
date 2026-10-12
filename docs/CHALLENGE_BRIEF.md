# Challenge brief — RadarML Lite, Track B

Synthèse des documents officiels (`README`, `hackathon/README.md`, `EVENT_RULES.md`,
`docs/participant_notes.md` du dépôt organisateur) et du papier du dataset.
**En cas de doute, la source faisant foi est le dépôt officiel.**

## 1. Tâche

Classer **un pulse radar** enregistré (1 récepteur, ADC0) parmi 7 familles de modulation.
On n'estime ni distance, ni cible, ni émetteur : uniquement la famille de forme d'onde.

## 2. Données

| Propriété | Valeur |
| --- | --- |
| Configurations de forme d'onde | 660 (modulation × durée × bande × fréquence centrale) |
| Pulses par configuration | 100 (indices source 0–99) → 66 000 pulses |
| Format fichier | `*_recorded.npy`, int16, `(1, 100, 33600)`, I/Q entrelacés `I0,Q0,I1,Q1…` |
| Un pulse | 16 800 échantillons complexes à **120 MS/s** = **140 µs** (pulse + bruit autour) |
| Durées de pulse | 10, 50, 100 µs (centré dans la fenêtre de 140 µs) |
| Sidecar JSON | `source_file` (nom d'origine — **clé de la graine du bruit**), SHA256 |
| Nom de fichier | `{durée}_{modulation}_{bande}_{fréquences}_recorded.npy` |

Les signaux contiennent déjà le bruit réel du récepteur. Les 13 SNR demandés
(+30 → −30 dB, pas de 5 dB) sont obtenus en **ajoutant** du bruit blanc gaussien.

### Classes et déséquilibre (split d'entraînement)

| Classe | Configs | Pulses train | Remarque |
| --- | ---: | ---: | --- |
| `barker` | 72 | 5 760 | 6 sous-types (5, 7, 11, 13, 5×11, 5×13) fusionnés — **ne pas les séparer** |
| `cw` | 12 | **960** | classe minoritaire (×16 moins que fsk) |
| `fmcw` | 96 | 7 680 | |
| `fsk` | 192 | **15 360** | `fsk#0`…`fsk#3` fusionnés |
| `hyp` | 96 | 7 680 | |
| `nlfm` | 96 | 7 680 | |
| `quad` | 96 | 7 680 | |

⚠️ La métrique est une **balanced accuracy** : chaque classe pèse autant. Le
déséquilibre (cw vs fsk) doit être géré (pondération de la loss, échantillonnage…).

D'après le papier : chirps linéaire / exponentiel / quadratique / tangentiel (NLFM),
CP-FSK à 4 fréquences, codes de Barker (sauts de phase de 180°). Les organisateurs
demandent toutefois de ne pas présumer de la correspondance exacte label ↔ équation :
apprendre à partir des exemples.

## 3. Split (fixe, imposé)

| Split | Pulses source | Exemples à 13 SNR |
| --- | ---: | ---: |
| train | 52 800 (80 / config) | 686 400 |
| validation | 13 200 (20 / config) | 171 600 |
| blind test (séparé, Day 5) | 13 200 | 171 600 |

- Utiliser **uniquement** `data/manifest/split_manifest.csv` (SHA256 vérifié). Ne jamais recréer de split.
- Toutes les variantes SNR / augmentations d'un pulse restent dans son split.
- Normalisations / features apprises : fit sur **train uniquement**.
- Ne **pas** ajouter la validation dans l'entraînement final.
- Train et validation partagent les configurations : on mesure la reconnaissance de
  nouveaux pulses de configurations connues (pas la généralisation à de nouvelles configs).

## 4. SNR : pièges à connaître

- Le SNR est estimé sur des **fenêtres fixes** : bruit 0–20 µs, signal 21–119 µs (98 µs).
  La puissance d'un pulse court est diluée dans cette fenêtre, donc le SNR *dans* le pulse est
  **plus élevé** que le nominal : **+9,9 dB** (10 µs), **+2,9 dB** (50 µs), 0 dB (100 µs).
  L'énergie totale du pulse rapportée au bruit est la même pour les trois durées
  (mesuré dans `notebooks/01_eda.ipynb`, section 3).
- Si le SNR demandé dépasse le SNR source, **aucun bruit n'est ajouté** (≈ 1/3 des cas à +30 dB).
- Écrêtage int16 : 5,38 % des valeurs I/Q à −30 dB, 0,24 % à −25 dB.
- Toujours parler de **« requested SNR (dB) »** dans les figures et le rapport.
- `radarml.noise.degrade_official` reproduit exactement l'expansion officielle
  (même fonction, même graine `md5("42_{source_file}_{snr}_{pulse}")`).

## 5. Track B — ce qui est autorisé

| Autorisé | Interdit |
| --- | --- |
| STFT / spectrogrammes ou autre représentation temps-fréquence calculée depuis l'I/Q | Utiliser nom de fichier, ID, métadonnées, SNR demandé comme entrée |
| Choix libre du modèle et des paramètres de représentation | Poids pré-entraînés, extracteurs appris externes |
| Augmentations dérivées du train qui préservent la classe | Données externes, autres enregistrements RadarML |
| Normalisation fixe par pulse (ex. par son propre RMS) | Recadrage basé sur la durée lue dans le nom de fichier |
| Librairies, papiers, code public, assistants IA (**à citer / déclarer**) | Envoyer les signaux de test à un service externe |

**Le calcul de la représentation TF doit être inclus dans le temps d'inférence.**

## 6. Métrique

Pour chaque SNR demandé : rappel moyen sur les 7 classes (balanced accuracy).
Score principal = moyenne sur les 13 SNR (poids égal par classe et par SNR).
Tie-break = moyenne à −20, −25, −30 dB (tolérance 1e-6). Référence « classe constante » ≈ 14,3 %.
Implémentation : `radarml.metrics.challenge_report` (miroir de `score_submission.py`).
Confusion matrix : lignes = vrai, colonnes = prédit, ordre alphabétique des classes.

## 7. Calendrier (Europe/London)

| Quand | Quoi |
| --- | --- |
| Day 0, 09:00 | Kick-off, accès aux données, choix du track |
| Day 1 | Premier pipeline complet + prédictions validation |
| Day 2 | Expériences signal / modèle (validation uniquement) |
| Day 3 | Robustesse par SNR, confusions, temps & mémoire |
| **Day 4, 17:00** | **Freeze** : modèle, preprocessing, config, versions, commande d'inférence, hashes SHA256 |
| **Day 5, 09:00** | Publication du blind test (~11,55 Go ; prévoir ~23,1 Go) |
| **Day 6, 17:00** | **Soumission finale** `predictions.csv` + rapport |

Après le freeze : aucun réentraînement, tuning de seuil, sélection de modèle ou adaptation au test.
Une seule entrée finale par track. Aucun retour de score avant la clôture.

## 8. Blind test (à partir du Day 5 seulement)

- **Ne pas télécharger `blind_test_participants.zip` avant l'annonce**, même s'il est visible.
- SHA256 du ZIP : `7d13639eea4aa936ad43f815d97b9d0a60cbaaf8fade6a472306c75a70223eab`.
- Shards anonymes `(N, 33600)` + `index.csv` ; chargement via
  `third_party/radarml_lite/participant_loader.py` (`Samples("test")`).
- Sortie : `predictions.csv` avec exactement `sample_id,predicted_label`, un ID par ligne,
  vérifier avec `scripts/validate_submission.py`.
