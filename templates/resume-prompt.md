# Prompt de résumé vidéo — Passe 2 (RESUME+NOTE+ARCHIVE)

Utilisé par le worker `researcher-videos` pour produire une note approfondie
dans `Knowledge base/` à partir d'un transcript JSON.

**Transcript format:** Chaque segment contient `speaker` (SPEAKER_00, SPEAKER_01…) et optionnellement `overlap: true`.
Les étiquettes speaker sont continues sur toute la vidéo. Les labels sont à utiliser tels quels
dans la note (SPEAKER_00, SPEAKER_01…), sauf si un mapping vers des noms réels est fourni
dans la clé `speaker_names` du JSON (issu de l'étape d'identification automatique).
Les segments avec `overlap: true` représentent des chevauchements (deux personnes parlant
simultanément) — transcription moins fiable.

## PASSE 1 — Extraction des concepts clés

Lis l'intégralité du transcript. **Ne produis QUE la liste de concepts.**
Pas de note, pas de chapitres, pas de résumé.

Pour chaque concept transversal identifié :

```
- Titre : nom court et descriptif
- Mécanisme : 2-4 phrases expliquant le "comment" derrière le concept,
  ancrées dans des segments spécifiques du transcript
- Timestamps : 2-3 moments où ce concept apparaît ou est développé
- Claim principal : l'affirmation centrale associée
- Type d'évidence : étude_citée | raisonnement | témoignage | assertion
```

Nombre de concepts :
- Vidéo ≤45 min → 3-6 concepts
- Vidéo 45-90 min → 5-8 concepts
- Vidéo 90+ min → 8-10 concepts

## PASSE 2 — Note finale

En utilisant les concepts extraits en Passe 1 + le transcript complet, rédige la note
en suivant **le modèle `templates/youtube-note-template.md`**. L'ordre des sections, les
libellés (anglais) et le format de chaque section sont définis dans le modèle — suis-le
fidèlement, ne réinvente pas la structure ici.

- **Contenu** : dans la langue DE LA VIDEO (anglais → anglais, français → français). Ne traduis jamais.
- **Libellés de section** : TOUJOURS en anglais (Metadata, Summary, Chapters, Key Concepts…), quelle que soit la langue source.
- **Transcription** : annexe en fin de note (format dialogue continu, `*(timestamp)* **SPEAKER_XX** : texte`). Le lecteur doit trouver l'essentiel (résumé, concepts) en premier, la transcription brute en dernier.

## Qualité

- **Ancrage** : chaque claim doit pouvoir être rattaché à un segment du transcript. Pas de paraphrase vague.
- **Densité** : une note de 29 min devrait faire ~5-8 KB de markdown. Si elle fait 2 KB, c'est trop superficiel.
- **Langue** : contenu dans la langue DE LA VIDEO, libellés en anglais. NE TRADUIS JAMAIS le contenu.
- **Ne pas halluciner de contenu** : si une section n'a pas de matière (ex: aucune référence externe citée), le dire sobrement ("No external references cited in the video").
