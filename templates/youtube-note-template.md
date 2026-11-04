---
date: YYYY-MM-DD
source: "<channel>, <publication date>"  # MUST be quoted — may contain `:`
source_url: https://youtube.com/watch?v=VIDEO_ID
source_files:
  video: http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/videos/<slug>.webm
  audio: http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/videos/<slug>.mp3
  transcript: http://vmi3304846.tail5c02a1.ts.net:9000/knowledge-base/videos/<slug>.json
confidence: plausible
tags: [tag1, tag2, tag3]
---

# Video Title

> **Language:** Content in the source language. Section labels are ALWAYS in English
> (Metadata, Summary, Chapters, Key Concepts…) regardless of the video's language.

## Metadata
- **Channel / Speaker(s):** ...
- **Duration:** XX min
- **Language:** ...
- **Published:** ...
- **Views:** ...

## Summary
3-4 dense sentences. The central thesis + the key mechanisms invoked.
Not "the microbiome is important" but "the microbiome drives inflammation
via metabolites X that modulate pathway Y".

## Chapters

| Timestamp | Title | Key claims | Mechanisms / Data |
|-----------|-------|-----------|-------------------|
| 00:00 | Introduction | The 2-3 main claims | The "how": figures, studies, protocols |
| 05:23 | The problem | ... | ... |

If a chapter is purely anecdotal (personal testimony), note it explicitly:
"(personal testimony, no data)". Redundancy with Key Concepts is intentional —
two different reading paths (thematic vs chronological).

## Chapter Summary
Detailed summary of each chapter with its main claims.

## Key Concepts
3-6 cross-cutting concepts, each developed individually. Each concept = a
sub-heading with 2-4 sentences explaining the mechanism, not just the conclusion.

Format per concept:
### Concept — explanation of the mechanism
What is claimed, how it works, data/figures/studies cited.
Practical implication if mentioned.

## Speaker Summary

| Speaker | Time | Topics | Position |
|---------|------|--------|----------|
| SPEAKER_00 (~XX min) | Topic 1, Topic 2 | Thesis defended | |
| SPEAKER_01 (~XX min) | Topic 3, Topic 4 | Counter-point if relevant | |

- If real names are known (identified from metadata), use them
- If >5 speakers, keep only the 5 most active
- If a speaker is purely reactive (short questions), note it
- `⚠️ Overlap` segments are not counted in speaking time

## Key Points
8-15 points. Each point must be actionable or contain a mechanism.
Format: "[Claim] — [Mechanism/Evidence]"
No generalities like "it's important", "listen to your body".

## Nuances & Limitations
- What the speaker presents as certainty vs hypothesis
- Studies/papers mentioned (even approximately: "a 2023 Harvard study…")
- Internal contradictions or extraordinary unsupported claims
- No external fact-checking — flag what the video itself presents or omits, without verifying

## Useful Extractions

### Quotes
> "exact quote" (HH:MM:SS)

### Protocols / Methods
- Dosages, routines, concrete practices mentioned

### External References
- Researchers, books, studies, institutions cited

## Transcript

Continuous dialogue format with timestamps. Full transcript at the end of the note (appendix).

```
*(HH:MM:SS)* **SPEAKER_00 (Name if known)** : text of the intervention...

*(HH:MM:SS)* **SPEAKER_01** : response...

⚠️ *(HH:MM:SS)* **SPEAKER_00 | SPEAKER_01** *(overlap)* : less reliable text...
```

- Real names replace labels if known (e.g. `SPEAKER_00 (Paul)`)
- `overlap: true` segments are prefixed with `⚠️` and annotated `*(overlap)*`
- No table — fluid dialogue for reading
- Full timestamps are in the MinIO JSON; in the note, rounded to the second

## See Also
- [[Related note]]
