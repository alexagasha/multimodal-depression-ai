# Data retention

What the system keeps, what it may discard, and who decides.

## The distinction that matters

Clinical documentation guidance separates **transient processing artefacts**
from the **record**. Raw interview audio and interim transcripts are the
former: they exist so a note can be produced. The signed note and its
provenance metadata are the latter, and are what the record consists of.

That distinction licenses discarding audio after a visit closes. It does not
require it, and in this project it mostly should not happen — see below.

## What the system holds

| Artefact | Location | Class | Default |
|---|---|---|---|
| Raw interview WAV | `data/live/sessions/{id}/{id}_AUDIO.*` | processing artefact | **kept** |
| Transcript CSV | `data/live/sessions/{id}/{id}_TRANSCRIPT.csv` | derived, but load-bearing | kept |
| Metadata JSON | `data/live/sessions/{id}/{id}_METADATA.json` | derived | kept |
| Clinician scores | `store/scale_responses/` | record | kept |
| Model prediction + XAI | `store/predictions/` | record | kept |
| Signed clinical notes | `store/clinical_notes/` | **record** | kept permanently |
| Risk assessments | `store/risk_assessments/` | **record** | kept permanently |

## The control

`DEP_RETAIN_AUDIO` — defaults to `1` (retain everything).

Set it to `0` and closing a visit deletes that visit's raw audio, stamping
`audio_discarded_at` on the session. Nothing else is removed. In particular
**the transcript is always kept**: the note, the evidence quotes and the risk
record were all derived from it, and deleting it would orphan every one of
them.

Deletion only ever happens on an explicit `POST /sessions/{id}/close`. There
is no background sweep, no age-based expiry, and no deletion of anything the
clinician wrote.

## Why the default is to keep everything

This is not a deployed clinical service. It is a study, and **for the research
cohort the recordings are the dataset** — the model is fitted on prosody
computed from those waveforms, and the transcript verification pass is not
finished. Discarding audio would destroy the ability to refit, re-verify, or
answer a reviewer's question about a feature.

Turning retention off is therefore a **study-protocol decision, not a
deployment convenience.** Before setting `DEP_RETAIN_AUDIO=0` anywhere that
handles real participant recordings, check:

1. the approved IRB protocol's stated retention period and destruction plan;
2. what participants were told during consent;
3. Uganda's data-protection obligations as they apply to the sponsoring
   institution.

None of those are settled by this document, and the code deliberately does not
assume an answer.

## What is not addressed here

- **Hosted ASR.** With `DEP_ASR_PROVIDER` set to a cloud provider, interview
  audio leaves the machine before any of this applies. That is a separate
  governance question, flagged in `src/pipelines/asr_pipeline.py`, and no
  retention setting here constrains what a third party keeps.
- **Backups and the store's durability.** `api/storage.py` is a local JSON
  stand-in for Firestore with no concurrency control; retention semantics on a
  real backing store are a Phase 4 concern.
- **De-identification.** The research export is already de-identified. The live
  store is not — it holds names as note authors and free-text clinical detail.
