# System roadmap — beyond the score model

The score model (`src/pipelines/` → `src/fusion/` → `src/safety/`) is
code-complete; see `docs/data-collection-tool-migration.md`. This doc tracks
the rest of the system: backend, data platform, web app, and the
GenUI/RAG layer on the score modal. Target infra is **Firebase**
end-to-end (Cloud Run, Firestore, Storage, Auth, Hosting) — Colab remains
training-only. This supersedes the README's earlier Railway/Supabase line.

## Phase 1 — Backend API + local dev stack — **done (partial)**

Delivered:
- `api/main.py` (FastAPI): create participant/session, submit PHQ-9/HAM-D
  item responses (referral flag computed immediately, independent of the ML
  call), upload audio, run scoring, fetch results.
- `api/storage.py`: local JSON document store, deliberately Firestore-shaped
  (collection/doc, get/set/list/update) so Phase 4 swaps the implementation
  only — no call-site changes in `api/main.py`.
- `api/genui.py`: score-modal narrative (Claude via `ANTHROPIC_API_KEY`,
  template fallback otherwise).
- Reuses the pipeline **unchanged**: uploads are staged into the same
  on-disk session-folder shape (`{id}_TRANSCRIPT.csv`, `{id}_AUDIO.*`,
  `{id}_METADATA.json`) that `sync.py`/`text_pipeline.py`/`audio_pipeline.py`/
  `metadata_pipeline.py` already read, under `data/live/sessions/`.
- `tests/test_api.py`: full local flow, isolated per-test via `tmp_path`.
- `docker/Dockerfile` + `docker-compose.yml` updated: `api` service (default,
  `docker compose up`) and `inference` batch service (`--profile batch`).

Not yet done (deferred, not blocking):
- **Firebase Local Emulator Suite** was scoped but not wired up — Phase 1
  ships with the plain local JSON store instead, which needs zero extra local
  tooling. Swapping `api/storage.py`'s `Store` for
  `firebase_admin.firestore.client()` (pointed at the emulator locally, the
  real project in prod) is Phase 4 work, not a Phase 1 gap.
- Real-time chunked/streaming audio upload (see Phase 2) — the current
  `/sessions/{id}/audio` endpoint takes one full-file upload.

## Phase 2 — Live audio capture + transcription

- `src/pipelines/asr_pipeline.py` (Whisper, mock fallback) — **built**, but
  only exercised via single full-file upload so far.
- Still open: chunked/streaming upload from the browser (`MediaRecorder`,
  ~30s chunks matching `SEGMENT_LEN_SEC`), a Storage-triggered transcription
  kickoff, and a push-to-talk speaker-label toggle in the UI (reuses
  `sync.INTERVIEWER_SPEAKER_NAMES` unchanged). True real-time streaming
  captions (as opposed to per-chunk transcription) is a further-out stretch.

## Phase 3 — Clinician/patient web app: score modal + GenUI narrative

- `web/index.html` is a **deliberately minimal stand-in**: vanilla HTML/JS,
  no build step, proves the full API contract end-to-end (including a working
  score modal with XAI bars, the GenUI narrative, and a referral banner that's
  visually and logically independent of the narrative/model).
- Still open: the real Next.js app (Firebase Auth login, Firebase Hosting),
  live recording UI (depends on Phase 2's chunk upload), the full participant
  intake / consent / Section-D-prompt-checklist screens, and WCAG 2.1 AA
  coverage (axe-core in CI) — methodology.txt's SUS ≥ 80 objective.

## Phase 4 — Firebase production deploy

Not started. When picked up:
- Cloud Run hosts the containerized API (same `docker/Dockerfile`, already
  built for this — no separate deploy-specific image needed).
- Firestore replaces `api/storage.py`'s local JSON store (same collection
  names, see `Store` interface). Security Rules keep PII/site/tribe out of
  the ML-facing read path — mirrors `metadata_pipeline.py`'s existing
  exclusion of `site` from the feature vector.
- Firebase Storage for audio/transcripts; Firebase Auth for clinician/RA
  accounts; trained `outputs/weights/fusion_head.npz` uploaded there and
  pulled by Cloud Run at startup.
- CI: `pytest` (already green, 22 passed / 1 skipped) + new web tests,
  deploying on merge.

## Phase 5 — RAG-grounded GenUI (DSM-5 retrieval)

Not started; scoped:
- Corpus: short **paraphrased** DSM-5 MDD criteria + PHQ-9/HAM-D item
  definitions (DSM-5 text itself is APA copyrighted — don't embed it
  verbatim).
- Index: a small local FAISS/Chroma index bundled into the Cloud Run
  container (corpus is small + static, no managed vector DB needed yet;
  Vertex AI Vector Search is the fallback if the team wants fully
  Firebase-native later).
- Retrieval flow: embed the participant's item-level responses + top
  XAI-attributed modality, retrieve top-k DSM-5 snippets, feed them into
  `api/genui.py`'s prompt as grounding context.
- Safety boundary unchanged: RAG/GenUI output stays explanatory-only, never
  feeds back into `phq9_pred`/`hamd_pred`/`binary_pred`/`risk_flag`.

## Phase 6 — remaining methodology.txt extras (stretch)

- Fairness/bias metrics (disparate impact by metadata subgroup) — extend
  `src/eval/metrics.py` once real labels exist.
- LoRA fine-tuning of BERT/Wav2Vec2 — **conflicts with the locked "frozen
  backbones" decision** in the README; needs an explicit decision to revisit,
  not a silent implementation.
- Opacus differential privacy, external DAIC-WOZ validation run, 20-
  psychiatrist ICC clinical validation — Phase-2-of-the-methodology-proposal
  / study-protocol scope, not code.
