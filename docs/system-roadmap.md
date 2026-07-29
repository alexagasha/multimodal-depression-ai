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

## Phase 2 — Live audio capture + transcription — **done (partial)**

Delivered:
- `src/pipelines/asr_pipeline.py` (Whisper, mock fallback).
- `web/components/AudioRecorder.tsx`: **real live mic recording** —
  `getUserMedia` + `AudioContext` + `ScriptProcessorNode` captures raw PCM,
  hand-encoded into a 16-bit WAV `Blob` on stop (browsers don't encode WAV
  directly via `MediaRecorder`), uploaded through the existing `/audio`
  endpoint unchanged. File-upload fallback when the mic isn't available.

Still open:
- **True chunked/streaming upload** — today's recorder is "record live, one
  upload on stop," not incremental ~30s chunks during the interview. A
  Storage-triggered transcription kickoff and a push-to-talk speaker-label
  toggle (reuses `sync.INTERVIEWER_SPEAKER_NAMES` unchanged) still need
  building for that.
- `ScriptProcessorNode` is deprecated (still universally supported); an
  `AudioWorklet` upgrade is a drop-in swap when picked up.
- True real-time streaming captions (vs. per-session transcription) — a
  further-out stretch.

## Phase 3 — Clinician/patient web app: score modal + GenUI narrative — **done**

Delivered — a real Next.js (App Router) + TypeScript + Tailwind app in
`web/`, organic sage/clay theme, hand-authored SVG assets (leaf mark, blobs,
empty state), no external image/font fetching at runtime:
- **Triage dashboard** (`app/page.tsx`) — risk-flagged sessions first, then
  by severity (`GET /sessions`, server-side enriched + sorted).
- **Intake** (`app/intake/page.tsx`) — consent + Section A form, draft
  autosave to localStorage.
- **Session workspace** (`app/sessions/[id]/page.tsx`) — status stepper,
  `ScaleForm` (PHQ-9/HAM-D with a per-item **read-aloud** button via the
  browser's `speechSynthesis` API — low-literacy accessibility, methodology's
  WCAG objective), `AudioRecorder` (Phase 2), `ClinicalNotes`.
- **Score modal** (`components/ScoreModal.tsx`) — gauges, XAI attribution
  bars, the GenUI narrative in its own labeled card, a **clinician
  review/adjust** action, and a **Print / Save PDF** button (print
  stylesheet in `globals.css`). The referral banner
  (`components/RiskBanner.tsx`) is deliberately unthemed and independent of
  the narrative/model, per the safety boundary in `api/genui.py`.
- **Participant trend page** (`app/participants/[id]/page.tsx`) —
  longitudinal PHQ-9/HAM-D view across a participant's repeat sessions
  (`TrendSparkline.tsx`, plain SVG, no charting library), plus a "follow-up
  session for this participant" action.
- **Clinical notes** — append-only (`api/main.py` `POST/GET
  /sessions/{id}/notes`), matching real clinical documentation practice
  (amend via a new note, never edit history).
- Verified via `npm run build` (clean) and a live browser walkthrough, which
  caught and fixed two real bugs: a nested-`<a>`-inside-`<a>` hydration error
  on the dashboard (fixed with the standard "stretched link" pattern) and an
  unhandled `FileNotFoundError` when scoring before any audio upload (now a
  clean 400).

Still open: Firebase Auth login, Firebase Hosting deploy, full WCAG 2.1 AA
audit (axe-core in CI) — the UI follows accessible patterns (semantic
labels, ARIA on the risk alert, read-aloud) but hasn't been formally audited.

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
- CI: `pytest` (already green, 31 passed / 1 skipped) + `npm run build` /
  `npm run lint` (already green) + new web tests, deploying on merge.

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
