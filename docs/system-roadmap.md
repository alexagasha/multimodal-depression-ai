# System roadmap — beyond the score model

The score model (`src/pipelines/` → `src/fusion/` → `src/safety/`) is
code-complete; see `docs/data-collection-tool-migration.md`. This doc tracks
the rest of the system: backend, data platform, web app, and the
GenUI/RAG layer on the score modal. Target infra is **Firebase**
end-to-end (Cloud Run, Firestore, Storage, Auth, Hosting) — Colab remains
training-only. This supersedes the README's earlier Railway/Supabase line.

## Phase 1 — Backend API + local dev stack — **done (partial)**

Delivered:
- `api/main.py` (FastAPI): create participant/session, list/get patients
  (`GET /participants`, `GET /participants/{id}`, enriched with visit
  count + latest scores for the roster), submit PHQ-9/HAM-D item responses
  (referral flag computed immediately, independent of the ML call), upload
  audio, run scoring (returns the clinician-entered totals alongside the AI
  estimate), fetch results.
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

## Phase 3 — Clinical-practice web app: patient roster + score modal — **done**

The app is deliberately framed as a **clinical-practice tool**, not a
research data-collection instrument: patients are a first-class, searchable
entity with a visit history (like a lightweight EHR module), the AI runs
**alongside** the clinician's own PHQ-9/HAM-D scoring as a second opinion
rather than replacing it, and the core loop is "pull up a patient, start
their next visit" — not a one-off intake form. Frontend routes/labels speak
patient/visit; the backend's underlying collections stay
participant/session (see `web/lib/api.ts`'s docstring for the translation
layer).

Delivered — a real Next.js (App Router) + TypeScript + Tailwind app in
`web/`, organic sage/clay theme, hand-authored SVG assets (leaf mark, blobs,
empty state), no external image/font fetching at runtime:
- **Patient roster** (`app/page.tsx`, the landing page) — every registered
  patient enriched with visit count, last-visit date, and their latest
  visit's risk_flag/scores (`GET /participants`, server-side joined +
  risk-first sorted, mirroring the same pattern `GET /sessions` already
  used). Client-side search by patient ID.
- **Register patient** (`app/patients/new/page.tsx`, was `/intake`) —
  consent + Section A form, draft autosave to localStorage.
- **Patient detail** (`app/patients/[id]/page.tsx`) — demographics summary
  card (Section A fields, captured at registration but previously never
  shown back), the severity trend chart, full visit history, and a primary
  **"Start new visit"** action — the loop the app is built around.
- **Visit workspace** (`app/visits/[id]/page.tsx`, was `/sessions/[id]`) —
  status stepper, `ScaleForm` (PHQ-9/HAM-D with a per-item **read-aloud**
  button via the browser's `speechSynthesis` API — low-literacy
  accessibility, methodology's WCAG objective), `AudioRecorder` (Phase 2),
  `ClinicalNotes`.
- **Score modal** (`components/ScoreModal.tsx` + `Gauge.tsx`) — each gauge
  shows the **AI estimate alongside the clinician's own administered
  score** (`phq9_clinician`/`hamd_clinician`, added to the `/score` response
  from the already-submitted `ScaleForm` data — no extra round-trip), XAI
  attribution bars, the GenUI narrative in its own labeled card, a
  **clinician review/adjust** action, and a **Print / Save PDF** button
  (print stylesheet in `globals.css`). The referral banner
  (`components/RiskBanner.tsx`) is deliberately unthemed and independent of
  the narrative/model, per the safety boundary in `api/genui.py`.
- **Symptom-subtype differential** (`api/subtype_differential.py` +
  `components/SubtypeDifferential.tsx`) — reads the transcript against four
  paraphrased DSM-5 specifier criteria (melancholic, atypical, anxious
  distress, psychotic features) and returns a qualitative likelihood +
  rationale per subtype, decision support for treatment-direction choice
  aimed specifically at a depression specialist (not a generalist symptom
  checker). Psychotic-features "possible"/"present" gets a distinct amber
  callout — clinically significant, but still AI-suggested, never conflated
  with the rule-based referral banner. Computed at scoring time, stored
  alongside the prediction. **No non-LLM fallback by design** — unlike the
  narrative, a subtype call requires actually reading the transcript, so it
  returns `null`/"unavailable" without `ANTHROPIC_API_KEY` rather than
  fabricating a differential. Same static-criteria-inlined-in-prompt
  approach as Phase 5 anticipates, just scoped to one feature now instead
  of full retrieval infrastructure.
- **Severity trend** (`components/TrendSparkline.tsx`) — plain SVG, no
  charting library, across a patient's full visit history
  (`GET /participants/{id}/sessions`).
- **Clinical notes** — append-only (`api/main.py` `POST/GET
  /sessions/{id}/notes`), matching real clinical documentation practice
  (amend via a new note, never edit history).
- Verified via `npm run build` (clean) and two live browser walkthroughs,
  which caught and fixed real bugs both times: a nested-`<a>`-inside-`<a>`
  hydration error on the roster (fixed with the standard "stretched link"
  pattern), an unhandled `FileNotFoundError` when scoring before any audio
  upload (now a clean 400), and a demographics-display bug where `age_band`
  ("26-35") lost its hyphen to an overly broad humanize-for-display regex.

Still open: Firebase Auth login, Firebase Hosting deploy, full WCAG 2.1 AA
audit (axe-core in CI) — the UI follows accessible patterns (semantic
labels, ARIA on the risk alert, read-aloud) but hasn't been formally audited.

## Phase 3b — GenAI dashboard features — **done (10/10)**

The full 10-feature spec a depression-specialist psychiatrist would want
from a GenAI-powered dashboard (source: `generative ai.txt`, originally
proposed then approved as the build spec). Feature 3 (subtype differential)
shipped in Phase 3 above; the other nine shipped in one pass. Shared
infrastructure: `api/llm_utils.py` centralizes the Claude-call-with-
honest-fallback convention (`call_claude()` returns `None` on any failure —
no API key, bad JSON, network error) so every module below returns `None`/
"unavailable" rather than fabricating output. Two of the nine are
deliberately **rule-based, not LLM** — safety-adjacent trend signals
shouldn't depend on model availability or be prone to hallucination, same
philosophy as `src/safety/risk_flag.py`.

1. **AI-drafted clinical note** (`api/note_draft.py`,
   `components/NoteDraftPanel.tsx`) — on-demand SOAP-format draft from the
   transcript + scores via `POST /sessions/{id}/note-draft`; the clinician
   edits inline and saves through the existing notes endpoint. The draft is
   never auto-submitted or stored on its own.
2. **Explainable score — evidence quotes** (`api/evidence.py`,
   `components/EvidenceQuotes.tsx`) — up to 3 exact transcript phrases
   behind the dominant modality's attribution, appended to the `/score`
   result as `evidence`. Not a literal saliency map (mock/frozen encoders
   don't support meaningful gradient attribution yet) — an honest LLM read
   grounded only in what was actually said.
4. **Relapse early-warning** (`api/trends.py::relapse_warning`) — **rule-
   based**: flags a monotonically non-improving HAM-D trend across the last
   3 scored visits with a ≥3-point delta, surfaced as `relapse_warning` on
   `GET /participants` and an amber banner on the patient detail page.
5. **Treatment-response overlay** (`treatment_events` collection,
   `POST/GET /participants/{id}/treatments`, `components/TreatmentEvents.tsx`,
   `TrendSparkline.tsx`) — medication/therapy-change events, snapped to the
   nearest visit by date and plotted as dashed markers on the severity
   trend (visit-index-based chart, not a true continuous time axis).
6. **Risk trajectory** (`api/trends.py::risk_trajectory`) — **rule-based**:
   flags 2+ referral-flagged visits in the last 3, distinct from a single
   flagged visit (already handled by `RiskBanner`) — a repeating pattern,
   surfaced the same way as relapse_warning.
7. **Guideline-grounded next-step suggestions** (`api/treatment_suggestions.py`)
   — 2-3 advisory considerations ("consider"/"may warrant," never a
   directive) from paraphrased, non-proprietary STAR\*D-style stepped-care
   heuristics + DSM-5 subtype-treatment matching, inlined in the prompt —
   same lightweight approach as the subtype differential, not full Phase 5
   retrieval.
8. **Natural-language caseload query** (`api/caseload_query.py`,
   `POST /query`) — the roster JSON passed as context, no function-calling
   infrastructure needed at local-demo scale; matching patients get
   highlighted on the roster page. Returns 503 (not a silent empty result)
   when unavailable, since it's a deliberate user action.
9. **Practice-level analytics** (`GET /analytics`, `app/analytics/page.tsx`)
   — patient/visit counts, referral-flag rate, caseness rate, and a PHQ-9
   severity-band distribution, all computed deterministically; an optional
   GenAI narrative (`api/analytics_summary.py`) summarizes the numbers in
   plain language on top, grounded strictly in what's passed to it.
10. **Patient-facing after-visit summary** (`api/patient_summary.py`,
    `components/PatientSummaryCard.tsx`) — plain-language, ~4th-grade
    reading level, no clinical jargon or scale names, gentle (not alarming)
    handling when a referral flag is active. Pairs with the same
    `speechSynthesis` read-aloud pattern already in `ScaleForm.tsx`.

Verified via `pytest` (46 passed / 1 skipped — includes the two rule-based
trend functions tested directly with synthetic visit sequences, and every
LLM module's unavailable-without-key path), `npm run build`/`lint` (clean),
and a live browser walkthrough exercising all nine: roster query box,
patient trend + treatment overlay, note-draft panel, and every score-modal
section, confirming honest "unavailable" rendering with zero console errors
in the absence of `ANTHROPIC_API_KEY`.

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
- CI: `pytest` (already green, 46 passed / 1 skipped) + `npm run build` /
  `npm run lint` (already green) + new web tests, deploying on merge.

## Phase 5 — RAG-grounded GenUI (DSM-5 retrieval)

Not started; scoped. Note: Phase 3b's `api/subtype_differential.py` and
`api/treatment_suggestions.py` already use this phase's *approach*
(paraphrased criteria inlined directly in the prompt) at a scope that
doesn't need real retrieval yet — this phase is what to reach for once the
corpus grows past what fits inline (full DSM-5, real clinical guidelines,
per-patient literature).
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
