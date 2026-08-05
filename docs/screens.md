# Web app screens

Six screens in `web/app/` (Next.js App Router — one `page.tsx` per route) plus
one modal overlay that isn't a route of its own. Nav only links to the three
top-level ones (`NavShell.tsx`); the rest are reached by drilling in from
those.

## 1. Patient roster — `/` (`web/app/page.tsx`)

The landing screen and the core loop's entry point — everything else is
reached from here.

- Lists every registered patient (`GET /participants`), referral-flagged
  patients first, then sorted by the severity of their latest visit.
- Each row: patient ID, visit count, last-seen date, latest PHQ-9/HAM-D,
  a pulsing **REFERRAL** badge if their most recent visit tripped the
  safety flag, and **Worsening trend** / **Rising risk pattern** badges from
  the rule-based relapse/risk-trajectory checks. Click a row to open the
  patient's detail page.
- Client-side search filters the list by patient ID (no name field —
  metadata is de-identified by design).
- **"Ask your caseload"** box: a free-text question (e.g. *"which patients
  haven't improved in 3 visits?"*) sent to the LLM-backed caseload-query
  endpoint; the answer only reasons over the roster already shown, and
  matching patients get highlighted. Reports "unavailable" with no fallback
  if `ANTHROPIC_API_KEY` isn't set.
- **"+ Register new patient"** button → screen 2.
- Empty state (no patients yet) shows an illustration and the same
  register-patient CTA.

## 2. Register patient — `/patients/new` (`web/app/patients/new/page.tsx`)

A short intake form, not a full session — it exists purely to create the
patient record and immediately hand off to their first visit.

- Informed-consent copy + a required checkbox before the form can submit.
- Section-A demographic fields (age band, sex, marital status, ethnicity,
  residence, education, employment, smartphone ownership, site) — the same
  fields the real PHQ-9/HAM-D instrument collects, kept separate from the
  scoring pipeline's inputs.
- Draft is autosaved to `localStorage` (`useDraftAutosave`) so a refresh
  doesn't lose in-progress entry.
- On submit: registers the patient, immediately starts their first visit,
  and redirects straight into screen 4 (`/visits/{session_id}`) — there's no
  intermediate "patient created" screen.

## 3. Patient detail — `/patients/[id]` (`web/app/patients/[id]/page.tsx`)

A patient's full record — the hub for a returning patient.

- Demographics card (everything captured at registration, read back).
- Active referral-flag banner and worsening-trend / rising-risk-pattern
  banners, if applicable to their latest visit.
- **Severity over time**: `TrendSparkline` — PHQ-9 and HAM-D line charts
  across all scored visits, with treatment events (medication/therapy
  changes) plotted as markers snapped to the nearest visit.
- Treatment-events list (`TreatmentEvents`) — add/view medication or
  therapy-change entries that feed the trend overlay above.
- Full visit history as a list — each row links to that visit's workspace
  (screen 4) and shows its date, PHQ-9/HAM-D, and referral badge if flagged.
- **"+ Start new visit"** button — the primary action, creates a new visit
  for this patient and redirects into it.

## 4. Visit workspace — `/visits/[id]` (`web/app/visits/[id]/page.tsx`)

Where a visit is actually conducted — the longest, most stateful screen.
Recording, transcription, and note-drafting all run **concurrently**: the
clinician presses Record, and the transcript and AI note assemble themselves
live; pressing Stop finalizes and scores automatically. The only buttons in
the flow are Record and Stop.

- Status stepper showing where the visit is (scales / audio / scored).
- Referral-flag banner (`RiskBanner`) if the safety flag is already active
  for this visit — fires independently of the AI model, the instant a
  PHQ-9 item 9 / HAM-D suicide-domain answer crosses threshold.
- **PHQ-9/HAM-D scale form** (`ScaleForm`) — clinician-administered scoring,
  item-by-item, with optional read-aloud per item for accessibility.
- **Audio recorder** (`AudioRecorder`) — live mic capture via
  `getUserMedia`/`AudioContext` (WAV, with a file-upload fallback), and a
  live waveform drawn from an `AnalyserNode` tap while recording.
  **Streams rather than batching**: every ~6s the audio captured since the
  last send is posted to `/audio/chunk`, so transcription and the AI note
  build up *during* the interview. On Stop it finalizes and scoring starts
  on its own.
- **Live transcript** (`LiveTranscript`) — recognised speech appears line by
  line while the interview is still running, timestamped and auto-scrolling.
- **AI-drafted clinical note** (`NoteDraftPanel`) — writes itself as the
  patient talks, redrafted from the running transcript; no "draft this"
  button. Once the clinician edits a section, live updates pause so typed
  text is never overwritten. Reports "unavailable" with no fallback if no
  LLM key is set.
- **Clinical notes** (`ClinicalNotes`) — append-only note history for the
  visit.
- "View last result" button (once scored) reopens the score modal without
  re-running scoring.

## 5. Score modal — overlay within screen 4 (`web/components/ScoreModal.tsx`)

Not a route — an overlay opened from the visit workspace after scoring.
This is the payoff screen: where the AI's read on the visit is presented for
clinician review.

- **Gauges**: PHQ-9 and HAM-D, each showing the **AI-predicted score as the
  primary ring and the clinician's own administered total as a second inner
  ring** — the AI runs alongside manual scoring, not in place of it. Both
  count up and sweep in on open. A caseness pill (Case / Non-case) pops in
  alongside them.
- **Modality attribution bars** — how much each modality (text, audio,
  metadata) contributed to the score, animated width bars.
- **Evidence quotes** — transcript snippets the model's decision is
  attributable to, giving the score a concrete "why."
- **AI-generated narrative** — a typewriter-revealed clinical-language
  summary of the result.
- **Subtype differential** — DSM-5-flavored symptom-subtype likelihoods
  (e.g. melancholic vs. atypical presentation), LLM-backed.
- **Next-step considerations** — guideline-grounded treatment suggestions,
  explicitly advisory.
- **Patient-facing after-visit summary** — a plain-language version of the
  result meant to be shared with the patient, distinct from the clinical
  narrative above.
- **Clinician review** — agree with the model or record an adjusted
  PHQ-9/HAM-D, plus a free-text comment; persisted per visit.
- **Print / Save PDF** — prints only the clinical content
  (`data-print-area`), stripped of app chrome, nav, and interactive controls.
- Every LLM-backed section (narrative, subtype differential, next-step
  suggestions, patient summary, caseload query on screen 1) shows
  "Not available — requires an LLM connection (ANTHROPIC_API_KEY)" instead
  of a fabricated answer when no key is configured — there is deliberately
  no non-LLM fallback for any of them.

## 6. Practice analytics — `/analytics` (`web/app/analytics/page.tsx`)

A population view across the whole caseload, not one patient's chart.

- Five headline stat tiles (animated count-up): total patients, total
  visits, scored visits, referral-flag rate, caseness rate.
- **PHQ-9 severity distribution** — a bar per severity band (minimal / mild
  / moderate / moderately severe / severe) showing how many scored visits
  fall into each.
- **AI caseload summary** — an LLM-generated narrative over the aggregate
  stats, typewriter-revealed; same "unavailable" honesty rule as above when
  no LLM key is set.

## Navigation summary

```
/  (Patient roster)
├── /patients/new  (Register patient) ──┐
├── /patients/[id]  (Patient detail)    │
│   └── /visits/[id]  (Visit workspace) ◄┘  (new patient → first visit)
│       └── Score modal (overlay, opened after "Run scoring")
└── /analytics  (Practice analytics)
```

Terminology note: the frontend speaks **Patient**/**Visit** throughout;
the backend's underlying collections and API paths still say
**participant**/**session** internally (`web/lib/api.ts` bridges the two) —
see [`docs/system-roadmap.md`](system-roadmap.md) for why the rename stopped
at the frontend boundary.
