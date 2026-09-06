# Psychiatric documentation: standard, gap analysis, and implementation plan

Status: **implemented.** Phases 1-5 shipped; see the commits from
"Keep the structure the note already had" onward. Part 3 is left in its
original planning form as the record of what was intended and why, with
per-phase notes on what actually landed.

The dashboard already drafts a SOAP note and saves it. This document asks a
narrower question: **would that note be acceptable as psychiatric
documentation?** It sets out what the standard actually requires, measures the
current code against it, and proposes a phased implementation.

---

## Part 1 — What the standard requires

### 1.1 The reference: APA psychiatric evaluation

The American Psychiatric Association's practice guideline for the psychiatric
evaluation of adults defines the components of an initial evaluation: reason
for evaluation, history of present illness, past psychiatric history,
substance use history, medical history, family history, developmental and
social history, review of systems, **mental status examination**, **risk
assessment**, functional assessment, **diagnostic formulation**, and treatment
planning.

Three of those are emphasised because they are where this project has nothing.

### 1.2 Note format: SOAP is the correct choice, and we already use it

Behavioural health uses three formats. SOAP (Subjective / Objective /
Assessment / Plan) separates patient report from clinician observation, which
is why it dominates medical and psychiatric settings where other providers
read the note. DAP (Data / Assessment / Plan) is faster and suits outpatient
counselling. BIRP (Behavior / Intervention / Response / Plan) is common in
community mental health because it evidences medical necessity.

`api/note_draft.py` and `api/live_note.py` both use SOAP. **No change needed —
this was the right call.** The problem is not the format; it is that the
sections are unstructured prose and one of them (Objective) has no clinical
spine.

### 1.3 The Mental Status Examination

The MSE is a structured observation of the patient's state *at the time of
interview*, across ten domains:

| Domain | Obtainable from participant audio? |
|---|---|
| Appearance | **No** — requires visual observation |
| Behaviour / motor activity | **No** (partially inferable from speech) |
| Attitude toward examiner | **No** |
| Speech (rate, volume, prosody, latency) | **Yes** — we measure this directly |
| Mood (patient's own words) | **Yes** — quotable from transcript |
| Affect | Partial — voice quality only, not facial |
| Thought process (coherence, tangentiality) | **Yes** — from transcript |
| Thought content | **Yes** — from transcript |
| Perception (hallucinations, if reported) | Partial — only if volunteered |
| Insight and judgement | Partial |
| Cognition | Partial — no formal testing in our instrument |

This table is the honest boundary of what this system can contribute, and it
should be visible in the product, not buried in a thesis.

Note the one genuinely favourable row: **speech**. The 24 prosodic features in
`src/pipelines/prosody_pipeline.py` already compute speaking rate, pause ratio
and pitch statistics. Those map directly onto the MSE speech domain. This is
the single place where the model can populate a standard clinical field with
measured evidence rather than an LLM's impression of a transcript.

### 1.4 Risk documentation is its own standard

Documenting suicidality is not a checkbox. When ideation is positive, the note
must record: passive versus active ideation; intent; presence or absence of a
plan; access to means; prior attempts; **protective factors** (reasons for
living, dependants, faith, social support); the safety plan; and — the element
that carries the most medicolegal weight — **the clinician's reasoning for the
level-of-care decision**, naming the specific factors behind it.

Two further rules matter for us:

- **Document the patient's exact words.** A verbatim quote is more defensible
  than a paraphrase. We hold a timestamped transcript, so this is unusually
  easy here — `api/evidence.py` already extracts quotes.
- **Re-document every time**, even when unchanged from the last visit.

### 1.5 AI-drafted notes carry additional obligations

Current guidance on ambient AI scribes converges on the same points:

- The clinician is the final authority. AI output is a **draft** requiring
  review and **attestation** before it enters the record.
- **Time-to-sign is audited.** A note signed seconds after the encounter closes
  is used as evidence that no meaningful review occurred.
- **Provenance must be distinguishable** — what the model wrote versus what the
  clinician changed.
- **Retention must separate transient from permanent.** Raw audio and interim
  transcripts are processing artefacts and may be deleted after verification;
  the signed note and its provenance metadata are the legal record.

Our prompts already carry the "draft only, psychiatrist remains responsible"
framing. That is the right instinct, but it is an instruction to the model, not
a property of the system — nothing in the code enforces it.

### 1.6 The Uganda context

The system is for Butabika and Lira, not a US practice, so the applicable
frame is not CMS or HIPAA:

- **HMIS 105** is the monthly facility summation form; it was revised after
  January 2020 to cover more mental, neurological and substance-use
  conditions. A note that cannot roll up into HMIS categories creates parallel
  paperwork rather than replacing any.
- **Uganda Clinical Guidelines** and **WHO mhGAP-IG** are what PHC staff are
  trained against. `api/treatment_suggestions.py` should cite mhGAP-IG's
  depression module specifically rather than generic guidance.
- The privacy obligation runs through the **study IRB** and Uganda's data
  protection law, not a Business Associate Agreement. This needs confirming
  against the approved protocol before any retention change ships.

---

## Part 2 — Gap analysis against the current code

| Requirement | Current state | Gap |
|---|---|---|
| SOAP format | `note_draft.py`, `live_note.py` | **None** |
| Structured note storage | `ClinicalNoteIn = {author, note_text}` | SOAP is flattened to a single string at `NoteDraftPanel.tsx:48`. Structure is generated, then discarded at the API boundary |
| Append-only history | `POST /sessions/{id}/notes` appends | **None** — correct, and documented as such |
| Mental status examination | absent | Objective is free prose with no clinical spine |
| Risk assessment record | `flag_risk()` returns a boolean; `RiskBanner` displays it | A screening trigger, not a documented assessment. **Nothing records what the clinician did in response.** Highest exposure in the system |
| Verbatim patient quotes in risk documentation | `evidence.py` extracts quotes, but not for this purpose | Unused capability, cheap to wire up |
| Clinician attestation | `author` is an unvalidated free-text box | No signature, no `signed_at`, no role, no identity check |
| AI provenance | absent | A saved note is indistinguishable from a clinician-written one |
| Time-to-review | absent | Not captured, so cannot be defended or audited |
| Diagnostic formulation | absent | Deliberate — see Part 4 |
| Retention policy | none; WAVs and transcripts persist in `data/live/` | Undefined for real clinical audio |
| HMIS 105 alignment | none | Output does not feed national reporting |

The through-line: **the system generates more clinical structure than it
keeps.** The LLM returns four typed SOAP fields and the frontend concatenates
them into a string. Fixing that is the foundation everything else sits on.

---

## Part 3 — Implementation plan

Ordered by clinical risk, not by ease. Phase 2 is the one that matters most;
Phase 1 comes first only because it is a prerequisite for the note work.

### Phase 1 — Make a note a record instead of a string (~0.5 day)

Extend `ClinicalNoteIn` in `api/main.py`:

```python
class ClinicalNoteIn(BaseModel):
    note_type: str            # initial_evaluation | progress | risk_assessment | addendum
    subjective: str
    objective: str
    assessment: str
    plan: str
    author: str
    author_role: str          # psychiatrist | psychiatric_clinical_officer | nurse
    source: str               # clinician | ai_draft_edited | ai_draft_accepted
    amends: str | None = None # note_id this addendum corrects
```

Keep accepting `note_text` for free-text addenda so nothing existing breaks.
`amends` preserves the append-only rule while making corrections traceable
rather than orphaned.

Then stop flattening at `web/components/NoteDraftPanel.tsx:48` — POST the four
fields — and update the `ClinicalNote` type and `addNote` in `web/lib/api.ts`,
and the rendering in `ClinicalNotes.tsx`.

### Phase 2 — Structured risk assessment (~1.5 days) — **highest priority**

New `api/risk_assessment.py` and a `RiskAssessmentIn` model covering the
required elements from §1.4: ideation type, intent, plan, means access, prior
attempts, protective factors, safety plan, disposition, and free-text
`clinical_reasoning`.

Endpoints `POST` / `GET /sessions/{id}/risk-assessment`.

Three design rules:

1. **This form is never AI-completed.** The model may surface candidate
   verbatim quotes from the transcript for one-click insertion; every
   judgement field stays human. This mirrors the boundary
   `src/safety/risk_flag.py` already draws and states in its docstring.
2. **When `risk_flag` is true, the visit cannot be marked complete without
   it.** Add it as a blocking step in `SessionStatusStepper.tsx`. Today the
   banner fires and the clinician can close the visit having recorded nothing.
3. **Quote insertion reuses `api/evidence.py`**, filtered to risk-relevant
   turns, so the note carries the patient's own words.

### Phase 3 — MSE block in the Objective section (~1 day)

New `api/mse.py` producing the ten domains, plus `web/components/MSEPanel.tsx`.

The governing rule is the table in §1.3: **prefilled where there is evidence,
blank where there is not.** Appearance, behaviour and attitude return
`"not observable — audio only"` and are the clinician's to complete. Never
fabricate a domain the recording cannot support. This is the same
honest-fallback convention the GenAI modules already follow by returning
`None` rather than a plausible guess.

The speech domain is populated from the existing prosodic features rather than
from the LLM — measured values, with the feature names already documented in
`FEATURE_NAMES`. That gives the MSE a row grounded in measurement instead of
inference, and gives the acoustic model a clinical purpose beyond a score.

### Phase 4 — Attestation and provenance (~0.5 day)

- Add `signed_at` and `signed_by`, distinct from `created_at` / `author`.
- Record when the draft was shown and compute review duration. Do not block on
  it — surface it. "Draft reviewed in 3 s" is exactly the pattern auditors
  challenge, and making it visible is what discourages it.
- Diff the submitted fields against the draft that was displayed to derive
  per-field provenance client-side; store the result with the note.
- Stamp the model identifier and prompt version (`api/llm_utils.py` already
  knows the model).
- Render a badge in `ClinicalNotes.tsx`: *Clinician-written* /
  *AI-drafted, clinician-edited* / *AI-drafted, accepted unchanged*.

### Phase 5 — Retention and Uganda reporting (~1 day)

- Write the retention policy down first, checked against the approved study
  protocol, then implement: raw WAV and interim transcript are transient and
  removable once the note is signed (behind `DEP_RETAIN_AUDIO`, defaulting to
  retain so nothing is lost by surprise); signed notes and provenance persist.
- Add an HMIS 105 roll-up export alongside `api/trends.py` /
  `api/analytics_summary.py`, so the facility gets a form it already submits
  rather than a second system to maintain.
- Point `api/treatment_suggestions.py` at mhGAP-IG's depression module.

**Total: roughly 4.5 days.** Phases 1 and 2 alone (2 days) close the gap that
actually carries risk.

### What actually landed

All five phases, with three deviations worth recording:

- **Phases 1 and 4 shipped together.** Both reshape `ClinicalNoteIn`, and
  splitting them would have meant changing the same model twice.
- **The MSE reports measurements without interpreting them.** The plan
  imagined the speech domain being described ("slowed", "reduced prosody").
  There is no reference distribution for this cohort, so asserting a normal
  range would have been fabrication. The measured values are shown and the
  interpretation is left to the clinician.
- **Retention defaults to keeping everything.** The plan proposed deleting raw
  audio once a note is signed. Implemented, but off by default and documented
  in `docs/data-retention.md` as a protocol decision — for the research cohort
  the recordings are the dataset.

`POST /sessions/{id}/close` was added as the gate the risk requirement needed:
there had been no notion of a visit being finished, so "cannot be completed
without a risk assessment" had nothing to attach to.

---

## Part 4 — Deliberately out of scope

- **Automated DSM-5-TR / ICD-11 diagnosis.** The model estimates symptom
  severity from a screening instrument; it does not diagnose. Diagnostic
  formulation stays entirely human, and the product should not imply otherwise.
- **Billing and E&M coding.** Irrelevant in this setting.
- **Becoming an EHR.** This is a screening adjunct. Notes should be
  exportable, not authoritative.
- **Free-text history sections** (past psychiatric, family, developmental
  history). The KoBo instrument does not collect them and the recording does
  not contain them; leaving them blank is more honest than inviting the model
  to infer them.

---

## Part 5 — Thesis linkage

Parts 1 and 2 provide a Discussion section that most comparable dissertations
lack: not "the model achieved AUC 0.826", but *what would have to be true for
this output to be enterable in a clinical record*. The §1.3 table is the
clearest statement of the system's boundary anywhere in the project, and
belongs in the limitations section and in `docs/model-card.md`.

---

## Sources

- [APA-aligned initial psychiatric evaluation components](https://www.icanotes.com/2025/06/20/essential-components-of-a-comprehensive-initial-psychiatric-evaluation-template/)
- [Initial Psychiatric Assessment — Merck Manual Professional](https://www.merckmanuals.com/professional/psychiatric-disorders/approach-to-the-patient-with-psychiatric-symptoms/initial-psychiatric-assessment)
- [Mental Status Examination — StatPearls, NCBI](https://www.ncbi.nlm.nih.gov/books/NBK546682/)
- [Comparing SOAP, BIRP and DAP progress notes](https://notedesigner.com/resources-comparing-types-of-progress-notes/)
- [Types of therapy notes: DAP vs SOAP vs BIRP — Ensora Health](https://ensorahealth.com/blog/understanding-the-differences-between-dap-soap-and-birp-notes/)
- [How to document suicidal ideation — Mentalyc](https://www.mentalyc.com/blog/how-to-document-suicidal-ideation)
- [Suicide: Assessment and Management — StatPearls, NCBI](https://www.ncbi.nlm.nih.gov/books/NBK617057/)
- [SAMHSA suicide assessment (SAFE-T)](https://library.samhsa.gov/sites/default/files/safet-flyer-pep24-01-036.pdf)
- [Barriers and opportunities of scaling ambient AI scribes — npj Digital Medicine](https://www.nature.com/articles/s41746-026-02554-0)
- [Ambient AI documentation compliance guidance](https://medcurity.com/ambient-ai-documentation-hipaa/)
- [Why ambient AI scribe notes fail audit](https://kevinmd.com/2026/08/why-ambient-ai-scribe-notes-fail-the-2026-cms-audit.html)
- [Adherence to clinical guidelines in mental health integration, Mbarara, Uganda — Int J Ment Health Syst](https://ijmhs.biomedcentral.com/articles/10.1186/s13033-021-00488-6)
- [Governance issues in integrating mental health into primary care in Uganda — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4806490/)
- [Diagnostic pattern of mental, neurological and substance use disorders at Ugandan PHC facilities](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11247730/)
