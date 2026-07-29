# Migration plan — aligning the pipeline to the real data-collection tool

**Source instrument:** `QUESTIONAIRE-V2_R_concise_May 26.docx`
**Key decision (agreed):** predict **PHQ-9 (/27) and HAM-D (multi-task, two outputs)**.
**Languages:** **DROPPED (2026-07-27) — study is English-only.** Multilingual support
(§1b below) was built speculatively and has been removed; see §1c.

**Status (2026-07-29): score model is code-complete — only real data + training
remain.** The last open code gap (safety/referral flag, §4) is now implemented.
Everything else in this doc's checklist was already done in code. The rest of the
system (backend API, live audio web app, GenUI/RAG, Firebase data platform/deploy) is
tracked separately — see the plan/roadmap, not this migration doc.

**Status (2026-07-27): English-only, plain BERT + Wav2Vec2 (2026-07-27).**
Data collection is starting soon and IRB/LUREC is cleared. The team confirmed the
study is **English-only** (no Luganda/Luo), so multilingual handling was ripped out:
`language` removed from metadata (16-dim now, was 19), `FUSION_INPUT_DIM` 1555→1552,
text encoder is plain `bert-base-uncased` (was `xlm-roberta-base`), audio encoder
stays `wav2vec2-base-960h` (was already English, multilingual-swap note removed).
Both models are **pre-downloaded and run on Google Colab**. 15/16 tests pass (1
skipped, torch-only, Colab).

**Status (2026-07-07): Phase-1 scaffold IMPLEMENTED and green (11/11 tests pass).**
Done: multi-task labels (§1), categorical metadata (§2), transcript speaker renamed
`Interviewer`, **HAM-D-derived caseness label + accuracy & noise-robustness eval
gates (§4b)**. Verified via regenerate → run_pipeline → metrics → robustness → pytest.

**Safety gap closed (2026-07-29):** the referral-protocol flag for PHQ-9 item 9 /
HAM-D suicide domain is now implemented — see §4 below and `src/safety/risk_flag.py`.
This was the last remaining *code* gap; the score model now needs only real data +
training, not further implementation.

**Still open (design/confirmation, not blocking the stub):**
- ⚠️ HAM-D denominator — implemented as **/44** (11 items × 4); form misprints /52. Confirm.
- Site excluded from features (recommended) — confirm.
- ~~Video capture/consent~~ — **video removed** (2026-07-07): not in the questionnaire; pipeline is text + audio + metadata.
- ~~Multilingual~~ — **dropped (2026-07-27)**: study confirmed English-only.

---

## 0. The shift in one paragraph

The repo was scaffolded to the **DAIC-WOZ / E-DAIC-WOZ** shape (single `phq8_score`
0–24, "Ellie" interviewer, 3-field metadata). The real study is a **bespoke,
interviewer-administered Uganda instrument** (sites: Butabika / Mulago) that
collects **two** severity scores (PHQ-9 self-report, HAM-D interviewer-rated),
a **9-field categorical metadata block**, and ~35 open-ended prompts explicitly
earmarked for text + audio modelling. Three things change everywhere: the
**label schema**, the **metadata schema**, and the **data source assumptions**.

---

## 1. Label schema → multi-task PHQ-9 + HAM-D

Currently `phq8_score` (0–24) is threaded through 6 files. Replace with two targets.

- [x] **`data/synthetic/generate_synthetic_data.py`**
  - `LABELS.csv` columns: drop `phq8_score`; add `phq9_score` (0–27) and
    `hamd_score` (0–44 — see ⚠️ below). Keep `participant_id`, `split`.
  - `generate()` label line (currently `np.clip(np.random.normal(8,5),0,24)`):
    emit both scores; give them positive correlation so multi-task eval is meaningful.
- [x] **`src/fusion/model.py`**
  - `OUTPUT_DIM = 1` → `2`; `W3` becomes shape `(2, HIDDEN2)`.
  - Replace `PHQ8_MIN, PHQ8_MAX = 0, 24` with per-head ranges:
    `PHQ9_RANGE = (0, 27)`, `HAMD_RANGE = (0, 44)`. Clamp each output independently.
  - `forward()` returns two values (dict or tuple), not a scalar.
  - `predict_label()` binary caseness from **PHQ-9 ≥ 10** (standard cutoff — keep
    `DEPRESSION_THRESHOLD = 10`). *(Superseded — see §4b: caseness is now HAM-D-derived.)*
  - Mirror the change in the commented `FusionHeadTorch` (final `nn.Linear(HIDDEN2, 2)`).
- [x] **`src/fusion/run_pipeline.py`**
  - `phq8_pred` → `phq9_pred`, `hamd_pred`; `phq8_true` → `phq9_true`, `hamd_true`.
  - `binary_pred` / `binary_true` derive from PHQ-9. *(Superseded — HAM-D-derived, §4b.)*
  - `run_all()` reads `phq9_score` + `hamd_score` from `LABELS.csv`.
- [x] **`src/eval/metrics.py`**
  - Report RMSE + MAE **per target** (`phq9`, `hamd`) — 4 regression numbers.
  - Keep F1 (weighted/macro) + confusion matrix on the PHQ-9 binary.
  - Update the `df must contain columns:` docstring contract.
- [x] **`src/xai/attribution.py`**
  - Permutation ablation measures the drop in **one** output; pick the target to
    attribute (recommend PHQ-9 primary) or return attributions for both.
  - `baseline_phq8_pred` key → `baseline_phq9_pred` (+ hamd if attributing both).
- [x] **`tests/test_end_to_end.py`**
  - Update column names, the sample eval frame (`phq8_true`/`phq8_pred`), and the
    model-output assertion (now 2 outputs, not a scalar). `"phq8_pred" in result`
    → `"phq9_pred"`.

> ⚠️ **HAM-D max is inconsistent in the form.** It lists **11 domains × 0–4 = 44**,
> but prints `Total Score: ______ / 52` (the standard HAM-D-17 max). Confirm the
> intended denominator before fixing `HAMD_RANGE`. 44 matches the 11-item design.

---

## 1b. Languages — DROPPED, historical record only

~~The study collects data in ~3 languages (assumed English / Luganda / Luo)~~ —
**superseded 2026-07-27: the study is English-only.** This section originally
built multilingual handling speculatively, ahead of confirmation. It has been
fully removed from the code (see §1c) and is kept here only as a record of what
was undone, in case multilingual support is revisited later.

Previously implemented (now removed): `language` metadata field + one-hot
encoding (3 dims), multilingual text encoder (XLM-RoBERTa), multilingual audio
encoder SWAP note (wav2vec2-large-xlsr-53), per-language robustness framing.

## 1c. English-only pivot (2026-07-27)

- [x] **`src/pipelines/text_pipeline.py`**: `DEFAULT_TEXT_MODEL` →
      `bert-base-uncased` (plain BERT, 768-d). Multilingual docstring removed.
- [x] **`src/pipelines/audio_pipeline.py`**: kept `wav2vec2-base-960h` (already
      English); removed the multilingual-XLSR SWAP note.
- [x] Both models are **pre-downloaded** and run on **Google Colab**.
- [x] **`src/pipelines/metadata_pipeline.py`**: removed `language` field and
      `LANGUAGE_CATS` one-hot (3 dims). `EMBED_DIM` 19 → **16**.
- [x] **`data/synthetic/generate_synthetic_data.py`**: removed `LANGUAGES`
      constant and the `language` key from `make_metadata()`.
- [x] **`src/fusion/aggregate.py`**: `METADATA_DIM` 19 → 16;
      `FUSION_INPUT_DIM` 1555 → **1552**.
- [x] **`data/edaic/adapt_edaic.py`**: dropped the `language: "english"` metadata
      key it used to stamp on adapted E-DAIC sessions (no longer a schema field).
- [x] 15/16 tests pass (1 skipped, torch-only) after regenerating synthetic data.

---

## 2. Metadata schema → 9 categorical fields (Section A)

Current encoder handles only `age:int`, `gender`, `education_years:int` → 3 dims.
The tool collects categoricals; age and education are **bands/levels**, not integers.

| Field (Section A) | Values | Suggested encoding | Dims |
|---|---|---|---|
| Age | 18–25 / 26–35 / 36–45 / 46–55 / 56–65 / Other | ordinal (band index) | 1 |
| Sex | Male / Female | binary | 1 |
| Marital status | Married / Divorced / Widowed / Single | one-hot (nominal) | 4 |
| Ethnicity/tribe | Bantu / Luo / Other | one-hot (nominal) | 3 |
| Residence | Urban / Semi-urban / Rural | ordinal | 1 |
| Education level | None / Primary / Secondary / Tertiary-University | ordinal | 1 |
| Employment | Employed / Unemployed / Self-employed / Student | one-hot (nominal) | 4 |
| Smartphone | Yes / No | binary | 1 |
| **Site** | Butabika / Mulago / Other | **exclude from features** (provenance) | 0 |

- [x] **`data/synthetic/generate_synthetic_data.py` → `make_metadata()`**: emit the
      fields above with categorical values matching the form's wording exactly.
- [x] **`src/pipelines/metadata_pipeline.py` → `encode_metadata()`**: replace the
      3-field encoder with the categorical scheme; recompute `EMBED_DIM`
      (≈ **16** with the table above — exact value depends on ordinal-vs-one-hot choices).
      Update the field-list docstring.
- [x] **`src/fusion/aggregate.py`**: `METADATA_DIM` must equal the new `EMBED_DIM`;
      `FUSION_INPUT_DIM` recomputes (e.g. 768 + 768 + 16 = **1552**). The XAI
      metadata slice keys off `METADATA_DIM`, so it follows automatically.
- [ ] Decisions to confirm:
  - **Site** as a feature? Recommend **no** (recruitment-site leakage/confound; a
    model can shortcut on "which hospital" instead of symptoms). Keep it in the
    metadata store for stratified analysis, out of the fusion vector.
  - "Tertiary / University" — one level or two? (form wording is ambiguous).
  - Age ordinal vs one-hot given the "Other" catch-all breaks strict ordering.

---

## 3. Data-source assumptions (DAIC-WOZ → own interviews)

- [ ] **Transcripts / `src/pipelines/sync.py`** — *mostly ready.*
      `INTERVIEWER_SPEAKER_NAMES = {"Ellie", "Interviewer"}` already strips a human
      interviewer, so real transcripts just need the interviewer turns labelled with
      one of those speaker strings (or extend the set). No 30s-segmenting change.
- [ ] **`generate_synthetic_data.py` transcript** (optional realism): rename the
      `"Ellie"` speaker to `"Interviewer"`; draw filler lines from Section D themes
      (mood, sleep, energy, coping, cultural idioms) instead of generic DAIC lines.
- [ ] **Participant IDs**: DAIC-style `300+i` numeric IDs → real study uses
      "Participant ID" + Site. Decide the ID scheme and how Site attaches.
- [ ] **README `data/real/` step**: still valid — drop real sessions into the same
      folder shape. Revisit the "E-DAIC-WOZ train+test vs test-only" open decision:
      this tool implies **own-collected primary data**; E-DAIC-WOZ becomes optional
      external/transfer data or is dropped.

---

## 4. Safety, ethics & governance (not optional)

- [x] **Suicide/self-harm items carry a referral protocol — implemented (2026-07-29).**
      PHQ-9 item 9, HAM-D "Suicide domain", and D2 Q9 all have "immediate risk
      assessment and referral" in the form. `src/safety/risk_flag.py` (`flag_risk()`)
      is a rule-based check — never dependent on the fusion head, XAI, or any
      generated narrative — that flags either item scoring above 0. Wired into
      `run_pipeline.run_participant()`, which surfaces `risk_flag` distinct from
      `binary_pred`. `LABELS.csv` now tracks `phq9_item9_score` /
      `hamd_suicide_item_score` per participant so the item isn't just baked into
      the aggregate total. Still open: the actual **alert routing** (who gets
      notified, how) is a product/ops decision for the web app + data platform,
      not the scoring pipeline — see the system roadmap.
- [ ] **PII / de-identification.** Participant ID, Site, Interviewer name, and tribe
      are sensitive. Separate identifiers from the ML feature store
      (Supabase) — the fusion vector must not carry Site/Interviewer (see §2).
- ~~**Multilingual text.**~~ **Dropped (2026-07-27)** — study confirmed English-only;
      see §1c.
- [x] **Video modality removed** (2026-07-07). The form never mentions video capture or
      consent, so video was dropped entirely: `video_pipeline.py` deleted, no video frames
      in synthetic data, no video branch in fusion/XAI. Pipeline is text + audio + metadata.
- [ ] **Small-N clinical sample.** A bespoke study likely has far fewer participants
      than DAIC. The frozen-backbone + small-head design is well suited to this;
      reconsider fixed train/dev/test vs **k-fold cross-validation** for eval, and
      watch class balance on the PHQ-9 binary.

---

## 4b. Acceptance criteria (eval gates) — implemented

Two project objectives are now instrumented as gates in the eval harness (build
now, certify later on a locked test set / real data):

- **Accuracy ≥ 0.85 on HAM-D caseness.** Caseness (the primary classification
  label) is **HAM-D-derived** — the interviewer-rated score is the clinical gold
  standard. Default cutoff **HAM-D ≥ 7** (standard HAM-D-17 ≥8/52 scaled to the
  11-item /44 form: 8×44/52≈7; `model.HAMD_CASENESS_THRESHOLD`). `metrics.py`
  reports `accuracy` + `accuracy_target_met` (`ACCURACY_TARGET = 0.85`). This
  supersedes the original §1 plan's PHQ-9≥10 caseness.
- **≤ 5% performance decline under noise.** `src/eval/robustness.py` sweeps audio
  SNR (dB) and text character-corruption rates, compares clean vs noisy accuracy,
  and reports worst-case relative decline vs the `MAX_ALLOWED_DECLINE = 0.05` gate.

Caveat (documented in the module): with mock encoders + untrained head the numbers
are not yet meaningful (stub predicts a near-constant score → trivial accuracy and
~0% decline). The harnesses are validated now; certify real numbers post-training,
with confidence intervals / k-fold given the small bespoke sample.

---

## 4c. E-DAIC smoke-test path (real-data plumbing check)

Tooling to run the pipeline on a real E-DAIC-WOZ session — a machinery/encoder
check, **not** an objectives eval (E-DAIC is PHQ-8, English, no HAM-D):
- `data/edaic/adapt_edaic.py` — converts an E-DAIC session (WAV + transcript;
  **ignores the ~700 MB `features/` folder**) into the pipeline schema.
- `audio_pipeline.load_audio` now reads `.wav` as well as `.npy`.
- `BertTextEncoder` / `Wav2Vec2AudioEncoder` load real HF models when
  torch+transformers are present (Colab GPU) and **fall back to mocks** otherwise —
  so the local suite stays green. Defaults `bert-base-uncased` (768-d) and
  `wav2vec2-base-960h` (768-d), both English, **pre-downloaded for Colab**;
  override via `DEP_TEXT_MODEL` / `DEP_AUDIO_MODEL`.
- `notebooks/edaic_colab.ipynb` — Colab bootstrap (mount Drive → install → adapt → run).

Verified locally on the real `304_P` session (mock encoders): adapter → 27 segments →
prediction. Note the E-DAIC `304_AUDIO.wav` has a bogus header data-size field (scipy
warns but reads the ~13 min correctly; libsndfile/soundfile on Colab handles it too).
E-DAIC is English, so this smoke test now matches the study's confirmed
English-only language setting.

---

## 4d. Training the head — `src/fusion/train.py`

Only the fusion head trains (BERT/Wav2Vec2 stay frozen). Two phases: **ENCODE**
every labeled participant once → cached fusion vectors (`outputs/cache/`); **TRAIN**
a PyTorch MLP (identical architecture to the numpy head) on those vectors, then
**EXPORT** the weights to `outputs/weights/fusion_head.npz` via `FusionHead.save/load`
— so inference stays numpy-only (no torch).
- Label = **one PHQ-9 + one HAM-D per participant** in `LABELS.csv` — no audio/text
  annotation. Loss is normalized per target so PHQ-9 and HAM-D weigh equally.
- Run (Colab, repo root): `python src/fusion/train.py --data_root data/edaic/sessions --epochs 300`
- Inference: `model = FusionHead.load('outputs/weights/fusion_head.npz')`.
- Prints dev/test metrics (accuracy on HAM-D caseness + per-target RMSE/MAE).
- Needs several labeled sessions — a single session (e.g. E-DAIC 304) can't train.
- Weights + cache live under gitignored `outputs/`.

---

## 5. Suggested order of work

1. Labels first (§1) — smallest, unblocks eval on both targets.
2. Metadata schema (§2) — self-contained; changes one dim constant that ripples cleanly.
3. Source realism + IDs (§3).
4. Governance decisions (§4) — mostly design/consent, but Site-exclusion (§2) and
   item-9 handling (§4) touch code.

Nothing here changes the locked architecture (frozen backbones, mean-pool,
concatenate, single fusion head) — only the **head width** (1→2 outputs), the
**metadata block width**, and the **label columns**.
