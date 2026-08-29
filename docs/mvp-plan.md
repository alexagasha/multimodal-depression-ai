# MVP plan: connecting the validated model to the dashboard

**Status:** Day 1 complete · **Estimate:** 4 working days · **Written:** 2026-08-20

The dashboard, API and model pipeline all exist and run end to end. What does
not exist is a *trained model behind them*. This plan closes that gap and
corrects three further mismatches between what the evaluation established and
what the application actually does.

---

## Why this is needed

`api/main.py` resolves `WEIGHTS_PATH` to `outputs/weights/fusion_head.npz` and
falls back to a freshly constructed `FusionHead()` when that file is absent:

```python
FusionHead.load(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else FusionHead()
```

That file has never been created. `FusionHead.__init__` seeds its weights from
`np.random.default_rng(seed)`, so **every severity score the dashboard has
displayed is random output from an untrained network.** This is the first thing
to fix and the reason the plan starts where it does.

The rule-based referral flag is unaffected. It is computed from clinician-entered
item scores by `src/safety/risk_flag.py` and is never gated on model output —
the separation that was designed in for exactly this situation.

### The three further mismatches

| # | Evaluation established | Application currently does |
|---|---|---|
| 1 | Ridge outperformed the MLP on every feature set (0.826 vs 0.695 prosodic; 0.791 vs 0.673 linguistic) | `FusionHead` is a 1552→256→64→2 MLP |
| 2 | Wav2Vec2 failed permutation testing (p = 0.082) and lost to three duration measures | `score_session` calls `run_audio_pipeline` (Wav2Vec2) |
| 3 | Calibrated threshold raises balanced accuracy 0.584 → 0.759 | `score_session` uses the clinical cutoff, giving specificity ≈ 0 |

---

## Day 1 — Fit and export a real model

**Goal:** the dashboard stops producing random numbers.

### Tasks

1. **Add `LinearHead` to `src/fusion/model.py`.** Ridge is an affine map, so the
   head is `y = Wx + b` with `W` of shape `(2, D)` and `b` of shape `(2,)`. It
   must expose the same interface as `FusionHead` — `forward`, `predict_label`,
   `save`, `load`, `caseness_threshold` — so `api/main.py` needs no special
   casing.

2. **Fold the standardiser into the weights.** The evaluated model is a
   `StandardScaler` followed by `Ridge`. Rather than shipping the scaler
   separately, collapse both into one affine transform:

   ```
   W' = W / scale
   b' = b - W · (mean / scale)
   ```

   One matrix and one vector, no preprocessing state to keep in sync at
   inference. Verify equivalence against the sklearn pipeline before trusting it.

3. **Write `scripts/fit_final_model.py`.** Fits ridge (α = 1000) on all 135
   analysed participants using the `text + prosodic + demographic` feature set,
   derives the decision threshold by the Youden criterion, and exports
   `outputs/weights/fusion_head.npz` containing `W`, `b`, `decision_threshold`,
   `feature_dim`, and provenance metadata (cache checksum, date, seed).

### Acceptance criteria — all met 2026-08-20

- [x] `LinearHead.forward` reproduces the sklearn pipeline on all 135 cached vectors to within 1e-4 — actual deviation 7.1e-15 (808) and 5.4e-13 (1552).
- [x] `outputs/weights/fusion_head.npz` exists and loads.
- [x] Loading returns a head whose `caseness_threshold` is the calibrated value, not 7.0.
- [x] Test suite passes (48 passed, 1 skipped).

### Outcome

Two heads were exported rather than one, following the sequencing note at the
foot of this plan:

| File | Features | Dim | Threshold | Purpose |
|---|---|---|---|---|
| `fusion_head.npz` | text + wav2vec2 + metadata | 1552 | 11.39 | matches what `score_session` assembles **today**, so the API serves a real model immediately |
| `fusion_head_808.npz` | text + prosody + metadata | 808 | 10.55 | the evaluated best; becomes the default at Day 3 |

Serving the trained head instead of the untrained one changes in-sample
correlation with observed HAM-D from r = 0.10 to r = 0.83, and predicted
caseness from a constant to 90 of 135. **Those in-sample figures are apparent
performance, not generalisation** — the cross-validated equivalents (r = 0.469,
MAE 5.24 for this feature set) remain the ones to report.

`api/main.py` now loads via `load_head` and **refuses weights whose dimension
does not match `FUSION_INPUT_DIM`**, falling back to an untrained head with a
loud warning rather than multiplying a vector assembled from different features.
Without that check, the Day 3 migration could have silently served plausible
nonsense.

### Decision: weights are committed

Resolved 2026-08-20 — the weights are committed so a fresh clone has a working
model rather than silently falling back to an untrained head.

`.gitignore` unignores `outputs/weights/*.npz` stepwise, because git cannot
re-include a path inside an ignored directory. Everything else under `outputs/`
stays ignored, including `outputs/cache/fusion_vectors.npz`, which does contain
per-participant feature vectors and must not be committed.

The weight files hold only `W`, `b`, `decision_threshold` and provenance
metadata — 24 KB in total, no participant data. Note that ridge coefficients
fitted where features outnumber participants are a linear function of the
training data; that is not a practical disclosure risk with frozen-encoder
embeddings and heavy regularisation, but it is the reason the feature cache
itself stays out of the repository.

### Notes

The threshold is fitted on all 135 participants here, whereas the reported
performance came from thresholds fitted per training fold. That is correct — the
reported figure estimates generalisation, the shipped threshold should use all
available data — but the two numbers are not the same thing and the distinction
belongs in the model card.

---

## Day 2 — Prosody as a pipeline module

**Goal:** the application can compute the features the model was actually trained on.

Prosody currently exists only as a notebook cell. It must become a module that
runs inside the request path.

### Tasks

1. **Create `src/pipelines/prosody_pipeline.py`** exposing
   `run_prosody_pipeline(pid, data_root) -> np.ndarray` of shape `(24,)`, with
   `EMBED_DIM = 24` and the feature names as a module constant so the XAI layer
   can label them.

2. **Match the training-time computation exactly.** In training, each measure was
   computed *per interview response* and then summarised across responses as a
   mean and a standard deviation. At inference the application holds one
   concatenated waveform plus a transcript with `start_time`/`stop_time` per
   turn. Prosody must therefore be computed **per transcript turn**, slicing the
   waveform by those timings, and aggregated identically. Computing it over the
   whole waveform in one pass would silently produce different features from
   those the model was fitted on.

3. **Update `src/fusion/aggregate.py`.** `FUSION_INPUT_DIM` becomes
   `768 (text) + 24 (prosody) + 16 (metadata) = 808`. Keep the Wav2Vec2 path
   available behind a flag for the thesis comparison, but it is no longer the
   default.

### Acceptance criteria

- [ ] For at least 5 participants, prosody computed by the module from the session WAV matches the training-time values (from `prosody.csv`) within 5% on every feature.
- [ ] `run_prosody_pipeline` completes in under 2 seconds for a 10-minute interview.
- [ ] A turn shorter than the minimum duration is skipped rather than producing NaN.

### Risk

**This is the highest-risk step in the plan.** Training features came from the
original per-prompt `.m4a` files decoded individually; inference features come
from slices of a re-encoded concatenated WAV. Sample-rate conversion, the
inserted inter-response silence, and boundary rounding can all shift the
measures. The 5-participant parity check is not optional — if it fails, the
model is being fed different features from those it was fitted on, and the
predictions are invalid in a way nothing downstream would reveal.

---

## Day 3 — Rewire the scoring path

**Goal:** the API serves calibrated predictions from the correct features.

### Tasks

1. **`api/main.py`, `score_session`.** Replace `run_audio_pipeline` with
   `run_prosody_pipeline`. Replace the hard-coded caseness comparison:

   ```python
   binary_pred = int(scores["hamd"] >= HAMD_CASENESS_THRESHOLD)   # before
   binary_pred = int(scores["hamd"] >= _fusion_head.caseness_threshold)   # after
   ```

2. **Update `src/xai/attribution.py`.** Attribution for a linear model is
   `coefficient × feature value`, which is exact rather than approximate. With
   named prosodic features this becomes clinically legible — "reduced pitch
   variability" and "longer pauses" instead of an unnamed dimension index.
   Modality-level attributions should still sum over the text, prosody and
   metadata blocks so the existing dashboard components keep working.

3. **Surface the operating point.** The API response should carry the threshold
   used and the model version, so a prediction can be traced to the weights that
   produced it.

### Acceptance criteria

- [ ] `POST /sessions/{id}/score` returns non-random, reproducible scores for the same input.
- [ ] Two sessions with clearly different severity produce appropriately ordered predictions.
- [ ] `binary_pred` is not 1 for every participant (the current behaviour).
- [ ] `api/tests` pass.

---

## Day 4 — End-to-end verification and honest interface copy

**Goal:** it works with a real recording, and it does not overclaim.

### Tasks

1. **Full run through the interface**: create a participant, record or upload
   audio, submit scale responses, score, read the note draft. Confirm the
   referral flag fires independently of the model.

2. **Write the interface copy.** The model reaches PPV 0.933 and **NPV 0.459** at
   the selected operating point. A clinician reading "not a case" and acting on
   it would be wrong more often than not. The interface must present the output
   as **prioritisation, never rule-out**, and must not imply diagnosis. Suggested
   framing, to be reviewed clinically:

   > Estimated severity is a decision aid based on 135 interviews at two
   > Ugandan sites. It supports prioritisation and does not exclude depression.
   > Clinical judgement and the recorded scale scores take precedence.

3. **Write `docs/model-card.md`**: training population, feature set, performance
   with confidence intervals, permutation results, operating point, known
   limitations (no external validation; NPV; cohort demographics; the collection
   period with degraded audio).

### Acceptance criteria

- [ ] One complete visit scored through the UI from recording to note draft.
- [ ] Referral flag verified to fire with the model absent.
- [ ] Model card committed.
- [ ] No screen presents the output as diagnostic.

---

## Out of scope

Deliberately excluded, and each would need to be stated as a limitation if the
tool were shown to anyone clinically:

- External validation on an independent cohort.
- Retraining on the ~40 further participants (the adapter supports adding them without disturbing splits; the model would need refitting and the threshold re-derived).
- Any regulatory or clinical-governance process.
- Live prosody during recording — scoring remains post-hoc.
- Replacing the Wav2Vec2 path entirely; it stays behind a flag for the thesis comparison.

---

## Sequencing note

Days 1 and 2 are independent and could run in parallel. Day 3 depends on both.
Day 4 depends on Day 3.

If time is short, **Day 1 alone is worth doing**: it replaces random output with
a real model, even if that model still runs on the weaker Wav2Vec2 features. Days
2 and 3 are what make the served model the one the thesis reports.

This plan competes with thesis writing for the same days. The thesis has a fixed
deadline; the MVP does not.
