# Model card: depression severity estimation

**Version** 2026-08-29 · **Status** research prototype · **Not for clinical use**

Weights: `outputs/weights/fusion_head.npz`
Comparison model: `outputs/weights/fusion_head_1552.npz`

---

## What it does

Estimates two depression severity scores from a recorded clinical interview:
PHQ-9 (0–27) and HAM-D (0–44). A screening priority is derived from the
predicted HAM-D by comparing it against a calibrated threshold.

It is a **decision aid for prioritisation**. It does not diagnose, and it does
not rule depression out. See *Negative predictive value* below, which is the
number that most constrains how this may be used.

## Intended use

**In scope.** Ordering a clinic list by likely severity; surfacing a second
opinion alongside a clinician's own scoring; research on multimodal severity
estimation in this population.

**Out of scope.** Diagnosis. Discharging or reassuring a patient. Triage without
a clinician. Any use outside adult, English-language, interviewer-administered
assessment at the kind of setting it was trained on. Use with any population
outside Uganda, where it has never been evaluated.

## Training data

153 adults recruited at Butabika National Referral Mental Hospital and at Lira
sites (Lira Regional Referral Hospital, Lira University), assessed once each in
an interviewer-administered session that was audio recorded. Ethics approval
obtained; all participants consented to participation and recording.

18 participants were excluded for insufficient recorded speech, leaving **135**.
All 18 came from a single collection period.

| | |
|---|---|
| Analysed | 135 |
| Meeting caseness (HAM-D ≥ 7) | 112 (83.0%) |
| Not meeting caseness | 23 (17.0%) |
| Female / male | 70 / 65 |
| Under 36 years | 88 (65.2%) |
| Secondary education or above | 114 (84.5%) |
| Lira / Butabika / other | 84 / 38 / 13 |
| Audio | 13.49 h, 3,915 recordings |

Interviews were conducted in English. Note that only 54.1% of participants named
English as the language they are most comfortable in.

## Inputs

| Modality | Representation | Dim |
|---|---|---|
| Transcript | frozen `bert-base-uncased`, mean-pooled over 30 s segments | 768 |
| Voice | 24 prosodic measures: pitch (mean, SD, semitone range), vocal effort (energy, dynamic range, voicing), timing (pause rate, pause fraction, mean and longest pause, speech fraction) — each computed per answer, summarised across answers as mean and SD | 24 |
| Demographic | age band, sex, marital status, ethnicity, residence, education, employment, smartphone ownership | 16 |

**Recruitment site is deliberately excluded** from the inputs. It is provenance,
not a clinical characteristic, and including it would let the model distinguish
patients by facility rather than by presentation.

Transcripts are generated automatically (Whisper large-v3). They are not
clinician-verified.

## Model

Ridge regression, α = 1000, on standardised features, fitted on all 135
participants. The standardiser is folded into the coefficients so inference is a
single affine map.

A multilayer perceptron (1552→256→64→2) was evaluated and **rejected**: it lost
on every feature set examined, by 0.08–0.13 AUC. At ~108 training participants
per fold, a 414,000-parameter network memorises.

## Performance

All figures are 5-fold stratified cross-validation repeated 5 times over the
whole cohort, reported as mean (SD) across the 25 folds.

| Feature set | AUC | HAM-D MAE | HAM-D r | permutation p |
|---|---|---|---|---|
| **Prosodic (24)** | **0.826 (0.081)** | 5.85 | 0.413 | **0.016** |
| **Linguistic + prosodic (792)** | **0.816 (0.103)** | **4.99** | **0.508** | **0.016** |
| Linguistic (768) | 0.791 (0.097) | 5.03 | 0.493 | 0.016 |
| Speech quantity (3) | 0.719 (0.137) | 6.04 | 0.306 | 0.016 |
| Self-supervised speech (768) | 0.605 | 5.83 | 0.241 | 0.082 |
| Sociodemographic (16) | 0.568 | 6.02 | 0.277 | 0.213 |
| Constant baseline | — | 6.18 | — | — |

At the calibrated operating point (linguistic + prosodic, Youden):

| | |
|---|---|
| Sensitivity | 0.811 |
| Specificity | 0.708 |
| Balanced accuracy | 0.759 |
| **Positive predictive value** | **0.933** |
| **Negative predictive value** | **0.459** |

### How to read these

**Accuracy is not reported as a headline.** At an 83% base rate a model that
calls everyone a case scores 83% accuracy with zero discrimination. Balanced
accuracy and AUC are the meaningful figures.

**Permutation p-values, not comparison against 0.5.** With 768 features and 135
participants, some features correlate with any fixed outcome by chance, and
cross-validation does not remove it because the same noise is in every fold. The
null distributions here are centred near 0.500 but with SD 0.066–0.079 and 95th
percentiles of 0.595–0.636. Performance is judged against those, not against
chance.

**Negative predictive value is 0.459.** When the model does not prioritise
someone, it is right fewer than half the time. This is partly arithmetic — with
83% prevalence, negative calls are hard for anyone — but it is the binding
constraint on use. The interface says "Lower priority", never "non-case", for
this reason.

## Threshold

The caseness cut point is **10.55 in predicted-HAM-D space**, not the clinical
cutoff of 7. A regularised regression shrinks toward the training mean, so
predicted scores occupy a narrower range than the instrument: at HAM-D ≥ 7 the
model classified almost everyone a case (specificity 0.19) despite ranking them
well. Calibrating raised balanced accuracy from 0.584 to 0.759.

The shipped threshold is fitted on all 135 participants. The reported
performance above used thresholds fitted per training fold. Both are correct for
their purpose; they are not the same number.

## Known limitations

**No external validation.** Performance describes this population only. Nothing
is known about transportability.

**Small sample.** 135 participants with 23 non-cases. All estimates carry wide
uncertainty, and specificity in particular rests on few people.

**Concurrent, not predictive.** Ratings and recordings come from the same
session. The model estimates current severity; it makes no claim about the
future.

**Speech duration is confounded with severity.** Total speech correlates with
symptom severity at r = 0.43 (r = 0.32 excluding the affected sessions). The
model can exploit recording length rather than voice quality. Duration measures
alone reach AUC 0.719, so this is not hypothetical.

**Audio quality and rating quality co-vary.** The 18 excluded participants came
from one collection period and also had lower, less variable symptom scores.
Recording adequacy and rating detail degraded together and cannot be fully
separated in this data.

**Transcripts are machine-generated.** Two-model agreement was 0.981 median
across 3,053 responses, and inspection found instances of fabricated text that
confidence scores alone did not flag. Verification covers 117 of 135
participants.

**Language.** Interviews were in English, but 46% of participants named another
language as their most comfortable. Performance for participants less fluent in
English has not been separately assessed.

**Fairness not assessed.** No subgroup analysis by sex, age, education or site
has been performed. At this sample size such analyses would not be informative,
but their absence means unequal performance across groups cannot be excluded.

## Safety

The suicide referral flag is **rule-based and independent of this model**. It is
computed from clinician-recorded PHQ-9 item 9, the HAM-D suicide domain and a
direct screening question, and fires whenever any is elevated. It is never gated
on model output or confidence and operates identically if the model is absent or
untrained.

In the analysed cohort it flagged 68 of 135 (50.4%). Eleven participants would
have been missed had either instrument item been used alone — which is why the
rule takes the union of all three sources.

## Provenance

Every prediction carries the feature set, input dimension, threshold and fitting
timestamp of the weights that produced it, so a stored result remains traceable
and a later refit does not silently change the meaning of earlier rows.

## Regeneration

```bash
python scripts/fit_final_model.py --cache <fusion_vectors.npz> --prosody <prosody.csv>
python src/eval/benchmark.py --cache ... --prosody ... --permtest 60
python src/eval/calibrate.py --cache ... --prosody ...
```

Feature order is a contract: `prosody_pipeline.FEATURE_NAMES` indexes the
coefficients. Reordering or renaming silently changes what each coefficient
multiplies, and nothing downstream reports an error.

## Maintenance

Roughly 40 further participants are expected. The ingestion pipeline adds them
without disturbing existing participant IDs or train/dev/test assignment, but
the model must be refitted and the threshold re-derived, and this card revised.
Do not add participants and continue serving these weights.
