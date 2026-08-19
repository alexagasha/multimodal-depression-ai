# KoBoToolbox study data → pipeline schema

This is the real training data: an interviewer-administered depression study run
at Butabika and Lira, exported from KoBoToolbox. It replaces E-DAIC-WOZ, which is
now only an optional plumbing smoke test (E-DAIC is PHQ-8, has no HAM-D, and was
never this project's training set).

**Nothing in this directory is data.** The study data lives outside the repo at
`../../../kobo/` (a sibling of the repo). `.gitignore` carries backstop patterns
in case someone points `--out` inside the tree, but the real protection is that
the default output path is outside git entirely.

---

## What the source data actually is

| | |
|---|---|
| Participants | 153 (≈40 more expected) |
| Audio | 4438 `.m4a` clips — **29 per participant, one per interview prompt** |
| Format | AAC, 32 kHz, mono, ~64 kbps, 13.22 hours total |
| Scores | PHQ-9 and HAM-D, **item-level**, plus 9 Section A demographics |
| Linkage | export row carries each clip's filename; folders named by KoBo `_uuid` |
| Completeness | 153/153 participants have all 29 clips; zero missing files |

### Facts verified against the data, not the docs

**HAM-D is scored /44, not /52.** All 11 items are 0–4 and the observed maximum
total is exactly 44. The paper form's `/52` label is a misprint. PHQ-9 totals
recompute exactly against the form's own computed field, max 27.

**The audio contains the participant only — no diarization needed.** Three
independent checks: clip length is uncorrelated with question length
(r=+0.14, p=0.47); the leading speech segment is a flat ~2.25s regardless of
whether the question is 7 or 26 words; and 34% of clips are shorter in total
than it would take to read their own question aloud. Consequently every
transcript row is `speaker="Participant"` and `sync.py`'s
`INTERVIEWER_SPEAKER_NAMES` filter is a no-op here.

**Recruitment sites are Butabika and Lira, not Butabika and Mulago.** There are
zero Mulago participants. `site` is written to metadata for stratified analysis
but is deliberately excluded from the 16-d feature vector
(`metadata_pipeline.py`) so the model cannot shortcut on hospital.

**The spoken audio is English — confirmed by manual listening (2026-08-08).**
The "English only" assumption in `audio_pipeline.py` and the English
`wav2vec2-base-960h` / BERT checkpoints are correct.

Do not be misled by the `language_most_comfortable` field, which records only 80
of 153 as English. That question captures a *preference*, not the language the
interview was conducted in — these were interviewer-administered under an
English-only protocol. The field is kept in metadata for provenance and is not a
model feature.

That column is also unreliable in its own right: all 32 "Luo" answers come from
Luo-ethnicity respondents and none from Bantu, so it is catching the ethnic label
rather than a language. Luo is a family (Lango, Acholi, Alur), not one language —
and since 31 of the 32 are at Lira they are most likely Lango speakers. Ethnicity
and language are separate columns in the export and separate keys in the metadata
JSON; don't read one as the other.

---

## Layout produced

```
<out>/<pid>/
    <pid>_TRANSCRIPT.csv    start_time, stop_time, speaker, value
    <pid>_SEGMENTS.csv      + prompt_index, prompt, source_file  (adapter-internal)
    <pid>_AUDIO.wav         29 clips concatenated, 16 kHz mono PCM16
    <pid>_METADATA.json     Section A, keyed for metadata_pipeline.py
<out>/LABELS.csv            participant_id, phq9_score, hamd_score,
                            phq9_item9_score, hamd_suicide_item_score, split
<out>/QC.csv                per-participant quality metrics + exclusion reasons
<out>/registry.json         uuid -> pid + split. DO NOT DELETE.
<out>/TRANSCRIPTS_PENDING.txt   present until Whisper has run
```

`participant_id` is an integer starting at **1001**, keeping clear of the
synthetic fixtures (300s) and E-DAIC. `src/xai/attribution.py` casts
`participant_id` with `int()`, which is why these are not `P001`-style strings.

### Why the clips are concatenated

The pipeline expects one waveform per participant and slices it into 30s windows
(`sync.build_segments`). Because each source clip's duration is known *before*
concatenation, the transcript's `start_time`/`stop_time` are exact with no
alignment step — only `value` needs filling. A 0.5s silence is inserted between
answers so segment boundaries don't run two unrelated answers together.

---

## Running it

```bash
python data/kobo/adapt_kobo.py
```

Defaults resolve to `../../../kobo/` for input and `../../../kobo/sessions` for
output. Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--min_speech_sec` | `30.0` | total VAD speech floor |
| `--min_speech_per_answer` | `1.0` | mean speech per prompt; catches all-one-word interviews |
| `--gap_sec` | `0.5` | silence inserted between concatenated answers |
| `--limit N` | – | adapt only the first N (smoke test) |
| `--seed` | `1337` | split seed; only affects participants not already in the registry |
| `--refresh` | – | recompute exclusions/splits/LABELS from existing `QC.csv`, no audio decode |
| `--reassign-splits` | – | with `--refresh`, discard and redo **all** splits |

`--refresh` exists because a full pass decodes 4438 clips and takes ~15 minutes,
while retuning a threshold should be instant. Use it to explore exclusion
criteria; use `--reassign-splits` only before anything downstream depends on the
current assignment.

### Current state (153 participants)

135 usable, **18 excluded — every one of them interviewer I4**. Total 13.82 h of
audio (the 0.6 h above the 13.22 h source is the inserted inter-answer gaps),
13.49 h usable. Splits: train 94 / dev 20 / test 21, each carrying negatives
(16 / 3 / 4) and no split more than 38% one interviewer.

Threshold sweep, if you want to move it:

| `--min_speech_sec` | excluded | of which I4 | negatives lost |
|---|---|---|---|
| 20 | 3 | 3 | 0 |
| **30 (default)** | **18** | **18** | **4** |
| 40 | 23 | 23 | 6 |
| 60 | 31 | 29 | 9 |
| 80 | 41 | 35 | 12 |

The default sits where it does because per-answer speech is the honest
discriminator: I4's median participant produces **1.35 s of speech per question**
against 4.7–8.1 s for every other interviewer. Below ~1 s per answer there is no
answer to model, only an utterance.

Then, on Colab with a GPU:

```bash
python data/kobo/transcribe_kobo.py --sessions /content/sessions --audio_root /content/audio
```

Transcription runs against the **original per-prompt clips**, not the
concatenated WAV — short independent utterances transcribe far better than one
long file where Whisper drifts across topic boundaries, and the row mapping
stays exact. Clips under `--min_clip_sec` (default 1.5s) are left blank on
purpose: Whisper hallucinates confidently on sub-second audio ("Thank you.",
"Bye.") and 26% of this corpus is under 3 seconds.

Finally, gate the training run on:

```bash
python data/kobo/validate_kobo.py
```

which exits non-zero if anything is structurally wrong.

### Getting this onto Colab

The adapted WAVs total roughly **1 GB** (16 kHz PCM16 barely compresses), against
**379 MB** for the source `.m4a`. Upload the m4a and run the adapter *on Colab*
rather than uploading the WAVs — `pip install av pandas openpyxl scipy` is all it
needs, and it keeps the round trip small:

```bash
pip install -q av openpyxl
python adapt_kobo.py --xlsx kobo_clean_analytic.xlsx --audio_root audio --out sessions
python transcribe_kobo.py --sessions sessions --audio_root audio
```

Upload only `kobo_clean_analytic.xlsx` and `audio/`. Do **not** upload
`kobo_pid_key.xlsx` — nothing in the pipeline reads it and it carries the
patient names.

---

## Adding the next ~40 participants

1. Export the new data from KoBo (questionnaire XLSX + attachments ZIP).
2. Re-run the cleaning step to regenerate `kobo_clean_analytic.xlsx`.
3. Drop the new audio folders into `../../../kobo/audio/`.
4. Re-run `adapt_kobo.py` with **no flags changed**.

Existing participants keep their `participant_id` **and their train/dev/test
assignment**, because both are keyed on KoBo `_uuid` and persisted in
`registry.json`. Only unseen uuids get new ids and fresh split assignments.

> **Never delete `registry.json`.** Regenerating it reshuffles every split, which
> silently moves material that was in `test` into `train`. If you need a clean
> reshuffle, delete the whole output directory so it is obvious what happened.

Re-running is otherwise idempotent — sessions are overwritten in place and
`transcribe_kobo.py` skips rows that already carry text unless `--overwrite`.

---

## Known problems this data carries

**Severe class imbalance.** Caseness at HAM-D ≥ 7 runs ~82% positive. Whatever
the final N, the negative class is small; stratify splits on caseness (the
adapter does) and weight the loss. Regression on the continuous scores is on much
firmer ground than the binary head.

**One interviewer's audio is systematically degraded.** I4 (`kampa`, Butabika
only) has a median of 76 seconds of audio per participant against 213–526s for
every other interviewer, and 67% of their caseload is under two minutes. It
begins partway through their fieldwork — their first 20 interviews are clean,
then everything from 31 July onward collapses. Their scores are also compressed
(PHQ-9 6.1±2.3 vs 9.7±5.3), so audio quality and label quality degrade together.

**Speech duration leaks the label — and excluding I4 does not fix it.** Total
speech correlates with PHQ-9 at **r = +0.43** across all 153. Drop I4 entirely and
it is still **r = +0.32** (HAM-D: +0.34 → +0.18). So longer answers genuinely
track higher severity in this cohort, independent of the I4 artifact.

That is a shortcut a fusion head can exploit without learning anything about
voice quality: it can read duration off the segment count and never listen. The
exclusion thresholds remove the most degenerate cases, they do not remove the
correlation. Before trusting any audio-branch result, run the ablation — train on
duration alone and see how much of the performance it recovers. If duration-only
gets close to the full model, the audio branch is not doing what you think.

If the incoming ~40 are collected by other interviewers this dilutes; if I4
collects them too, ask about the recording setup before the session starts.

**One ceiling case.** The participant scoring 27/27 on PHQ-9 and 44/44 on HAM-D
is flat at maximum on all twenty items across both instruments — the only person
at either ceiling. It was not a rushed interview (19 minutes of audio, the most
talkative in the study). Worth verifying against the recording before it anchors
the top of the regression range.

**Identifiers.** The raw KoBo `Participant ID` field contains real patient names,
occupations used as IDs, and colliding numbers — it is unusable as a key and is
never read by this pipeline. Participants are keyed on `_uuid`. The mapping back
to names lives in `kobo_pid_key.xlsx`, which is kept *outside* the modelling
directory on purpose: nothing in this pipeline reads it, so the session output
can be uploaded to Colab without carrying identifiers.
