# Running the KoBo data on Colab

End-to-end: upload → adapt → detect language → transcribe → bring results home.

Budget **2–3 hours** on a T4 for the transcription step. Everything before it takes
about 20 minutes.

---

## Step 1 — Upload (local, ~10 min depending on your connection)

Upload the single file `kobo/kobo_colab_bundle.zip` (380 MB) to Google Drive.
Put it at the top level of *My Drive* so the paths below work unchanged.

Upload the **zip**, not the folder. Google Drive uploads 4438 loose files roughly
an order of magnitude slower than one archive of the same size, and the browser
tab frequently stalls partway through a large folder upload.

> **Do not upload `kobo_pid_key.xlsx`.** It maps participants back to patient
> names. Nothing in the pipeline reads it, and it is deliberately not in the
> bundle — I checked.

---

## Step 2 — New Colab notebook, GPU on

<https://colab.research.google.com> → New notebook →
**Runtime → Change runtime type → T4 GPU** → Save.

Verify:

```python
!nvidia-smi
```

If that errors, the GPU isn't attached and Whisper will run on CPU — roughly 20×
slower. Fix it before continuing.

---

## Step 3 — Mount Drive and unpack

```python
from google.colab import drive
drive.mount('/content/drive')

!cp "/content/drive/MyDrive/kobo_colab_bundle.zip" /content/
!cd /content && unzip -q kobo_colab_bundle.zip -d kobo
!ls /content/kobo && ls /content/kobo/audio | head -3
!ls /content/kobo/audio | wc -l      # expect 153
```

Unpack to `/content`, **not** to Drive. `/content` is local SSD; Drive is a
network mount and decoding 4438 clips across it is painfully slow.

---

## Step 4 — Install the adapter's dependencies

```python
!pip install -q av openpyxl
```

`av` is PyAV — it carries its own ffmpeg, which is what decodes the `.m4a` files.
Without it step 5 dies with `ModuleNotFoundError: No module named 'av'` the moment
it touches the first clip.

**Install `openai-whisper` later, not here** (it's step 6a). It pulls in `numba`,
which pins particular numpy versions and can prompt a Colab runtime restart — you
do not want that landing halfway through a 15-minute decode.

---

## Step 5 — Build the session layout (~10–15 min)

```python
!cd /content/kobo && python code/adapt_kobo.py \
    --xlsx kobo_clean_analytic.xlsx \
    --audio_root audio \
    --out /content/sessions
```

Expect: `sessions written: 153`, `usable: 135`, `excluded: 18`, splits
`train 94 / dev 20 / test 21`. If those numbers differ, something is off — stop
and check before transcribing.

Then verify:

```python
!cd /content/kobo && python code/validate_kobo.py --out /content/sessions
```

21 checks should pass, with warnings only for pending transcripts and one null
`age_band`.

### Checkpoint the small files to Drive now

```python
!mkdir -p "/content/drive/MyDrive/kobo_results"
!cd /content/sessions && \
  tar czf "/content/drive/MyDrive/kobo_results/sessions_meta.tgz" \
  --exclude='*.wav' .
```

That's under 1 MB and it is everything you can't cheaply regenerate. The 1.48 GB
of WAVs stays on `/content` — it's derived data, rebuildable by re-running step 5.

---

## Step 6 — Language check (optional, ~2 min)

**Already settled: the audio is English**, confirmed by manual listening on
2026-08-08. Go straight to step 7 with `--language en`.

The `language_most_comfortable` field showing only 80/153 as English is a
*preference* question, not the language of the interview — these were
interviewer-administered under an English-only protocol. Ignore it here.

Keep this as a cheap sanity check if you ever add participants from a new site or
interviewer:

```python
!cd /content/kobo && python code/transcribe_kobo.py \
    --sessions /content/sessions \
    --audio_root audio \
    --detect-language --limit 20
```

Worth knowing why it would matter: Whisper `large-v3` has no model for Luganda,
Lango/Acholi, Runyankole or Ateso. On unsupported audio it does not fail loudly —
it emits fluent text in a language it *does* know. So a future batch that drifted
off-protocol would poison the text branch with plausible-looking invented
sentences rather than erroring.

---

## Step 6a — Now install Whisper

```python
!pip install -q openai-whisper
```

If Colab prompts to restart the runtime, accept it. `/content` survives a restart,
so `sessions/` and the unpacked audio are still there — just re-run the mount cell
and continue. (A *disconnect* is different and does wipe `/content`; see the
recovery steps at the end of step 7.)

---

## Step 7 — Transcribe (~2–3 h on T4)

**Do not run this as one three-hour cell.** `/content` is wiped on disconnect and
the free tier caps at 12 hours, so a single long cell risks losing everything.
Run it in batches that checkpoint to Drive after each one:

```python
import subprocess, os

os.makedirs('/content/drive/MyDrive/kobo_results', exist_ok=True)
TOTAL, BATCH = 153, 20

for end in range(BATCH, TOTAL + BATCH, BATCH):
    print(f"\n===== up to participant {min(end, TOTAL)} =====", flush=True)
    subprocess.run(
        ['python', 'code/transcribe_kobo.py',
         '--sessions', '/content/sessions', '--audio_root', 'audio',
         '--language', 'en', '--limit', str(end)],
        cwd='/content/kobo', check=True)
    subprocess.run(
        ['tar', 'czf', '/content/drive/MyDrive/kobo_results/sessions_meta.tgz',
         '--exclude=*.wav', '-C', '/content/sessions', '.'], check=True)
    print(f"checkpointed to Drive after {min(end, TOTAL)}", flush=True)

print("\nall done")
```

This leans on the script being resumable — rows already carrying text are skipped,
so each pass does only the new batch and cheaply re-scans the rest. Worst case a
disconnect costs you 20 participants, about 20 minutes.

Clips under 1.5s are skipped deliberately — Whisper hallucinates confidently on
sub-second audio ("Thank you.", "Bye.") and 26% of this corpus is under 3s.

### If the runtime dies mid-run

You do **not** need to re-run the adapter. Transcription reads only
`_SEGMENTS.csv`, `_TRANSCRIPT.csv` and the source `.m4a` files — it never touches
the WAVs, and the checkpoint tarball carries every CSV it needs. Recovery is ~3
minutes, not 18:

```python
from google.colab import drive
drive.mount('/content/drive')

!cp "/content/drive/MyDrive/kobo_colab_bundle.zip" /content/
!cd /content && unzip -q kobo_colab_bundle.zip -d kobo
!mkdir -p /content/sessions
!tar xzf "/content/drive/MyDrive/kobo_results/sessions_meta.tgz" -C /content/sessions
!pip install -q openai-whisper
```

Then re-run the batch loop; it skips everything already transcribed. No `av`
install needed either, since nothing is being decoded.

Re-run the adapter (step 5) only when you actually want the WAVs back — that is
at training time, not during transcription.

---

## Step 8 — Bring the results home

```python
!cd /content/sessions && \
  tar czf "/content/drive/MyDrive/kobo_results/sessions_final.tgz" \
  --exclude='*.wav' .
```

Download `sessions_final.tgz` from Drive, then locally:

```bash
tar xzf sessions_final.tgz -C "C:/Users/Dev/Desktop/depression ai model/kobo/sessions"
python data/kobo/validate_kobo.py
```

The `TRANSCRIPTS_PENDING.txt` warning should now be gone. If it isn't, some
transcripts are still empty — check `transcription_report.json` for which.

---

## What not to do

**Don't upload the WAVs.** 1.48 GB against 380 MB of source, for a file you can
rebuild in 15 minutes with one command.

**Don't run `adapt_kobo.py` with different `--min_speech_sec` on Colab than you
used locally** without also re-running it locally. The exclusion set feeds the
splits, and you want one answer for which participants are in the study.

**Don't delete `registry.json`** — it is inside `sessions_meta.tgz` and it is what
keeps participant IDs and train/dev/test assignment stable. Losing it reshuffles
the splits and quietly moves test material into train.
