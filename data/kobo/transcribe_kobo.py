"""
Fill the `value` column of every {pid}_TRANSCRIPT.csv using Whisper large-v3.

Run this on Colab (GPU). adapt_kobo.py has already written the hard part —
each transcript row's start_time/stop_time is exact, because the boundaries
come from the known duration of each source clip, not from alignment. So this
job transcribes each of the 29 clips independently and drops the text into the
matching row. No forced alignment, no diarization.

    pip install -q openai-whisper
    python transcribe_kobo.py --sessions /content/sessions --audio_root /content/audio

Design notes
------------
* Transcribes the ORIGINAL per-prompt clips, not the concatenated WAV. Short
  independent utterances transcribe far better than one long file where Whisper
  drifts across topic boundaries, and the row mapping stays exact.
* Clips below --min_clip_sec are left blank rather than transcribed. Whisper
  hallucinates confidently on sub-second audio ("Thank you.", "Bye.") and that
  text would become training data. 26% of this corpus is under 3 seconds.
* Resumable: a clip already carrying text is skipped unless --overwrite.
* --language defaults to en. The audio was confirmed English by manual listening
  (2026-08-08); the `language_most_comfortable` field showing 80/153 English is a
  *preference* question, not the interview language. Forcing en beats auto-detect
  here: detection is unreliable on 2-second accented utterances.

ON TRANSLATION
    --task translate makes Whisper output English from another source language.
    It is deliberately NOT the default and will not rescue this dataset: Whisper
    has no model for Luganda, Lango/Acholi, Runyankole or Ateso, and on
    unsupported audio it does not error - it emits fluent English that was never
    said. detect_language cannot warn you either, since it always returns one of
    its 99 known languages. Use --task translate only for a language in
    WHISPER_LANGS below.

QUALITY SIGNALS
    Every transcribed row records avg_logprob, no_speech_prob and the compression
    ratio into {pid}_SEGMENTS.csv. These are how you catch hallucination and
    code-switching after the fact - see --flag_only. A participant dropping a
    Luganda phrase into an English answer produces plausible-looking English with
    a poor avg_logprob, which is invisible in the text alone.
"""
import argparse
import json
import os
import sys

import pandas as pd

QUALITY_COLS = ["avg_logprob", "no_speech_prob", "compression_ratio",
                "asr_language", "n_chars"]


def load_model(size="large-v3"):
    import whisper
    return whisper.load_model(size)


def transcribe_session(model, sessions, audio_root, pid, language, min_clip_sec,
                       overwrite, detect_only):
    sdir = os.path.join(sessions, str(pid))
    seg_path = os.path.join(sdir, f"{pid}_SEGMENTS.csv")
    tr_path = os.path.join(sdir, f"{pid}_TRANSCRIPT.csv")
    if not os.path.exists(seg_path):
        return None

    segs = pd.read_csv(seg_path, keep_default_na=False)
    tr = pd.read_csv(tr_path, keep_default_na=False)

    # locate the source clips by filename (folders are named by kobo uuid)
    index = {}
    for d in os.listdir(audio_root):
        p = os.path.join(audio_root, d)
        if os.path.isdir(p):
            for fn in os.listdir(p):
                index[fn] = os.path.join(p, fn)

    for c in QUALITY_COLS:
        if c not in segs.columns:
            segs[c] = ""

    langs, n_done, n_skip = [], 0, 0
    for i, row in segs.iterrows():
        existing = str(tr.at[i, "value"]).strip()
        if existing and not overwrite:
            continue
        src = index.get(str(row.source_file))
        if not src:
            n_skip += 1
            continue
        dur = float(row.stop_time) - float(row.start_time)
        if dur < min_clip_sec:
            tr.at[i, "value"] = ""
            n_skip += 1
            continue

        if detect_only:
            import whisper
            audio = whisper.load_audio(src)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
            _, probs = model.detect_language(mel)
            top = max(probs, key=probs.get)
            langs.append({"pid": pid, "prompt_index": int(row.prompt_index),
                          "dur": round(dur, 2), "lang": top,
                          "p": round(float(probs[top]), 4)})
            continue

        out = model.transcribe(src, language=language, fp16=True,
                               condition_on_previous_text=False)
        tr.at[i, "value"] = out["text"].strip()

        # Per-clip quality signals. These are how hallucination and
        # code-switching get caught after the fact - the text alone looks fine.
        ws = out.get("segments") or []
        if ws:
            segs.at[i, "avg_logprob"] = round(
                sum(s["avg_logprob"] for s in ws) / len(ws), 4)
            segs.at[i, "no_speech_prob"] = round(
                max(s["no_speech_prob"] for s in ws), 4)
            segs.at[i, "compression_ratio"] = round(
                max(s["compression_ratio"] for s in ws), 3)
        segs.at[i, "asr_language"] = out.get("language", language)
        segs.at[i, "n_chars"] = len(str(tr.at[i, "value"]))
        n_done += 1

    if detect_only:
        return {"pid": pid, "languages": langs}

    tr.to_csv(tr_path, index=False)
    segs.to_csv(seg_path, index=False)
    return {"pid": pid, "transcribed": n_done, "skipped": n_skip,
            "filled": int((tr["value"].astype(str).str.strip() != "").sum()),
            "rows": len(tr)}


def flag_sessions(sessions):
    """Report clips whose transcription looks unreliable. Run after transcribing.

    Thresholds are Whisper's own internal fallback triggers:
      avg_logprob      < -1.00  the decode was low-confidence
      compression_ratio > 2.40  output is repetitive (classic hallucination loop)
      no_speech_prob    > 0.60  the model thinks there is no speech at all
    A clip tripping one of these is a candidate for manual review, not an
    automatic delete - accented English legitimately scores lower.
    """
    rows = []
    for d in sorted(os.listdir(sessions)):
        sp = os.path.join(sessions, d, f"{d}_SEGMENTS.csv")
        if not (d.isdigit() and os.path.exists(sp)):
            continue
        s = pd.read_csv(sp, keep_default_na=False)
        if "avg_logprob" not in s.columns:
            continue
        s["participant_id"] = int(d)
        rows.append(s)
    if not rows:
        sys.exit(f"No quality columns found under {sessions}. Transcribe first "
                 f"(this data is written during transcription).")

    a = pd.concat(rows, ignore_index=True)
    a = a[pd.to_numeric(a.avg_logprob, errors="coerce").notna()].copy()
    for c in ("avg_logprob", "no_speech_prob", "compression_ratio", "n_chars"):
        a[c] = pd.to_numeric(a[c], errors="coerce")

    a["low_conf"] = a.avg_logprob < -1.00
    a["repetitive"] = a.compression_ratio > 2.40
    a["no_speech"] = a.no_speech_prob > 0.60
    a["suspect"] = a.low_conf | a.repetitive | a.no_speech

    print(f"clips with quality data : {len(a)}")
    print(f"  low confidence  (avg_logprob < -1.00) : {int(a.low_conf.sum())}")
    print(f"  repetitive     (compression > 2.40)   : {int(a.repetitive.sum())}")
    print(f"  no speech      (no_speech_prob > .60) : {int(a.no_speech.sum())}")
    print(f"  SUSPECT (any of the above)            : {int(a.suspect.sum())}"
          f"  ({a.suspect.mean():.1%})")
    print(f"\navg_logprob distribution:\n"
          f"{a.avg_logprob.describe(percentiles=[.05, .25, .5, .75]).round(3).to_string()}")

    per = a.groupby("participant_id").suspect.agg(["sum", "size"])
    per["rate"] = per["sum"] / per["size"]
    worst = per.sort_values("rate", ascending=False).head(10)
    print(f"\nworst participants by suspect rate:\n{worst.to_string()}")

    out = os.path.join(sessions, "quality_flags.csv")
    a[a.suspect][["participant_id", "prompt_index", "avg_logprob",
                  "no_speech_prob", "compression_ratio", "n_chars",
                  "source_file"]].to_csv(out, index=False)
    print(f"\nflagged rows -> {out}")
    return a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="en")
    ap.add_argument("--min_clip_sec", type=float, default=1.5,
                    help="below this, leave blank - Whisper hallucinates on very short audio")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--detect-language", dest="detect", action="store_true",
                    help="report detected language per clip instead of transcribing")
    ap.add_argument("--flag_only", action="store_true",
                    help="no transcription: re-scan existing quality signals and "
                         "report clips that look hallucinated or code-switched")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    if a.flag_only:
        flag_sessions(a.sessions)
        sys.exit(0)

    pids = sorted(int(d) for d in os.listdir(a.sessions)
                  if d.isdigit() and os.path.isdir(os.path.join(a.sessions, d)))

    # Fail loudly rather than reporting success on zero work. A crashed
    # adapt_kobo.py leaves exactly one empty session dir behind (it mkdirs
    # before decoding), which used to sail through this as "[1/1] None".
    if not pids:
        sys.exit(f"No session directories in {a.sessions}. Run adapt_kobo.py first.")
    ready = [p for p in pids
             if os.path.exists(os.path.join(a.sessions, str(p), f"{p}_SEGMENTS.csv"))]
    if not ready:
        sys.exit(f"{len(pids)} session dir(s) in {a.sessions} but none contain a "
                 f"_SEGMENTS.csv — adapt_kobo.py did not complete. Re-run it and "
                 f"wait for 'sessions written: N' before transcribing.")
    if len(ready) < len(pids):
        print(f"! {len(pids) - len(ready)} session dir(s) have no _SEGMENTS.csv "
              f"and will be skipped — adapt_kobo.py may have been interrupted.")
    pids = ready
    if a.limit:
        pids = pids[:a.limit]

    model = load_model(a.model)
    results = []
    for n, pid in enumerate(pids, 1):
        r = transcribe_session(model, a.sessions, a.audio_root, pid, a.language,
                               a.min_clip_sec, a.overwrite, a.detect)
        if r:
            results.append(r)
        print(f"[{n}/{len(pids)}] {r}")

    if a.detect:
        from collections import Counter
        flat = [x for r in results for x in r["languages"]]
        c = Counter(x["lang"] for x in flat)
        print("\ndetected languages across all clips:", dict(c.most_common()))

        # Detection on short accented speech is close to a coin toss - the
        # detector must return one of its 99 languages and cannot say "unsure".
        # Cross-tabbing against clip duration shows whether a non-'en' result
        # carries information or is just the classifier guessing.
        det = pd.DataFrame(flat)
        det["bucket"] = pd.cut(det.dur, [0, 3, 6, 12, 30, 1e9],
                               labels=["<3s", "3-6s", "6-12s", "12-30s", ">30s"])
        det["is_en"] = det.lang == "en"
        tab = det.groupby("bucket", observed=False).agg(
            clips=("is_en", "size"), pct_en=("is_en", "mean"),
            mean_conf=("p", "mean"))
        tab["pct_en"] = (tab.pct_en * 100).round(1)
        tab["mean_conf"] = tab.mean_conf.round(3)
        print(f"\nEnglish share by clip length:\n{tab.to_string()}")
        print("\nIf pct_en climbs sharply with duration, the non-English results "
              "are detector noise on short clips, not code-switching.")
        out = os.path.join(a.sessions, "language_detection.csv")
        det.to_csv(out, index=False)
        print(f"per-clip detections -> {out}")
    else:
        out = os.path.join(a.sessions, "transcription_report.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        tot = sum(r["transcribed"] for r in results)
        blank = sum(r["rows"] - r["filled"] for r in results)
        filled = sum(r["filled"] for r in results)
        print(f"\ntranscribed {tot} clips; {blank} rows left blank (too short/missing)")
        print(f"report -> {out}")
        if filled == 0:
            sys.exit("Nothing was transcribed and no row carries text. Check that "
                     "--audio_root points at the folder holding the KoBo uuid "
                     "directories, and that adapt_kobo.py completed.")
