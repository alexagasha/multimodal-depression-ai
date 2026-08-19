"""
Second-model agreement pass over the KoBo transcripts.

WHY
    avg_logprob tells you the model was uncertain. It does not tell you the
    output is wrong. Two independent models decoding the same clip does: where
    large-v3 and medium produce substantially different text, that clip is
    unreliable - hallucination, code-switching, or unintelligible audio. Where
    they agree closely, the transcript is trustworthy regardless of logprob.

    This matters here because forcing --language en on non-English audio does
    not error, it invents fluent English. A single model cannot self-report
    that. Two can.

RESUMABILITY (this is designed to be killed)
    Colab reclaims idle runtimes after ~90 minutes, so an overnight run WILL be
    interrupted. Every participant is flushed to disk the moment it finishes,
    and rows already carrying a verify result are skipped on restart. Point
    --sessions at a Drive-backed directory and a disconnect costs you one
    participant, not the night.

    On reconnect: re-unpack the audio to /content, re-run the same command.

USAGE
    python verify_transcripts.py --sessions <dir> --audio_root <dir>          # pass
    python verify_transcripts.py --sessions <dir> --report                    # summary

OUTPUT
    Into {pid}_SEGMENTS.csv:  verify_text, verify_logprob, agree_ratio, agree_jaccard
    Into <sessions>/agreement_flags.csv:  the rows worth listening to
"""
import argparse
import difflib
import glob
import os
import re
import sys

import pandas as pd

VERIFY_COLS = ["verify_text", "verify_logprob", "agree_ratio", "agree_jaccard"]

# Below these, the two models disagree enough that the clip needs an ear.
RATIO_FLOOR = 0.60      # difflib similarity on normalised characters
JACCARD_FLOOR = 0.50    # token overlap


def _norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def agreement(a, b):
    na, nb = _norm(a), _norm(b)
    if not na and not nb:
        return 1.0, 1.0
    if not na or not nb:
        return 0.0, 0.0
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return round(ratio, 4), round(jac, 4)


def verify_session(model, sessions, audio_root, pid, language, min_clip_sec, index):
    sdir = os.path.join(sessions, str(pid))
    seg_path = os.path.join(sdir, f"{pid}_SEGMENTS.csv")
    tr_path = os.path.join(sdir, f"{pid}_TRANSCRIPT.csv")
    if not (os.path.exists(seg_path) and os.path.exists(tr_path)):
        return None

    segs = pd.read_csv(seg_path, keep_default_na=False)
    tr = pd.read_csv(tr_path, keep_default_na=False)
    for c in VERIFY_COLS:
        if c not in segs.columns:
            segs[c] = ""

    n_done = n_skip = 0
    for i, row in segs.iterrows():
        primary = str(tr.at[i, "value"]).strip()
        if not primary:                       # nothing to compare against
            n_skip += 1
            continue
        if str(segs.at[i, "verify_text"]).strip():   # already done
            continue
        src = index.get(str(row.source_file))
        if not src:
            n_skip += 1
            continue
        if float(row.stop_time) - float(row.start_time) < min_clip_sec:
            n_skip += 1
            continue

        out = model.transcribe(src, language=language, fp16=True,
                               condition_on_previous_text=False)
        vt = out["text"].strip()
        ws = out.get("segments") or []
        segs.at[i, "verify_text"] = vt
        if ws:
            segs.at[i, "verify_logprob"] = round(
                sum(s["avg_logprob"] for s in ws) / len(ws), 4)
        r, j = agreement(primary, vt)
        segs.at[i, "agree_ratio"] = r
        segs.at[i, "agree_jaccard"] = j
        n_done += 1

    # flush per participant - this is what makes an interrupted run cheap
    segs.to_csv(seg_path, index=False)
    return {"pid": pid, "verified": n_done, "skipped": n_skip}


def report(sessions):
    rows = []
    for f in sorted(glob.glob(os.path.join(sessions, "*", "*_SEGMENTS.csv"))):
        pid = os.path.basename(os.path.dirname(f))
        if not pid.isdigit():
            continue
        s = pd.read_csv(f, keep_default_na=False)
        if "agree_ratio" not in s.columns:
            continue
        s["participant_id"] = int(pid)
        rows.append(s)
    if not rows:
        sys.exit(f"No agreement data under {sessions}. Run the pass first.")

    a = pd.concat(rows, ignore_index=True)
    a["agree_ratio"] = pd.to_numeric(a.agree_ratio, errors="coerce")
    a["agree_jaccard"] = pd.to_numeric(a.agree_jaccard, errors="coerce")
    a = a[a.agree_ratio.notna()].copy()

    a["disagree"] = (a.agree_ratio < RATIO_FLOOR) | (a.agree_jaccard < JACCARD_FLOOR)
    print(f"clips with two decodes : {len(a)}")
    print(f"  median char similarity : {a.agree_ratio.median():.3f}")
    print(f"  median token overlap   : {a.agree_jaccard.median():.3f}")
    print(f"  DISAGREE (ratio<{RATIO_FLOOR} or jaccard<{JACCARD_FLOOR}) : "
          f"{int(a.disagree.sum())} ({a.disagree.mean():.1%})")
    print(f"\nagreement distribution:\n"
          f"{a.agree_ratio.describe(percentiles=[.05, .25, .5, .75]).round(3).to_string()}")

    per = a.groupby("participant_id").disagree.agg(["sum", "size"])
    per["rate"] = (per["sum"] / per["size"]).round(3)
    print(f"\nworst participants by disagreement:\n"
          f"{per.sort_values('rate', ascending=False).head(12).to_string()}")

    out = os.path.join(sessions, "agreement_flags.csv")
    cols = [c for c in ["participant_id", "prompt_index", "agree_ratio",
                        "agree_jaccard", "avg_logprob", "verify_logprob",
                        "source_file", "verify_text"] if c in a.columns]
    a[a.disagree][cols].to_csv(out, index=False)
    print(f"\nflagged rows -> {out}")
    print("\nRead this as: high disagreement = listen to the clip. Low "
          "disagreement = trust the transcript even if avg_logprob is poor "
          "(accented English scores low but decodes consistently).")
    return a


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--audio_root", default=None)
    ap.add_argument("--model", default="medium",
                    help="second model; must DIFFER from the one used originally")
    ap.add_argument("--language", default="en")
    ap.add_argument("--min_clip_sec", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", action="store_true", help="summarise, don't transcribe")
    a = ap.parse_args()

    if a.report:
        report(a.sessions)
        sys.exit(0)
    if not a.audio_root:
        sys.exit("--audio_root is required for the verification pass")

    pids = sorted(int(d) for d in os.listdir(a.sessions)
                  if d.isdigit() and os.path.isdir(os.path.join(a.sessions, d)))
    pids = [p for p in pids
            if os.path.exists(os.path.join(a.sessions, str(p), f"{p}_SEGMENTS.csv"))]
    if not pids:
        sys.exit(f"No usable session dirs in {a.sessions}")
    if a.limit:
        pids = pids[:a.limit]

    index = {}
    for d in os.listdir(a.audio_root):
        p = os.path.join(a.audio_root, d)
        if os.path.isdir(p):
            for fn in os.listdir(p):
                index[fn] = os.path.join(p, fn)
    print(f"{len(index)} source clips indexed; {len(pids)} participants")

    import whisper
    model = whisper.load_model(a.model)

    done = 0
    for n, pid in enumerate(pids, 1):
        r = verify_session(model, a.sessions, a.audio_root, pid, a.language,
                           a.min_clip_sec, index)
        if r:
            done += r["verified"]
        print(f"[{n}/{len(pids)}] {r}", flush=True)

    print(f"\nverified {done} clips with '{a.model}'")
    print("now run:  --report   for the agreement summary")
