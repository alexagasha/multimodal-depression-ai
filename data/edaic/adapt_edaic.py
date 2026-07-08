"""
Adapt an E-DAIC-WOZ session into the schema this pipeline expects.

E-DAIC ships, per session:  {pid}_AUDIO.wav, {pid}_Transcript.csv (Start_Time,
End_Time, Text[, Confidence]), and a large features/ folder. Our raw-waveform +
transcript architecture only needs the WAV and the transcript — the features/
folder (OpenSMILE / OpenFace / CNN .mat) is ignored (hand-engineered + video,
which this project does not use).

Produces, under <out>/<pid>/:
    {pid}_TRANSCRIPT.csv   start_time, stop_time, speaker, value
    {pid}_AUDIO.wav        (copied verbatim)
    {pid}_METADATA.json    sex + language (+ source tag)
and appends a row to <out>/LABELS.csv when an E-DAIC label CSV is supplied.

CAVEATS (why this is a smoke test, not an evaluation of the objectives):
  - E-DAIC has PHQ-8 only -> phq9_score is filled from PHQ-8 as a proxy, and
    hamd_score is left blank (no HAM-D in E-DAIC), so HAM-D accuracy cannot be
    evaluated here.
  - E-DAIC is English -> language is set to "english"; the ~3-language
    robustness gate is not exercised.
  - E-DAIC transcripts have no speaker column and may include a few setup/agent
    lines at the start; all rows are kept as participant speech.
"""
import argparse
import glob
import json
import os
import shutil

import pandas as pd


def _find(session_dir, *patterns):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(session_dir, pat)))
        if hits:
            return hits[0]
    return None


def _norm_transcript(src_csv):
    df = pd.read_csv(src_csv)
    lower = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    c_start = pick("start_time", "start", "starttime")
    c_stop = pick("stop_time", "end_time", "end", "stop", "endtime")
    c_text = pick("value", "text", "utterance", "transcript")
    c_spk = pick("speaker", "participant")
    if c_start is None or c_stop is None or c_text is None:
        raise ValueError(f"Unrecognised transcript columns in {src_csv}: {list(df.columns)}")

    return pd.DataFrame({
        "start_time": df[c_start].astype(float),
        "stop_time": df[c_stop].astype(float),
        "speaker": df[c_spk] if c_spk else "Participant",
        "value": df[c_text].fillna("").astype(str),
    })


def _lookup_label(labels_csv, pid):
    if not labels_csv or not os.path.exists(labels_csv):
        return None
    df = pd.read_csv(labels_csv)
    lower = {c.lower().strip(): c for c in df.columns}
    id_col = lower.get("participant_id") or lower.get("participant") or lower.get("id")
    if id_col is None:
        return None
    row = df[df[id_col].astype(str) == str(pid)]
    if row.empty:
        return None
    row = row.iloc[0]

    def g(*names):
        for n in names:
            if n in lower:
                return row[lower[n]]
        return None

    return {"phq8": g("phq8_score", "phq_score", "phq8", "phq_8_score"),
            "gender": g("gender", "sex")}


def adapt_session(session_dir, pid, out_dir, split="test", labels_csv=None):
    """Convert one E-DAIC session folder into a pipeline session dir. Returns the dir."""
    session_dir = os.path.abspath(session_dir)
    wav = _find(session_dir, "*AUDIO.wav", "*.wav")
    tsv = _find(session_dir, "*ranscript*.csv", "*TRANSCRIPT*.csv", "*transcript*.csv")
    if wav is None or tsv is None:
        raise FileNotFoundError(f"Need a .wav and a transcript .csv in {session_dir}")

    dst = os.path.join(out_dir, str(pid))
    os.makedirs(dst, exist_ok=True)

    _norm_transcript(tsv).to_csv(os.path.join(dst, f"{pid}_TRANSCRIPT.csv"), index=False)
    shutil.copyfile(wav, os.path.join(dst, f"{pid}_AUDIO.wav"))

    lab = _lookup_label(labels_csv, pid)
    sex = "unknown"
    if lab and lab.get("gender") is not None:
        g = str(lab["gender"]).strip().lower()
        sex = {"1": "female", "0": "male", "f": "female", "m": "male",
               "female": "female", "male": "male"}.get(g, "unknown")
    meta = {"participant_id": int(pid), "sex": sex, "language": "english",
            "source": "e-daic-woz"}
    with open(os.path.join(dst, f"{pid}_METADATA.json"), "w") as f:
        json.dump(meta, f)

    if lab is not None and lab.get("phq8") is not None and not pd.isna(lab["phq8"]):
        # phq9_score <- PHQ-8 (proxy; the two differ by item 9). hamd_score unknown.
        rec = {"participant_id": int(pid), "phq9_score": int(round(float(lab["phq8"]))),
               "hamd_score": "", "split": split}
        labels_path = os.path.join(out_dir, "LABELS.csv")
        if os.path.exists(labels_path):
            df = pd.read_csv(labels_path)
            df = df[df["participant_id"] != int(pid)]
            df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
        else:
            df = pd.DataFrame([rec], columns=["participant_id", "phq9_score", "hamd_score", "split"])
        df.to_csv(labels_path, index=False)

    print(f"Adapted E-DAIC session {pid} -> {dst}"
          + ("" if lab else "  (no label row; pass --labels_csv for PHQ-8)"))
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Adapt an E-DAIC session to the pipeline schema.")
    ap.add_argument("--session", required=True, help="E-DAIC session folder (holds the .wav + transcript)")
    ap.add_argument("--pid", required=True, type=int, help="participant id, e.g. 304")
    ap.add_argument("--out", default=os.path.join("data", "edaic", "sessions"))
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--labels_csv", default=None, help="E-DAIC split/label CSV with PHQ-8 + gender")
    args = ap.parse_args()
    adapt_session(args.session, args.pid, args.out, args.split, args.labels_csv)
