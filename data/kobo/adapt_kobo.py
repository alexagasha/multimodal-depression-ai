"""
Adapt the KoBoToolbox study export into the schema this pipeline expects.

KoBo ships, per participant: 29 separate .m4a clips (one per interview prompt,
AAC 32 kHz mono) in a folder named by KoBo submission UUID, plus one XLSX row
carrying PHQ-9 / HAM-D / Section A demographics.

Produces, under <out>/<pid>/:
    {pid}_TRANSCRIPT.csv   start_time, stop_time, speaker, value
    {pid}_AUDIO.wav        29 clips concatenated, 16 kHz mono PCM16
    {pid}_METADATA.json    Section A fields, keyed for metadata_pipeline.py
and at <out>/:
    LABELS.csv             participant_id, phq9_score, hamd_score,
                           phq9_item9_score, hamd_suicide_item_score, split
    QC.csv                 per-participant quality metrics + exclusion reasons
    registry.json          uuid -> pid + split, so re-runs are stable

WHY THE CLIPS ARE CONCATENATED
    The pipeline expects one waveform per participant and slices it into 30s
    windows (sync.build_segments). Because each clip's duration is known before
    concatenation, the transcript's start_time/stop_time are exact without any
    alignment step — only `value` needs filling, which transcribe_kobo.py does.

WHY THERE IS NO DIARIZATION
    Verified against the data: clip length is uncorrelated with question length
    (r=+0.14, p=0.47) and 34% of clips are shorter than it would take to read
    their own question aloud. The interviewer is not in the recordings, so every
    transcript row is speaker="Participant" and sync.py's INTERVIEWER_SPEAKER_NAMES
    filter is a no-op here.

RE-RUN SAFETY (the study is still collecting; ~40 more expected)
    Participant IDs and train/dev/test assignment are keyed on KoBo _uuid and
    persisted in registry.json. Re-running after new submissions arrive keeps
    every existing participant's id and split identical, and assigns new ones
    only to unseen uuids. Never delete registry.json — doing so reshuffles the
    splits and leaks test material into train.
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

TARGET_SR = 16000          # matches audio_pipeline.TARGET_SR
PID_BASE = 1000            # kobo pids start at 1001; keeps clear of synthetic (300s)
DEFAULT_GAP_SEC = 0.5      # silence inserted between consecutive answers

# Section A -> the exact vocabulary metadata_pipeline.py matches on.
# NOTE: the export writes age bands with an EN DASH (18–25); the pipeline
# matches an ASCII hyphen. Unnormalised, every participant silently encodes
# as "unknown" (0.5). _norm() below fixes this.
# Each map accepts BOTH the raw KoBo label and the already-normalised token, so
# the adapter works against the cleaned workbook or a fresh raw export.
AGE = {"18-25", "26-35", "36-45", "46-55", "56-65"}
MARITAL = {"married": "married", "divorced": "divorced",
           "widow/er": "widowed", "widowed": "widowed", "single": "single"}
ETHNIC = {"bantu": "bantu", "luo": "luo", "other": "other"}
RESID = {"urban": "urban", "semi-urban": "semi-urban", "rural": "rural"}
EDU = {"no formal education": "none", "none": "none", "primary": "primary",
       "secondary": "secondary", "tertiary / university": "tertiary_university",
       "tertiary/university": "tertiary_university",
       "tertiary_university": "tertiary_university"}
EMPLOY = {"employed": "employed", "unemployed": "unemployed",
          "self-employed": "self-employed", "student": "student"}
SEX = {"male": "male", "female": "female"}
YESNO = {"yes": "yes", "no": "no"}


def _norm(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = re.sub(r"\s+", " ", str(v)).replace("–", "-").replace("—", "-").strip()
    return s or None


def _score(v):
    """'Not at all (0)' / '2 - Moderate' -> int."""
    s = _norm(v)
    if s is None:
        return None
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------
def _decode(path, sr=TARGET_SR):
    """Decode any container to mono float32 at `sr` via PyAV (bundled ffmpeg)."""
    import av
    with av.open(path) as c:
        rs = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=sr)
        chunks = []
        for frame in c.decode(audio=0):
            for out in rs.resample(frame):
                chunks.append(out.to_ndarray().ravel())
        for out in rs.resample(None):          # flush
            chunks.append(out.to_ndarray().ravel())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _speech_seconds(x, sr=TARGET_SR):
    """Energy-VAD speech duration. Used for the usability threshold, because
    clip count is a constant 29 for everyone and tells you nothing."""
    win, hop = int(0.040 * sr), int(0.010 * sr)
    if len(x) < win:
        return 0.0
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt((x[idx] ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    thr = max(np.percentile(db, 92) - 18, -55)
    return float((db > thr).sum() * 0.010)


def _write_wav(path, audio, sr=TARGET_SR):
    from scipy.io import wavfile
    clipped = np.clip(audio, -1.0, 1.0)
    wavfile.write(path, sr, (clipped * 32767.0).astype(np.int16))


# --------------------------------------------------------------------------
# registry: stable pid + split across re-runs
# --------------------------------------------------------------------------
def _load_registry(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"by_uuid": {}, "next_pid": PID_BASE + 1}


def _save_registry(path, reg):
    with open(path, "w") as f:
        json.dump(reg, f, indent=2, sort_keys=True)


def _stratified_splits(df, seed, ratios):
    """Stratify on caseness so each split carries a representative share of the
    scarce negatives. Interviewer is deliberately NOT grouped on: with one
    interviewer holding 45 of 153 participants, grouping would make a whole
    split single-interviewer, which is worse than the leakage it prevents.
    Interviewer composition per split is reported in QC.csv instead.

    Seeding is derived arithmetically, NOT from hash() — Python randomises
    string hashing per process, so a hash-derived seed would silently produce
    different splits on every machine.
    """
    out = {}
    for case_val, grp in df.groupby("caseness"):
        ids = sorted(grp.uuid.tolist())
        rng = np.random.default_rng((int(seed) * 2 + int(bool(case_val))) % (2**32))
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(n * ratios[0]))
        n_dv = int(round(n * ratios[1]))
        for i, u in enumerate(ids):
            out[u] = "train" if i < n_tr else ("dev" if i < n_tr + n_dv else "test")
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def _finalise(out_dir, qcdf, reg, reg_path, labels, min_speech_sec,
              min_speech_per_answer, seed, ratios, reassign=False):
    """Apply usability thresholds, assign splits, write LABELS/QC/registry.

    Split out from adapt() so --refresh can retune thresholds without
    re-decoding 4438 clips (a full pass takes ~15 minutes)."""
    qcdf = qcdf.copy()
    qcdf["speech_per_answer"] = (qcdf.speech_sec /
                                 qcdf.clips.replace(0, np.nan)).round(2)

    def usable_row(r):
        if r.clips == 0:
            return False
        if r.speech_sec < min_speech_sec:
            return False
        if (r.speech_per_answer or 0) < min_speech_per_answer:
            return False
        return True

    qcdf["usable"] = qcdf.apply(usable_row, axis=1)

    def reason_row(r):
        # NB: float('nan') is truthy, so `r.reason or ""` would keep "nan"
        prev = "" if pd.isna(r.reason) else str(r.reason)
        parts = [p for p in prev.split(";")
                 if p and p != "nan"          # literal, from an earlier refresh
                 and not p.startswith("low_speech")
                 and not p.startswith("thin_answers")]
        if r.clips == 0:
            parts.append("no_audio")
        if r.speech_sec < min_speech_sec:
            parts.append(f"low_speech({r.speech_sec:.0f}s<{min_speech_sec:.0f}s)")
        if (r.speech_per_answer or 0) < min_speech_per_answer:
            parts.append(f"thin_answers({r.speech_per_answer:.2f}s/answer"
                         f"<{min_speech_per_answer}s)")
        return ";".join(dict.fromkeys(parts))

    qcdf["reason"] = qcdf.apply(reason_row, axis=1)

    if reassign:
        for u in reg["by_uuid"]:
            reg["by_uuid"][u]["split"] = None

    usable = qcdf[qcdf.usable].copy()
    # participants that just became unusable must not keep a split
    for u in reg["by_uuid"]:
        if u not in set(usable.uuid):
            reg["by_uuid"][u]["split"] = None
    fresh = usable[usable.uuid.map(
        lambda u: reg["by_uuid"].get(u, {}).get("split") is None)]
    if len(fresh):
        for u, s in _stratified_splits(fresh, seed, ratios).items():
            reg["by_uuid"][u]["split"] = s
        print(f"assigned splits to {len(fresh)} participant(s)")
    qcdf["split"] = qcdf.uuid.map(lambda u: reg["by_uuid"].get(u, {}).get("split"))

    lab = labels.set_index("participant_id")
    recs = []
    for _, q in qcdf[qcdf.usable & qcdf.split.notna()].iterrows():
        s = lab.loc[q.sheet_pid]
        recs.append({
            "participant_id": q.participant_id,
            "phq9_score": int(s.phq9_score),
            "hamd_score": int(s.hamd_score),
            "phq9_item9_score": int(s.phq9_item9_score),
            "hamd_suicide_item_score": int(s.hamd_suicide_item_score),
            "split": q.split,
        })
    pd.DataFrame(recs, columns=["participant_id", "phq9_score", "hamd_score",
                                "phq9_item9_score", "hamd_suicide_item_score",
                                "split"]).to_csv(
        os.path.join(out_dir, "LABELS.csv"), index=False)
    qcdf.to_csv(os.path.join(out_dir, "QC.csv"), index=False)
    _save_registry(reg_path, reg)
    return qcdf


def refresh(out_dir, xlsx, min_speech_sec, min_speech_per_answer, seed,
            ratios=(0.70, 0.15, 0.15), reassign=False):
    """Recompute exclusions, splits and LABELS.csv from an existing QC.csv."""
    reg_path = os.path.join(out_dir, "registry.json")
    reg = _load_registry(reg_path)
    qcdf = pd.read_csv(os.path.join(out_dir, "QC.csv"))
    labels = pd.read_excel(xlsx, sheet_name="labels")
    qcdf = _finalise(out_dir, qcdf, reg, reg_path, labels, min_speech_sec,
                     min_speech_per_answer, seed, ratios, reassign)
    _report(out_dir, qcdf)
    return qcdf


def adapt(xlsx, audio_root, out_dir, min_speech_sec=30.0, gap_sec=DEFAULT_GAP_SEC,
          seed=1337, ratios=(0.70, 0.15, 0.15), limit=None,
          min_speech_per_answer=1.0):
    os.makedirs(out_dir, exist_ok=True)
    reg_path = os.path.join(out_dir, "registry.json")
    reg = _load_registry(reg_path)

    labels = pd.read_excel(xlsx, sheet_name="labels")
    audio_map = pd.read_excel(xlsx, sheet_name="audio_map")
    prompts = pd.read_excel(xlsx, sheet_name="prompts")

    qcol = [c for c in audio_map.columns if c.endswith("_file")]
    prompt_text = dict(zip(prompts.prompt_index, prompts.column_name))

    # Locate each participant's folder by clip FILENAME, not by the identifier
    # key. The audio folders happen to be named by KoBo submission uuid, but
    # deriving the link from filenames keeps kobo_pid_key.xlsx (which holds the
    # patient names) entirely out of the modelling path.
    index = {}
    for d in os.listdir(audio_root):
        p = os.path.join(audio_root, d)
        if os.path.isdir(p):
            for fn in os.listdir(p):
                index[fn] = (d, p)

    rows = labels.merge(audio_map, on="participant_id", how="left")
    if limit:
        rows = rows.head(limit)

    qc = []
    print(f"adapting {len(rows)} participants -> {out_dir}")
    for i, r in rows.reset_index(drop=True).iterrows():
        sheet_pid = r.participant_id
        # resolve the folder from the first clip filename that exists on disk
        uuid, folder = None, None
        for c in qcol:
            fn = _norm(r[c])
            if fn and fn in index:
                uuid, folder = index[fn]
                break
        if uuid is None:
            uuid = f"NOAUDIO::{sheet_pid}"

        # stable pid
        if uuid in reg["by_uuid"]:
            pid = reg["by_uuid"][uuid]["pid"]
        else:
            pid = reg["next_pid"]
            reg["next_pid"] += 1
            reg["by_uuid"][uuid] = {"pid": pid, "sheet_pid": sheet_pid, "split": None}

        dst = os.path.join(out_dir, str(pid))
        os.makedirs(dst, exist_ok=True)

        # ---- audio: concatenate the 29 clips in prompt order ----
        segs, parts, missing = [], [], 0
        t = 0.0
        gap = np.zeros(int(gap_sec * TARGET_SR), dtype=np.float32)
        for k, c in enumerate(qcol, start=1):
            fn = _norm(r[c])
            path = os.path.join(folder, fn) if (folder and fn) else None
            if not path or not os.path.exists(path):
                missing += 1
                continue
            x = _decode(path)
            if len(x) == 0:
                missing += 1
                continue
            dur = len(x) / TARGET_SR
            segs.append({"start_time": round(t, 3), "stop_time": round(t + dur, 3),
                         "speaker": "Participant", "value": "",
                         "prompt_index": k,
                         "prompt": str(prompt_text.get(k, ""))[:200],
                         "source_file": fn})
            parts.append(x)
            if k < len(qcol):
                parts.append(gap)
                t += dur + gap_sec
            else:
                t += dur

        if not parts:
            qc.append({"participant_id": pid, "sheet_pid": sheet_pid, "uuid": uuid,
                       "usable": False, "reason": "no_audio", "clips": 0,
                       "audio_sec": 0.0, "speech_sec": 0.0})
            print(f"  [{pid}] NO AUDIO — skipped")
            continue

        audio = np.concatenate(parts)
        _write_wav(os.path.join(dst, f"{pid}_AUDIO.wav"), audio)
        speech = _speech_seconds(audio)

        # ---- transcript scaffold (value filled later by transcribe_kobo.py) ----
        tdf = pd.DataFrame(segs)
        tdf[["start_time", "stop_time", "speaker", "value"]].to_csv(
            os.path.join(dst, f"{pid}_TRANSCRIPT.csv"), index=False)
        tdf.to_csv(os.path.join(dst, f"{pid}_SEGMENTS.csv"), index=False)

        # ---- metadata ----
        age = _norm(r.age_band)
        lk = lambda v, table: table.get((_norm(v) or "").lower())
        meta = {
            "participant_id": pid,
            "age_band": age if age in AGE else None,
            "sex": lk(r.sex, SEX),
            "marital_status": lk(r.marital_status, MARITAL),
            "ethnicity": lk(r.ethnicity, ETHNIC),
            "residence": lk(r.residence, RESID),
            "education_level": lk(r.education_level, EDU),
            "employment_status": lk(r.employment_status, EMPLOY),
            "smartphone": lk(r.smartphone, YESNO),
            # provenance only - metadata_pipeline excludes these from the vector
            "site": _norm(r.site),
            "interviewer_code": _norm(r.interviewer_code),
            "language_most_comfortable": _norm(r.language_most_comfortable),
            "source": "kobo-uganda-2026",
        }
        with open(os.path.join(dst, f"{pid}_METADATA.json"), "w") as f:
            json.dump(meta, f, indent=2)

        reasons = []
        if missing:
            reasons.append(f"missing_clips({missing})")
        # any unmapped demographic silently encodes as "unknown" in the 16-d
        # vector, so surface it rather than letting it pass
        unmapped = [k for k in ("age_band", "sex", "marital_status", "ethnicity",
                                "residence", "education_level", "employment_status",
                                "smartphone") if meta[k] is None]
        if unmapped:
            reasons.append("unmapped:" + "+".join(unmapped))

        qc.append({
            "participant_id": pid, "sheet_pid": sheet_pid, "uuid": uuid,
            "usable": True,   # decided in _finalise from the speech thresholds
            "reason": ";".join(reasons), "clips": len(segs), "missing_clips": missing,
            "audio_sec": round(len(audio) / TARGET_SR, 1),
            "speech_sec": round(speech, 1),
            "speech_ratio": round(speech / (len(audio) / TARGET_SR), 3),
            "phq9_score": int(r.phq9_score), "hamd_score": int(r.hamd_score),
            "caseness": bool(r.caseness_hamd_ge7),
            "interviewer_code": meta["interviewer_code"], "site": meta["site"],
            "language": meta["language_most_comfortable"],
        })
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(rows)}")

    qcdf = pd.DataFrame(qc)
    qcdf = _finalise(out_dir, qcdf, reg, reg_path, labels, min_speech_sec,
                     min_speech_per_answer, seed, ratios)
    _report(out_dir, qcdf)
    return qcdf


def _report(out_dir, qcdf):
    # TRANSCRIPT.csv rows carry correct timings and speakers but EMPTY text until
    # transcribe_kobo.py runs. Left unfilled, pandas reads "" back as NaN and
    # sync.participant_only_ranges() stringifies it to the literal "nan" - which
    # would silently train BERT on garbage. Mark it loudly.
    pending = os.path.join(out_dir, "TRANSCRIPTS_PENDING.txt")
    unfilled = []
    for _, q in qcdf.iterrows():
        tp = os.path.join(out_dir, str(q.participant_id),
                          f"{q.participant_id}_TRANSCRIPT.csv")
        if os.path.exists(tp):
            t = pd.read_csv(tp, keep_default_na=False)
            if (t["value"].astype(str).str.strip() == "").all():
                unfilled.append(str(q.participant_id))
    if unfilled:
        with open(pending, "w") as f:
            f.write("Transcripts are scaffolds only - timings and speakers are\n"
                    "correct, `value` is empty. Run transcribe_kobo.py on Colab\n"
                    "before any text/fusion run.\n\n" + "\n".join(unfilled) + "\n")
    elif os.path.exists(pending):
        os.remove(pending)

    # ---- report ----
    print("\n" + "=" * 66)
    print(f"sessions written : {len(qcdf)}")
    print(f"usable           : {int(qcdf.usable.sum())}")
    print(f"excluded         : {int((~qcdf.usable).sum())}")
    if (~qcdf.usable).any():
        print(qcdf[~qcdf.usable][["participant_id", "speech_sec", "speech_per_answer",
                                  "interviewer_code", "reason"]].to_string(index=False))
    print(f"\ntotal audio      : {qcdf.audio_sec.sum() / 3600:.2f} h")
    print(f"usable audio     : {qcdf[qcdf.usable].audio_sec.sum() / 3600:.2f} h")
    u = qcdf[qcdf.usable & qcdf.split.notna()]
    print(f"\nsplits: {u.split.value_counts().to_dict()}")
    print("\ncaseness per split:")
    print(pd.crosstab(u.split, u.caseness).to_string())
    print("\ninterviewer per split (watch for a split dominated by one):")
    print(pd.crosstab(u.split, u.interviewer_code).to_string())
    bad = [k for k, v in qcdf[qcdf.reason.str.contains("unmapped", na=False)]
           .reason.items()]
    if bad:
        print(f"\n! {len(bad)} participant(s) have an unmapped demographic "
              f"(would encode as unknown) - see QC.csv")
    if unfilled:
        print(f"\n! TRANSCRIPTS ARE EMPTY for {len(unfilled)} participant(s).")
        print("  Run transcribe_kobo.py on Colab before any text or fusion run.")
    print("=" * 66)
    return qcdf


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default_kobo = os.path.abspath(os.path.join(here, "..", "..", "..", "kobo"))
    ap = argparse.ArgumentParser(description="Adapt the KoBo study export to the pipeline schema.")
    ap.add_argument("--xlsx", default=os.path.join(default_kobo, "kobo_clean_analytic.xlsx"))
    ap.add_argument("--audio_root", default=os.path.join(default_kobo, "audio"))
    ap.add_argument("--out", default=os.path.join(default_kobo, "sessions"))
    ap.add_argument("--min_speech_sec", type=float, default=30.0,
                    help="total VAD speech below this -> excluded")
    ap.add_argument("--min_speech_per_answer", type=float, default=1.0,
                    help="mean speech per prompt below this -> excluded "
                         "(catches interviews where every answer is one word)")
    ap.add_argument("--gap_sec", type=float, default=DEFAULT_GAP_SEC)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit", type=int, default=None, help="adapt only the first N (smoke test)")
    ap.add_argument("--refresh", action="store_true",
                    help="recompute exclusions/splits/LABELS from the existing "
                         "QC.csv without re-decoding audio (~instant vs ~15 min)")
    ap.add_argument("--reassign-splits", dest="reassign", action="store_true",
                    help="with --refresh: discard and redo ALL split assignments. "
                         "Only safe before anything downstream depends on them.")
    a = ap.parse_args()
    if a.refresh:
        refresh(a.out, a.xlsx, a.min_speech_sec, a.min_speech_per_answer,
                a.seed, reassign=a.reassign)
    else:
        adapt(a.xlsx, a.audio_root, a.out, a.min_speech_sec, a.gap_sec, a.seed,
              limit=a.limit, min_speech_per_answer=a.min_speech_per_answer)
