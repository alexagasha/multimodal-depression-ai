"""
Validate an adapted KoBo dataset before it goes anywhere near training.

Run after adapt_kobo.py, and again after adding new participants:

    python data/kobo/validate_kobo.py

Checks structure, schema, value ranges, split integrity, and that every
session actually round-trips through the real pipeline code (metadata ->
16-d vector, audio -> 16 kHz load, transcript -> 30s segments). Exits
non-zero if anything fails, so it can gate a training run.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

def _wire_repo(repo=None):
    """Put the real pipeline modules on sys.path.

    This validator deliberately imports the production code rather than
    reimplementing its checks. That means it needs the repo's src/ tree —
    which is NOT in the Colab bundle (that carries only data/kobo/*.py). On
    Colab either pass --repo, or skip validation there and run it locally
    after pulling the results back.
    """
    repo = repo or os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    src = os.path.join(repo, "src")
    if not os.path.isdir(os.path.join(src, "pipelines")):
        sys.exit(
            f"Cannot find the pipeline modules under {src}.\n"
            f"validate_kobo.py checks the data by importing the real "
            f"metadata_pipeline / audio_pipeline / sync / risk_flag code.\n"
            f"  - locally: run it from the repo, or pass --repo <repo root>\n"
            f"  - on Colab: skip this step; validate after copying results home")
    sys.path.insert(0, os.path.join(src, "pipelines"))
    sys.path.insert(0, src)
    return repo


PHQ9_MAX, HAMD_MAX, ITEM9_MAX, SUICIDE_MAX = 27, 44, 3, 4
LABEL_COLS = ["participant_id", "phq9_score", "hamd_score",
              "phq9_item9_score", "hamd_suicide_item_score", "split"]


class Report:
    def __init__(self):
        self.fail, self.warn, self.ok = [], [], []

    def check(self, cond, msg, warn_only=False):
        if cond:
            self.ok.append(msg)
        elif warn_only:
            self.warn.append(msg)
        else:
            self.fail.append(msg)
        return cond

    def show(self):
        for m in self.ok:
            print(f"  PASS  {m}")
        for m in self.warn:
            print(f"  WARN  {m}")
        for m in self.fail:
            print(f"  FAIL  {m}")
        print(f"\n{len(self.ok)} passed, {len(self.warn)} warnings, {len(self.fail)} failures")
        return len(self.fail) == 0


def main(out_dir):
    import metadata_pipeline as mp
    import audio_pipeline as ap
    import sync
    from safety.risk_flag import flag_risk

    r = Report()
    print(f"validating {out_dir}\n")

    # ---- top-level files ----
    labels_path = os.path.join(out_dir, "LABELS.csv")
    r.check(os.path.exists(labels_path), "LABELS.csv exists")
    r.check(os.path.exists(os.path.join(out_dir, "registry.json")),
            "registry.json exists (needed for stable splits on re-run)")
    if not os.path.exists(labels_path):
        return r.show()

    lab = pd.read_csv(labels_path)
    r.check(list(lab.columns) == LABEL_COLS,
            f"LABELS.csv schema == spec (got {list(lab.columns)})")
    r.check(lab.participant_id.is_unique, "participant_id unique in LABELS.csv")
    r.check(not lab.isna().any().any(), "no NaN in LABELS.csv")

    # ---- value ranges ----
    for col, hi in [("phq9_score", PHQ9_MAX), ("hamd_score", HAMD_MAX),
                    ("phq9_item9_score", ITEM9_MAX),
                    ("hamd_suicide_item_score", SUICIDE_MAX)]:
        bad = lab[(lab[col] < 0) | (lab[col] > hi)]
        r.check(len(bad) == 0, f"{col} within 0..{hi} "
                               f"(observed {lab[col].min()}..{lab[col].max()})")

    r.check(set(lab.split) <= {"train", "dev", "test"},
            f"splits are train/dev/test (got {sorted(set(lab.split))})")

    # ---- per-session structure ----
    pids = sorted(int(d) for d in os.listdir(out_dir)
                  if d.isdigit() and os.path.isdir(os.path.join(out_dir, d)))
    r.check(len(pids) > 0, f"{len(pids)} session directories found")
    labelled = set(lab.participant_id)
    r.check(labelled <= set(pids),
            "every LABELS row has a session directory")

    missing_files, bad_time, bad_sr, unknown_dims, short_seg = [], [], [], [], []
    null_meta = []
    durations, seg_counts = [], []
    for pid in pids:
        d = os.path.join(out_dir, str(pid))
        for suffix in ("_TRANSCRIPT.csv", "_AUDIO.wav", "_METADATA.json"):
            if not os.path.exists(os.path.join(d, f"{pid}{suffix}")):
                missing_files.append(f"{pid}{suffix}")

        t = pd.read_csv(os.path.join(d, f"{pid}_TRANSCRIPT.csv"), keep_default_na=False)
        if not (t.start_time.is_monotonic_increasing and
                (t.start_time.values[1:] >= t.stop_time.values[:-1]).all() and
                (t.stop_time >= t.start_time).all()):
            bad_time.append(pid)

        audio, sr = ap.load_audio(pid, out_dir)
        if sr != ap.TARGET_SR or audio.ndim != 1:
            bad_sr.append(pid)
        durations.append(len(audio) / sr)
        # transcript must not claim time beyond the waveform
        if float(t.stop_time.max()) > len(audio) / sr + 0.05:
            bad_time.append(pid)

        meta = json.load(open(os.path.join(d, f"{pid}_METADATA.json")))
        vec = mp.encode_metadata(meta)
        if vec.shape != (mp.EMBED_DIM,):
            unknown_dims.append(pid)
        # one-hot blocks must each sum to 1; 0 means the category never matched
        if not (np.isclose(vec[2:6].sum(), 1) and np.isclose(vec[6:9].sum(), 1)
                and np.isclose(vec[11:15].sum(), 1)):
            unknown_dims.append(pid)
        # Ordinal fields can't be checked from the vector - 0.5 is the "unknown"
        # sentinel but also the legitimate encoding of age_band 36-45 and
        # residence semi-urban. Check the source JSON instead.
        null_fields = [k for k in ("age_band", "sex", "marital_status", "ethnicity",
                                   "residence", "education_level",
                                   "employment_status", "smartphone")
                       if meta.get(k) is None]
        if null_fields:
            null_meta.append((pid, "+".join(null_fields)))

        segs = sync.build_segments(pid, out_dir)
        seg_counts.append(len(segs))
        if len(segs) == 0:
            short_seg.append(pid)

    r.check(not missing_files, f"all sessions have TRANSCRIPT/AUDIO/METADATA "
                               f"({len(missing_files)} missing)")
    r.check(not bad_time, f"transcript timings monotonic, non-overlapping, "
                          f"within waveform ({len(set(bad_time))} bad)")
    r.check(not bad_sr, f"all audio is mono at {ap.TARGET_SR} Hz ({len(bad_sr)} bad)")
    r.check(not unknown_dims,
            f"all metadata encodes to {mp.EMBED_DIM}-d with every categorical matched "
            f"({len(set(unknown_dims))} with unmatched fields)")
    r.check(not null_meta,
            "no null demographic fields (each encodes as 'unknown' in the vector)"
            + (f" — {', '.join(f'{p}:{f}' for p, f in null_meta[:5])}" if null_meta else ""),
            warn_only=True)
    r.check(not short_seg, f"every session yields >=1 30s segment ({len(short_seg)} empty)")

    # ---- transcripts filled? ----
    pending = os.path.join(out_dir, "TRANSCRIPTS_PENDING.txt")
    r.check(not os.path.exists(pending),
            "transcripts are filled (run transcribe_kobo.py if this warns)",
            warn_only=True)

    # ---- split integrity ----
    if len(lab):
        vc = lab.split.value_counts()
        r.check(len(vc) >= 2, f"more than one split populated: {vc.to_dict()}")
        case = None
        qc_path = os.path.join(out_dir, "QC.csv")
        if os.path.exists(qc_path):
            qc = pd.read_csv(qc_path)
            m = lab.merge(qc[["participant_id", "caseness", "interviewer_code"]],
                          on="participant_id", how="left")
            ct = pd.crosstab(m.split, m.caseness)
            case = ct
            # every split should contain at least one negative
            if False in ct.columns:
                r.check((ct[False] > 0).all(),
                        f"every split contains negatives: {ct[False].to_dict()}")
            else:
                r.check(False, "no negative cases in any split")
            dom = pd.crosstab(m.split, m.interviewer_code)
            worst = (dom.max(axis=1) / dom.sum(axis=1)).max()
            r.check(worst < 0.6,
                    f"no split is dominated by one interviewer (max share {worst:.0%})",
                    warn_only=True)

    # ---- referral flag ----
    flags = [flag_risk(int(x.phq9_item9_score), int(x.hamd_suicide_item_score))
             for x in lab.itertuples()]
    r.check(True, f"referral flag computes for all {len(flags)} rows "
                  f"({sum(flags)} positive, {sum(flags)/max(len(flags),1):.0%})")

    print(f"\n--- summary ---")
    print(f"sessions        : {len(pids)}")
    print(f"labelled        : {len(lab)}")
    print(f"audio           : {sum(durations)/3600:.2f} h "
          f"(median {np.median(durations):.0f}s per participant)")
    print(f"30s segments    : {sum(seg_counts)} total, median {np.median(seg_counts):.0f}/participant")
    print(f"splits          : {lab.split.value_counts().to_dict()}")
    if case is not None:
        print(f"caseness/split  :\n{case.to_string()}")
    print()
    return r.show()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default = os.path.abspath(os.path.join(here, "..", "..", "..", "kobo", "sessions"))
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", default=default)
    ap_.add_argument("--repo", default=None,
                     help="repo root holding src/pipelines (needed off-repo, e.g. Colab)")
    a = ap_.parse_args()
    _wire_repo(a.repo)
    sys.exit(0 if main(a.out) else 1)
