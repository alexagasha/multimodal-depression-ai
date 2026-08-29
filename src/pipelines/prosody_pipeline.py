"""
Prosody pipeline: waveform -> 24 interpretable acoustic features per participant.

WHY THIS EXISTS
    wav2vec2-base-960h is trained for speech recognition. Its final layer
    encodes phonetic content — which the text branch already carries — and
    attenuates the paralinguistic properties by which depressed speech is
    clinically described. Evaluated over the whole cohort it did not clear its
    own permutation null (AUC 0.605, p = 0.082) and lost to three measures of
    speech quantity. These 24 explicit measures reached AUC 0.826 with p = 0.016,
    the best of any single modality, and unlike a 768-dimensional embedding they
    can be named to a clinician: reduced pitch variation, longer pauses.

THE FEATURE ORDER IS A CONTRACT
    FEATURE_NAMES below is the exact column order the shipped model was fitted
    on. The weights in outputs/weights/fusion_head_808.npz are indexed by it.
    Reordering, inserting or renaming anything here silently changes what each
    coefficient multiplies, and nothing downstream would report an error.

MATCHING TRAINING-TIME COMPUTATION
    In training each measure was computed per interview response, from the
    original per-prompt recording, then summarised across a participant's
    responses as a mean and a standard deviation. At inference the application
    holds one concatenated waveform plus a transcript carrying start_time and
    stop_time per turn, so the same measures are computed per TURN by slicing
    the waveform, and aggregated identically. Computing over the whole waveform
    in one pass would produce different features from those the model was fitted
    on — plausible ones, and wrong.

    Constants below (frame size, VAD margin, pause threshold, minimum turn
    length, sample-level std convention) all reproduce the training script and
    must not be tuned independently of a refit.
"""
import os

import numpy as np
import pandas as pd

SR = 16000
FRAME_SEC, HOP_SEC = 0.040, 0.010
VAD_MARGIN_DB = 18.0          # below the 92nd percentile
VAD_FLOOR_DB = -55.0
MIN_VOICED_FRAMES = 5
PAUSE_MIN_SEC = 0.200         # silent runs at least this long count as pauses
MIN_TURN_SEC = 1.0            # shorter turns contribute nothing
F0_MIN, F0_MAX = 60.0, 400.0
F0_PEAK_MIN = 0.35            # autocorrelation peak below this -> unvoiced

# Per-turn measures, in the order they are summarised.
BASE_MEASURES = [
    "speech_frac", "f0_mean", "f0_sd", "f0_range_st", "voiced_frac",
    "rms_mean", "rms_sd", "db_range", "n_pauses_per_s", "pause_frac",
    "mean_pause", "longest_pause",
]
# Each becomes a mean and an SD across turns. Duration measures are excluded:
# how MUCH was said is already captured elsewhere and correlates with severity
# on its own, so including it here would confound the acoustic signal.
FEATURE_NAMES = [f"{m}_{stat}" for m in BASE_MEASURES for stat in ("mean", "sd")]
EMBED_DIM = len(FEATURE_NAMES)          # 24

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic", "sessions")


# --------------------------------------------------------------------------
def _frames(x):
    win, hop = int(FRAME_SEC * SR), int(HOP_SEC * SR)
    if len(x) < win:
        return np.zeros((0, win), dtype=np.float32)
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def _f0_track(fr):
    """Autocorrelation pitch per frame; nan where unvoiced or weakly periodic.

    Only ever called on voiced frames — the unvoiced ones are discarded by
    every consumer, so computing them was pure waste. The transform length is
    the smallest even size that still avoids circular wrap-around (2 * window),
    not the next power of two: numpy factorises 1280 efficiently, and the
    smaller transform is a further saving at identical output."""
    if len(fr) == 0:
        return np.zeros(0, dtype=np.float64)
    fr = fr - fr.mean(axis=1, keepdims=True)
    w = np.hanning(fr.shape[1])[None, :]
    nfft = 2 * fr.shape[1]
    S = np.fft.rfft(fr * w, n=nfft, axis=1)
    ac = np.fft.irfft(np.abs(S) ** 2, n=nfft, axis=1)[:, : fr.shape[1]]
    a0 = ac[:, :1].copy()
    a0[a0 == 0] = 1e-12
    acn = ac / a0
    lo, hi = int(SR / F0_MAX), min(int(SR / F0_MIN), acn.shape[1] - 1)
    seg = acn[:, lo:hi]
    if seg.size == 0:
        return np.full(len(fr), np.nan)
    k = np.argmax(seg, axis=1)
    peak = seg[np.arange(len(seg)), k]
    f0 = SR / np.maximum(k + lo, 1)
    f0[peak < F0_PEAK_MIN] = np.nan
    return f0


def turn_features(x):
    """Pitch, effort and timing for one spoken turn. None if unusable."""
    fr = _frames(np.asarray(x, dtype=np.float32))
    if len(fr) == 0:
        return None
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    thr = max(np.percentile(db, 92) - VAD_MARGIN_DB, VAD_FLOOR_DB)
    voiced = db > thr
    if voiced.sum() < MIN_VOICED_FRAMES:
        return None

    # Pitch is only ever read at voiced frames, so only compute it there.
    # f0_at_voiced[i] corresponds to the i-th voiced frame.
    f0_at_voiced = _f0_track(fr[voiced])
    f0v = f0_at_voiced[~np.isnan(f0_at_voiced)]

    # interior silences only: a pause is bounded by speech on both sides
    pauses, i, min_frames = [], 0, int(PAUSE_MIN_SEC / HOP_SEC)
    while i < len(voiced):
        if not voiced[i]:
            j = i
            while j < len(voiced) and not voiced[j]:
                j += 1
            if 0 < i and j < len(voiced) and (j - i) >= min_frames:
                pauses.append((j - i) * HOP_SEC)
            i = j
        else:
            i += 1

    dur = len(x) / SR
    return {
        "speech_frac": float(voiced.mean()),
        "f0_mean": float(np.mean(f0v)) if len(f0v) > 5 else np.nan,
        "f0_sd": float(np.std(f0v)) if len(f0v) > 5 else np.nan,
        # semitones are speaker-normalised, so a low-pitched and a high-pitched
        # speaker with equally flat delivery score alike — raw Hz cannot do this
        "f0_range_st": (float(12 * np.log2(np.percentile(f0v, 90)
                                           / max(np.percentile(f0v, 10), 1e-6)))
                        if len(f0v) > 20 else np.nan),
        "voiced_frac": (float(np.mean(~np.isnan(f0_at_voiced)))
                        if len(f0_at_voiced) else np.nan),
        "rms_mean": float(rms[voiced].mean()),
        "rms_sd": float(rms[voiced].std()),
        "db_range": float(np.percentile(db[voiced], 90) - np.percentile(db[voiced], 10)),
        "n_pauses_per_s": len(pauses) / dur,
        "pause_frac": float(sum(pauses) / dur),
        "mean_pause": float(np.mean(pauses)) if pauses else 0.0,
        "longest_pause": float(max(pauses)) if pauses else 0.0,
    }


def aggregate_turns(rows):
    """Mean and SD of each measure across a participant's turns.

    The SD columns carry real signal: how much delivery varies between
    questions is itself a marker, and averaging alone would erase it.
    Sample standard deviation (ddof=1) and nan -> 0 both reproduce the
    pandas defaults used when the shipped model was fitted."""
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    if not rows:
        return vec
    df = pd.DataFrame(rows)
    for i, name in enumerate(FEATURE_NAMES):
        measure, stat = name.rsplit("_", 1)
        if measure not in df.columns:
            continue
        s = df[measure]
        v = s.mean() if stat == "mean" else s.std()      # ddof=1, as in training
        vec[i] = 0.0 if pd.isna(v) else float(v)
    return vec


# --------------------------------------------------------------------------
def _load_wav(path):
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        from scipy.io import wavfile
        sr, raw = wavfile.read(path)
        audio = (raw.astype(np.float32) / float(np.iinfo(raw.dtype).max)
                 if np.issubdtype(raw.dtype, np.integer) else raw.astype(np.float32))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def prosody_from_turns(audio, turns, sr=SR):
    """turns: iterable of (start_sec, stop_sec). Returns the 24-d vector."""
    rows = []
    for start, stop in turns:
        if stop - start < MIN_TURN_SEC:
            continue
        seg = audio[int(start * sr):int(stop * sr)]
        f = turn_features(seg)
        if f:
            rows.append(f)
    return aggregate_turns(rows)


def run_prosody_pipeline(pid, data_root=DATA_ROOT):
    """Participant-level prosody from the session waveform and transcript."""
    session = os.path.join(data_root, str(pid))
    wav = os.path.join(session, f"{pid}_AUDIO.wav")
    tsv = os.path.join(session, f"{pid}_TRANSCRIPT.csv")
    if not os.path.exists(wav):
        raise FileNotFoundError(f"no {pid}_AUDIO.wav in {session}")
    audio, sr = _load_wav(wav)
    if sr != SR:
        raise ValueError(f"{pid}_AUDIO.wav is {sr} Hz; prosody features were "
                         f"computed at {SR} Hz and are not comparable across rates")

    if os.path.exists(tsv):
        t = pd.read_csv(tsv)
        turns = list(zip(t["start_time"].astype(float), t["stop_time"].astype(float)))
    else:
        # No transcript: fall back to one turn spanning the recording. This is
        # NOT equivalent to the per-turn computation the model was fitted on and
        # should only occur outside the normal path.
        print(f"[prosody] no transcript for {pid}; computing over the whole "
              f"recording, which does not match training-time features")
        turns = [(0.0, len(audio) / sr)]
    return prosody_from_turns(audio, turns, sr)


if __name__ == "__main__":
    import sys
    pid = sys.argv[1] if len(sys.argv) > 1 else "300"
    root = sys.argv[2] if len(sys.argv) > 2 else DATA_ROOT
    v = run_prosody_pipeline(pid, root)
    print(f"prosody vector for {pid}: dim={v.shape[0]}")
    for n, x in zip(FEATURE_NAMES, v):
        print(f"  {n:<22} {x: .4f}")
