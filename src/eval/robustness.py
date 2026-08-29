"""
Robustness evaluation: performance decline under realistic input noise.

Acceptance gate: <= 5% relative decline in the primary metric (accuracy) under
noise. This is the *harness* — built now so robustness is a tracked gate from
day one and can be run in CI. The numbers only become meaningful once the real
frozen backbones + trained weights replace the mock encoders/untrained head.

Noise models (realistic for phone-recorded, English-language clinic interviews):
  - audio: additive Gaussian noise at a target SNR (dB)
  - text:  character-level corruption (substitute / delete / insert) at a rate

NOTE on the stub: with mock (hash-based) encoders and an untrained head the model
predicts a near-constant score, so accuracy barely moves while hamd_mae shifts as
noise perturbs the embeddings. Expect a trivial accuracy result until real weights
are in place; what we validate here is that the harness runs and measures decline.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipelines.sync import build_segments, DATA_ROOT
from src.pipelines.text_pipeline import BertTextEncoder, clean_text
from src.pipelines.audio_pipeline import (
    Wav2Vec2AudioEncoder, load_audio, resample_if_needed, slice_segment, TARGET_SR,
)
from src.pipelines.metadata_pipeline import run_metadata_pipeline
from src.fusion.aggregate import build_participant_vector, ACOUSTIC
from src.fusion.model import FusionHead, HAMD_CASENESS_THRESHOLD
from src.eval.metrics import compute_metrics

MAX_ALLOWED_DECLINE = 0.05  # 5% relative decline gate
_ALPHABET = list("abcdefghijklmnopqrstuvwxyz ")


def add_audio_noise(waveform: np.ndarray, snr_db: float, rng) -> np.ndarray:
    """Add Gaussian noise at the given signal-to-noise ratio (dB)."""
    if waveform.size == 0:
        return waveform
    sig_power = float(np.mean(waveform ** 2)) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=waveform.shape).astype(np.float32)
    return (waveform + noise).astype(np.float32)


def corrupt_text(text: str, rate: float, rng) -> str:
    """Character-level corruption of ~`rate` fraction of characters."""
    if not text or rate <= 0:
        return text
    chars = list(text)
    n_edits = int(round(len(chars) * rate))
    for _ in range(n_edits):
        if not chars:
            break
        i = int(rng.integers(0, len(chars)))
        op = int(rng.integers(0, 3))
        if op == 0:                                   # substitute
            chars[i] = str(rng.choice(_ALPHABET))
        elif op == 1:                                 # delete
            del chars[i]
        else:                                         # insert
            chars.insert(i, str(rng.choice(_ALPHABET)))
    return "".join(chars)


def _predict(pid, text_enc, audio_enc, model, data_root,
             audio_snr_db=None, text_rate=0.0, seed=0):
    """One participant's prediction, optionally under noise. Returns dict or None."""
    segments = build_segments(pid, data_root=data_root)
    if not segments:
        return None
    rng = np.random.default_rng(seed + int(pid))

    text_embs = []
    for seg in segments:
        txt = clean_text(seg["text"])
        if text_rate:
            txt = corrupt_text(txt, text_rate, rng)
        text_embs.append(text_enc.encode(txt))

    audio, sr = load_audio(pid, data_root)
    audio = resample_if_needed(audio, sr)

    if ACOUSTIC == "prosody":
        # Noise is added to the waveform before feature extraction, not to an
        # embedding afterwards, so the perturbation reaches pitch tracking and
        # voice-activity detection the way real recording noise would. Applied
        # per segment, matching the wav2vec2 branch, so the two conditions differ
        # only in the representation and not in how they are degraded.
        from src.pipelines.prosody_pipeline import prosody_from_turns
        degraded = audio.copy()
        if audio_snr_db is not None:
            for seg in segments:
                a, b = int(seg["start"] * TARGET_SR), int(seg["end"] * TARGET_SR)
                if a < len(degraded):
                    degraded[a:b] = add_audio_noise(degraded[a:b], audio_snr_db, rng)
        turns = [(s["start"], s["end"]) for s in segments]
        acoustic = prosody_from_turns(degraded, turns, TARGET_SR)
    else:
        acoustic = []
        for seg in segments:
            clip = slice_segment(audio, TARGET_SR, seg["start"], seg["end"])
            if audio_snr_db is not None:
                clip = add_audio_noise(clip, audio_snr_db, rng)
            acoustic.append(audio_enc.encode(clip))

    meta = run_metadata_pipeline(pid, data_root=data_root)
    fvec = build_participant_vector(text_embs, acoustic, meta)
    scores = model.forward(fvec)
    return {
        "participant_id": pid,
        "phq9_pred": round(float(scores["phq9"]), 3),
        "hamd_pred": round(float(scores["hamd"]), 3),
        "binary_pred": int(scores["hamd"] >= model.caseness_threshold),
    }


def _predictions_df(labels_df, text_enc, audio_enc, model, data_root, **noise):
    rows = []
    for _, row in labels_df.iterrows():
        r = _predict(int(row["participant_id"]), text_enc, audio_enc, model, data_root, **noise)
        if r is None:
            continue
        r["phq9_true"] = row["phq9_score"]
        r["hamd_true"] = row["hamd_score"]
        r["binary_true"] = int(row["hamd_score"] >= HAMD_CASENESS_THRESHOLD)
        rows.append(r)
    return pd.DataFrame(rows)


def evaluate_robustness(data_root=DATA_ROOT, labels_path=None,
                        audio_snr_levels=(20, 10, 5), text_rates=(0.05, 0.10, 0.20),
                        primary="accuracy", verbose=True):
    """
    Compare clean vs noisy performance; report worst-case relative decline in the
    primary metric and whether it is within the <=5% gate.
    """
    if labels_path is None:
        labels_path = os.path.join(data_root, "LABELS.csv")
    labels_df = pd.read_csv(labels_path)

    text_enc = BertTextEncoder()
    audio_enc = Wav2Vec2AudioEncoder()
    model = FusionHead()

    clean_metrics = compute_metrics(
        _predictions_df(labels_df, text_enc, audio_enc, model, data_root), verbose=False)
    clean_score = clean_metrics[primary]

    conditions = ([("audio@SNR%gdB" % s, {"audio_snr_db": s}) for s in audio_snr_levels]
                  + [("text@%.0f%%" % (r * 100), {"text_rate": r}) for r in text_rates])

    by_condition = []
    worst_decline = 0.0
    for label, cond in conditions:
        m = compute_metrics(
            _predictions_df(labels_df, text_enc, audio_enc, model, data_root, **cond),
            verbose=False)
        noisy_score = m[primary]
        decline = 0.0 if clean_score == 0 else (clean_score - noisy_score) / abs(clean_score)
        worst_decline = max(worst_decline, decline)
        by_condition.append({"condition": label, primary: round(noisy_score, 4),
                             "hamd_mae": m["hamd_mae"], "rel_decline": round(decline, 4)})

    report = {
        "primary_metric": primary,
        "clean_score": round(clean_score, 4),
        "clean_hamd_mae": clean_metrics["hamd_mae"],
        "worst_rel_decline": round(worst_decline, 4),
        "max_allowed_decline": MAX_ALLOWED_DECLINE,
        "gate_passed": bool(worst_decline <= MAX_ALLOWED_DECLINE),
        "by_condition": by_condition,
    }

    if verbose:
        print("=" * 60)
        print("ROBUSTNESS EVALUATION  (primary = %s)" % primary)
        print("=" * 60)
        print("Clean %s: %s   (clean HAM-D MAE: %s)"
              % (primary, report["clean_score"], report["clean_hamd_mae"]))
        for r in by_condition:
            print("  %-14s %s=%-8s hamd_mae=%-8s decline=%.1f%%"
                  % (r["condition"], primary, r[primary], r["hamd_mae"], r["rel_decline"] * 100))
        print("-" * 60)
        verdict = "PASS" if report["gate_passed"] else "FAIL"
        print("Worst decline: %.1f%%  (gate <= %.0f%%): %s"
              % (report["worst_rel_decline"] * 100, MAX_ALLOWED_DECLINE * 100, verdict))
        print("NOTE: mock encoders/untrained head -> numbers not yet meaningful;")
        print("      the harness is what is validated here (see module docstring).")
        print("=" * 60)

    return report


if __name__ == "__main__":
    evaluate_robustness()
