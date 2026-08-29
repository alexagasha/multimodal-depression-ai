"""
Guards the one failure mode in the prosody pipeline that nothing else catches.

Training features were computed from the original per-prompt recordings,
decoded individually. Inference computes them from slices of a re-encoded,
concatenated 16-bit waveform. If those drift apart, the model is served
different features from the ones it was fitted on — and because the result is
still a well-formed vector producing a plausible score, no error is raised
anywhere downstream. Only a comparison against the training-time values reveals
it.

WHY THE TOLERANCE IS ON PREDICTIONS, NOT ON FEATURES
    A per-feature tolerance was tried first and rejected as the wrong
    instrument. Re-encoding to 16-bit shifts frame energies slightly, which
    moves the percentile-based voice-activity threshold, which occasionally
    pushes a silence across the 200 ms pause cutoff — adding or removing a whole
    pause. That makes the five pause-derived features disagree by up to ~7%
    while pitch, energy and speech-fraction agree to within 1%.

    Measured end to end, that discrepancy moves predicted HAM-D by at most 0.06
    points on a 0-44 scale, against the model's own cross-validated error of
    4.99 points, and changed no caseness classification. So the feature-level
    figure overstates a difference that does not reach the output. The
    tolerance below is therefore expressed where it matters, with a loose
    feature-level check kept to catch gross breakage.

These tests skip when the study data is absent, which is the normal case: it is
clinical data and is deliberately not in the repository.
"""
import os

import numpy as np
import pytest

from src.pipelines.prosody_pipeline import (
    run_prosody_pipeline, FEATURE_NAMES, EMBED_DIM, aggregate_turns, turn_features,
)

SESSIONS = os.environ.get(
    "DEP_KOBO_SESSIONS",
    os.path.join(os.path.dirname(__file__), "..", "..", "kobo", "sessions"))
PROSODY_CSV = os.environ.get("DEP_PROSODY_CSV", "")

# what a difference in features is allowed to do to a prediction
MAX_HAMD_SHIFT = 0.5      # points on a 0-44 scale; observed max was 0.06
MAX_PHQ9_SHIFT = 0.5      # points on a 0-27 scale; observed max was 0.01


def _sessions():
    if not os.path.isdir(SESSIONS):
        return []
    return sorted(d for d in os.listdir(SESSIONS)
                  if d.isdigit()
                  and os.path.exists(os.path.join(SESSIONS, d, f"{d}_AUDIO.wav")))


# ---------------------------------------------------------------- unit level
def test_feature_names_match_embed_dim():
    assert len(FEATURE_NAMES) == EMBED_DIM == 24
    assert len(set(FEATURE_NAMES)) == EMBED_DIM, "duplicate feature name"


def test_feature_order_is_the_shipped_contract():
    """The weights are indexed by this order; a reorder is silent corruption."""
    assert FEATURE_NAMES[0] == "speech_frac_mean"
    assert FEATURE_NAMES[1] == "speech_frac_sd"
    assert FEATURE_NAMES[-2] == "longest_pause_mean"
    assert FEATURE_NAMES[-1] == "longest_pause_sd"


def test_aggregate_handles_no_usable_turns():
    v = aggregate_turns([])
    assert v.shape == (EMBED_DIM,) and not np.any(np.isnan(v))


def test_aggregate_single_turn_gives_zero_sd_not_nan():
    """One turn means the SD is undefined; it must become 0, as in training."""
    rng = np.random.default_rng(0)
    f = turn_features(rng.normal(0, 0.1, 16000 * 3).astype(np.float32))
    assert f is not None
    v = aggregate_turns([f])
    assert not np.any(np.isnan(v)), "nan leaked into the feature vector"
    for i, name in enumerate(FEATURE_NAMES):
        if name.endswith("_sd"):
            assert v[i] == 0.0


def test_silence_yields_no_features():
    assert turn_features(np.zeros(16000 * 2, dtype=np.float32)) is None


# ------------------------------------------------------- training/inference parity
@pytest.mark.skipif(not _sessions() or not os.path.exists(PROSODY_CSV),
                    reason="study data or training prosody table not available")
def test_inference_features_reproduce_training_features():
    """The gate: features computed from the served waveform must produce the
    same predictions as the features the model was fitted on."""
    import pandas as pd
    from src.fusion.model import LinearHead

    train = pd.read_csv(PROSODY_CSV).set_index("participant_id")
    weights = os.path.join(os.path.dirname(__file__), "..",
                           "outputs", "weights", "fusion_head_808.npz")
    if not os.path.exists(weights):
        pytest.skip("808-d weights not built; run scripts/fit_final_model.py")
    head = LinearHead.load(weights)

    pids = [int(p) for p in _sessions() if int(p) in train.index][:5]
    assert pids, "no overlap between sessions on disk and the training table"

    # text and metadata are held fixed so any shift is attributable to prosody
    rng = np.random.default_rng(0)
    text = rng.normal(0, 0.3, 768).astype(np.float64)
    meta = rng.random(16).astype(np.float64)

    for pid in pids:
        want = np.array([float(train.at[pid, n]) for n in FEATURE_NAMES])
        got = run_prosody_pipeline(pid, SESSIONS).astype(np.float64)

        assert got.shape == (EMBED_DIM,)
        assert not np.any(np.isnan(got)), f"nan in prosody for {pid}"

        # loose feature check — catches gross breakage, not quantisation noise
        rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-6)
        assert np.median(rel) < 0.05, (
            f"pid {pid}: median feature error {np.median(rel):.3f} — the "
            f"inference computation has diverged from training, not merely "
            f"quantised differently")

        a = head.forward(np.concatenate([text, want, meta]))
        b = head.forward(np.concatenate([text, got, meta]))
        assert abs(a["hamd"] - b["hamd"]) < MAX_HAMD_SHIFT, (
            f"pid {pid}: HAM-D moves {abs(a['hamd']-b['hamd']):.3f} points "
            f"between training-time and inference-time features")
        assert abs(a["phq9"] - b["phq9"]) < MAX_PHQ9_SHIFT
