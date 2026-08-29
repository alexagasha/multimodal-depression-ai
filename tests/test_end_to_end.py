"""
End-to-end smoke tests against synthetic data.
Run: pytest tests/ -v
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "sessions")
LABELS_PATH = os.path.join(DATA_ROOT, "LABELS.csv")


@pytest.fixture(scope="session", autouse=True)
def generate_synthetic_data():
    """Generate synthetic data once for the whole test session."""
    if not os.path.exists(LABELS_PATH):
        import subprocess
        subprocess.run([
            sys.executable,
            os.path.join(os.path.dirname(__file__), "..", "data", "synthetic", "generate_synthetic_data.py"),
            "--n_participants", "5"
        ], check=True)


def test_sync_segments():
    from src.pipelines.sync import build_segments
    segs = build_segments(300, data_root=DATA_ROOT)
    assert len(segs) >= 1, "Should produce at least one segment"
    for s in segs:
        assert "start" in s and "end" in s and "text" in s
        assert s["end"] - s["start"] == pytest.approx(30.0, abs=0.01)


def test_text_pipeline():
    from src.pipelines.sync import build_segments
    from src.pipelines.text_pipeline import BertTextEncoder, run_text_pipeline
    segs = build_segments(300, data_root=DATA_ROOT)
    enc = BertTextEncoder()
    embs = run_text_pipeline(segs, enc)
    assert len(embs) == len(segs)
    assert embs[0].shape == (768,)
    assert embs[0].dtype == np.float32


def test_audio_pipeline():
    from src.pipelines.sync import build_segments
    from src.pipelines.audio_pipeline import Wav2Vec2AudioEncoder, run_audio_pipeline
    segs = build_segments(300, data_root=DATA_ROOT)
    enc = Wav2Vec2AudioEncoder()
    embs = run_audio_pipeline(300, segs, enc, data_root=DATA_ROOT)
    assert len(embs) == len(segs)
    assert embs[0].shape == (768,)
    assert embs[0].dtype == np.float32


def test_metadata_pipeline():
    from src.pipelines.metadata_pipeline import run_metadata_pipeline, EMBED_DIM
    vec = run_metadata_pipeline(300, data_root=DATA_ROOT)
    assert vec.shape == (EMBED_DIM,)
    assert vec.dtype == np.float32


def test_aggregation():
    """Built against the configured acoustic representation, not a fixed 768.

    Pinning this to wav2vec2 would leave it green while the served
    configuration was broken, which is the opposite of what it is for."""
    from src.fusion.aggregate import (build_participant_vector, FUSION_INPUT_DIM,
                                      METADATA_DIM, ACOUSTIC, ACOUSTIC_DIM)
    text_embs = [np.random.randn(768).astype(np.float32) for _ in range(4)]
    meta_vec = np.random.randn(METADATA_DIM).astype(np.float32)
    acoustic = (np.random.randn(ACOUSTIC_DIM).astype(np.float32) if ACOUSTIC == "prosody"
                else [np.random.randn(ACOUSTIC_DIM).astype(np.float32) for _ in range(4)])
    fvec = build_participant_vector(text_embs, acoustic, meta_vec)
    assert fvec.shape == (FUSION_INPUT_DIM,), f"Expected ({FUSION_INPUT_DIM},), got {fvec.shape}"


def test_aggregation_rejects_wrong_acoustic_width():
    """A vector of the wrong provenance must not be silently accepted."""
    import pytest
    from src.fusion.aggregate import (build_participant_vector, METADATA_DIM,
                                      ACOUSTIC_DIM)
    text_embs = [np.random.randn(768).astype(np.float32) for _ in range(2)]
    meta_vec = np.random.randn(METADATA_DIM).astype(np.float32)
    wrong = np.random.randn(ACOUSTIC_DIM + 7).astype(np.float32)
    with pytest.raises(ValueError):
        build_participant_vector(text_embs, wrong, meta_vec)


def test_fusion_head():
    from src.fusion.model import FusionHead, PHQ9_RANGE, HAMD_RANGE
    from src.fusion.aggregate import FUSION_INPUT_DIM
    model = FusionHead(seed=0)
    dummy = np.random.randn(FUSION_INPUT_DIM).astype(np.float32)
    scores = model.forward(dummy)
    assert PHQ9_RANGE[0] <= scores["phq9"] <= PHQ9_RANGE[1], "PHQ-9 out of range"
    assert HAMD_RANGE[0] <= scores["hamd"] <= HAMD_RANGE[1], "HAM-D out of range"
    label = model.predict_label(dummy)
    assert label in (0, 1)


def test_xai_attribution():
    from src.fusion.model import FusionHead
    from src.fusion.aggregate import FUSION_INPUT_DIM
    from src.xai.attribution import explain_participant, MODALITY_SLICES
    model = FusionHead(seed=1)
    vec = np.random.randn(FUSION_INPUT_DIM).astype(np.float32)
    result = explain_participant(vec, model)
    assert "baseline_phq9_pred" in result
    assert "modality_attributions" in result
    # All active modalities should have an attribution score
    for mod in MODALITY_SLICES:
        assert mod in result["modality_attributions"]


def test_full_pipeline_end_to_end():
    from src.fusion.run_pipeline import run_participant
    from src.pipelines.text_pipeline import BertTextEncoder
    from src.pipelines.audio_pipeline import Wav2Vec2AudioEncoder
    from src.fusion.model import FusionHead
    text_enc = BertTextEncoder()
    audio_enc = Wav2Vec2AudioEncoder()
    model = FusionHead()
    result = run_participant(300, text_enc, audio_enc, model, data_root=DATA_ROOT)
    assert result is not None
    assert "phq9_pred" in result
    assert "hamd_pred" in result
    assert "binary_pred" in result
    assert result["n_segments"] >= 1
    print(f"\n[E2E] pid=300 | PHQ-9 pred={result['phq9_pred']} HAM-D pred={result['hamd_pred']} | binary={result['binary_pred']}")


def test_risk_flag_independent_of_model():
    """
    risk_flag must reflect the item-level PHQ-9/HAM-D scores regardless of the
    (untrained, near-random) fusion head's prediction — see src/safety/risk_flag.py.
    """
    from src.fusion.run_pipeline import run_participant
    from src.pipelines.text_pipeline import BertTextEncoder
    from src.pipelines.audio_pipeline import Wav2Vec2AudioEncoder
    from src.fusion.model import FusionHead
    text_enc = BertTextEncoder()
    audio_enc = Wav2Vec2AudioEncoder()
    model = FusionHead()

    elevated = run_participant(300, text_enc, audio_enc, model, data_root=DATA_ROOT,
                                phq9_item9=2, hamd_suicide_item=0)
    assert elevated["risk_flag"] is True

    clear = run_participant(300, text_enc, audio_enc, model, data_root=DATA_ROOT,
                             phq9_item9=0, hamd_suicide_item=0)
    assert clear["risk_flag"] is False

    # No item scores supplied -> risk_flag omitted, not silently False.
    omitted = run_participant(300, text_enc, audio_enc, model, data_root=DATA_ROOT)
    assert "risk_flag" not in omitted


def test_flag_risk_rules():
    from src.safety.risk_flag import flag_risk
    assert flag_risk(0, 0) is False
    assert flag_risk(1, 0) is True
    assert flag_risk(0, 1) is True
    assert flag_risk(None, 2) is True
    with pytest.raises(ValueError):
        flag_risk(None, None)


def test_metrics_harness():
    import pandas as pd
    from src.eval.metrics import compute_metrics
    df = pd.DataFrame({
        "participant_id": [300, 301, 302, 303],
        "phq9_true":  [18,  4,  14,  7],
        "phq9_pred":  [16.2, 5.1, 13.5, 9.3],
        "hamd_true":  [28,  6,  22,  11],
        "hamd_pred":  [26.0, 7.5, 20.4, 13.1],
        "binary_true": [1,   0,   1,   0],
        "binary_pred": [1,   0,   1,   0],
    })
    metrics = compute_metrics(df, verbose=False)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_weighted"] == pytest.approx(1.0)
    assert metrics["phq9_rmse"] < 3.0


def test_noise_functions_change_input():
    from src.eval.robustness import add_audio_noise, corrupt_text
    rng = np.random.default_rng(0)
    wave = (np.random.randn(16000) * 0.1).astype(np.float32)
    noisy = add_audio_noise(wave, snr_db=5, rng=rng)
    assert noisy.shape == wave.shape
    assert not np.allclose(noisy, wave)
    txt = "how are you feeling today"
    assert corrupt_text(txt, rate=0.3, rng=rng) != txt


def test_robustness_harness_runs():
    from src.eval.robustness import evaluate_robustness, MAX_ALLOWED_DECLINE
    report = evaluate_robustness(data_root=DATA_ROOT, audio_snr_levels=(10,),
                                 text_rates=(0.1,), verbose=False)
    assert report["max_allowed_decline"] == MAX_ALLOWED_DECLINE
    assert "gate_passed" in report and "worst_rel_decline" in report
    assert len(report["by_condition"]) == 2


def test_wav_loader(tmp_path):
    from scipy.io import wavfile
    from src.pipelines.audio_pipeline import _load_wav
    sr = 16000
    sig = (np.random.randn(sr) * 5000).astype(np.int16)
    p = tmp_path / "clip.wav"
    wavfile.write(str(p), sr, sig)
    audio, got_sr = _load_wav(str(p))
    assert got_sr == sr
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert np.abs(audio).max() <= 1.0 + 1e-6


def test_edaic_adapter(tmp_path):
    import pandas as pd
    from scipy.io import wavfile
    from data.edaic.adapt_edaic import adapt_session
    from src.pipelines.sync import build_segments
    sess = tmp_path / "999_P"
    sess.mkdir()
    pd.DataFrame({"Start_Time": [1.0, 35.0], "End_Time": [5.0, 40.0],
                  "Text": ["i feel low", "hard to sleep"], "Confidence": [0.9, 0.8]}
                 ).to_csv(sess / "999_Transcript.csv", index=False)
    wavfile.write(str(sess / "999_AUDIO.wav"), 16000,
                  (np.random.randn(16000) * 3000).astype(np.int16))
    out = tmp_path / "sessions"
    adapt_session(str(sess), pid=999, out_dir=str(out), split="test")
    tdf = pd.read_csv(out / "999" / "999_TRANSCRIPT.csv")
    assert {"start_time", "stop_time", "speaker", "value"}.issubset(tdf.columns)
    assert (out / "999" / "999_AUDIO.wav").exists()
    assert (out / "999" / "999_METADATA.json").exists()
    segs = build_segments(999, data_root=str(out))
    assert len(segs) >= 1


def test_fusion_head_save_load(tmp_path):
    from src.fusion.model import FusionHead
    from src.fusion.aggregate import FUSION_INPUT_DIM
    m = FusionHead(seed=3)
    x = np.random.default_rng(1).normal(size=FUSION_INPUT_DIM).astype(np.float32)
    before = m.forward(x)
    p = str(tmp_path / "w.npz")
    m.save(p)
    after = FusionHead.load(p).forward(x)
    assert after["phq9"] == pytest.approx(before["phq9"])
    assert after["hamd"] == pytest.approx(before["hamd"])


def test_train_build_dataset():
    from src.fusion.train import build_dataset
    from src.fusion.aggregate import FUSION_INPUT_DIM
    X, Y, splits, pids = build_dataset(DATA_ROOT, LABELS_PATH, cache_path=None)
    assert X.shape[1] == FUSION_INPUT_DIM
    assert X.shape[0] == Y.shape[0] == len(pids) >= 1
    assert Y.shape[1] == 2


def test_train_smoke(tmp_path):
    pytest.importorskip("torch")  # runs on Colab; skipped where torch is absent
    from src.fusion.train import train, export_numpy
    from src.fusion.model import FusionHead
    from src.fusion.aggregate import FUSION_INPUT_DIM
    rng = np.random.default_rng(0)
    n = 12
    X = rng.normal(size=(n, FUSION_INPUT_DIM)).astype(np.float32)
    proj = X @ rng.normal(size=FUSION_INPUT_DIM)
    proj = (proj - proj.min()) / (np.ptp(proj) + 1e-9)
    Y = np.stack([proj * 27, proj * 44], axis=1).astype(np.float32)
    splits = np.array(["train"] * 8 + ["dev"] * 4)
    net = train(X, Y, splits, epochs=20, verbose=False)
    wpath = str(tmp_path / "head.npz")
    export_numpy(net, wpath)
    out = FusionHead.load(wpath).forward(X[0])
    assert set(out) == {"phq9", "hamd"}
    assert 0.0 <= out["phq9"] <= 27.0 and 0.0 <= out["hamd"] <= 44.0
