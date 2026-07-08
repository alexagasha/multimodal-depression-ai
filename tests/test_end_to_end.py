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
    from src.pipelines.metadata_pipeline import run_metadata_pipeline
    vec = run_metadata_pipeline(300, data_root=DATA_ROOT)
    assert vec.shape == (3,)
    assert vec.dtype == np.float32


def test_aggregation():
    from src.fusion.aggregate import build_participant_vector, FUSION_INPUT_DIM
    text_embs = [np.random.randn(768).astype(np.float32) for _ in range(4)]
    audio_embs = [np.random.randn(768).astype(np.float32) for _ in range(4)]
    meta_vec = np.random.randn(3).astype(np.float32)
    fvec = build_participant_vector(text_embs, audio_embs, meta_vec)
    assert fvec.shape == (FUSION_INPUT_DIM,), f"Expected ({FUSION_INPUT_DIM},), got {fvec.shape}"


def test_fusion_head():
    from src.fusion.model import FusionHead, DEPRESSION_THRESHOLD
    from src.fusion.aggregate import FUSION_INPUT_DIM
    model = FusionHead(seed=0)
    dummy = np.random.randn(FUSION_INPUT_DIM).astype(np.float32)
    score = model.forward(dummy)
    assert 0.0 <= score <= 24.0, "PHQ-8 prediction must be in [0, 24]"
    label = model.predict_label(dummy)
    assert label in (0, 1)


def test_xai_attribution():
    from src.fusion.model import FusionHead
    from src.fusion.aggregate import FUSION_INPUT_DIM
    from src.xai.attribution import explain_participant, MODALITY_SLICES
    model = FusionHead(seed=1)
    vec = np.random.randn(FUSION_INPUT_DIM).astype(np.float32)
    result = explain_participant(vec, model)
    assert "baseline_phq8_pred" in result
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
    assert "phq8_pred" in result
    assert "binary_pred" in result
    assert result["n_segments"] >= 1
    print(f"\n[E2E] pid=300 | PHQ-8 pred={result['phq8_pred']} | binary={result['binary_pred']}")


def test_metrics_harness():
    import pandas as pd
    from src.eval.metrics import compute_metrics
    df = pd.DataFrame({
        "participant_id": [300, 301, 302, 303],
        "phq8_true":  [15,  3,  12,  6],
        "phq8_pred":  [13.2, 4.1, 11.5, 8.3],
        "binary_true": [1,   0,   1,   0],
        "binary_pred": [1,   0,   1,   0],
    })
    metrics = compute_metrics(df, verbose=False)
    assert metrics["f1_weighted"] == pytest.approx(1.0)
    assert metrics["rmse"] < 3.0
