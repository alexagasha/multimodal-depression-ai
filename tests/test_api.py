"""
End-to-end test of the local FastAPI backend (Phase 1 of the system roadmap):
create participant -> session -> scale responses -> audio upload (ASR mock) ->
score -> results. Exercises the exact HTTP surface the score-modal web page
(Phase 3) calls.

Each test gets an isolated store/session-data root via tmp_path, so tests
don't pollute (or depend on) the real data/live/ directory.
"""
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

import api.main as api_main
from api.storage import Store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.store, "root", str(tmp_path / "store"))
    monkeypatch.setattr(api_main, "LIVE_DATA_ROOT", str(tmp_path / "sessions"))
    return TestClient(api_main.app)


PARTICIPANT_BODY = {
    "age_band": "26-35", "sex": "female", "marital_status": "married",
    "ethnicity": "bantu", "residence": "urban", "education_level": "secondary",
    "employment_status": "employed", "smartphone": "yes", "site": "butabika",
}


def _wav_bytes(duration_sec=40, sr=16000):
    import io
    wave = (np.random.randn(int(sr * duration_sec)) * 3000).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, wave)
    buf.seek(0)
    return buf


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_full_session_flow(client):
    r = client.post("/participants", json=PARTICIPANT_BODY)
    assert r.status_code == 200
    participant_id = r.json()["participant_id"]

    r = client.post("/sessions", json={"participant_id": participant_id})
    assert r.status_code == 200
    session_id = r.json()["session_id"]

    r = client.post(f"/sessions/{session_id}/scale-responses", json={
        "phq9_total": 15, "hamd_total": 20, "phq9_item9": 2, "hamd_suicide_item": 0,
    })
    assert r.status_code == 200
    assert r.json()["risk_flag"] is True  # phq9_item9=2 triggers it

    r = client.post(f"/sessions/{session_id}/audio",
                     files={"file": ("session.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200
    assert r.json()["n_transcript_rows"] >= 1

    r = client.post(f"/sessions/{session_id}/score")
    assert r.status_code == 200
    result = r.json()
    for key in ("phq9_pred", "hamd_pred", "binary_pred", "risk_flag",
                "modality_attributions", "narrative"):
        assert key in result
    assert result["risk_flag"] is True  # carried through from scale-responses, not recomputed differently
    assert set(result["modality_attributions"]) == {"text", "audio", "metadata"}

    r = client.get(f"/sessions/{session_id}/results")
    assert r.status_code == 200
    assert r.json() == result


def test_score_without_scale_responses_rejected(client):
    r = client.post("/participants", json=PARTICIPANT_BODY)
    participant_id = r.json()["participant_id"]
    r = client.post("/sessions", json={"participant_id": participant_id})
    session_id = r.json()["session_id"]

    r = client.post(f"/sessions/{session_id}/audio",
                     files={"file": ("session.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 200

    r = client.post(f"/sessions/{session_id}/score")
    assert r.status_code == 400


def test_unknown_session_404s(client):
    assert client.post("/sessions/doesnotexist/scale-responses", json={
        "phq9_total": 0, "hamd_total": 0, "phq9_item9": 0, "hamd_suicide_item": 0,
    }).status_code == 404
    assert client.get("/sessions/doesnotexist/results").status_code == 404


def test_clear_risk_flag_when_items_are_zero(client):
    r = client.post("/participants", json=PARTICIPANT_BODY)
    participant_id = r.json()["participant_id"]
    r = client.post("/sessions", json={"participant_id": participant_id})
    session_id = r.json()["session_id"]

    r = client.post(f"/sessions/{session_id}/scale-responses", json={
        "phq9_total": 3, "hamd_total": 4, "phq9_item9": 0, "hamd_suicide_item": 0,
    })
    assert r.json()["risk_flag"] is False
