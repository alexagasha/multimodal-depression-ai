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


def test_score_without_audio_rejected_cleanly(client):
    """Regression test: scoring before any audio upload must return a clean
    400, not crash with an unhandled FileNotFoundError (build_segments reads
    a transcript file that doesn't exist yet if audio was never uploaded)."""
    r = client.post("/participants", json=PARTICIPANT_BODY)
    participant_id = r.json()["participant_id"]
    r = client.post("/sessions", json={"participant_id": participant_id})
    session_id = r.json()["session_id"]

    client.post(f"/sessions/{session_id}/scale-responses", json={
        "phq9_total": 10, "hamd_total": 15, "phq9_item9": 0, "hamd_suicide_item": 0,
    })

    r = client.post(f"/sessions/{session_id}/score")
    assert r.status_code == 400
    assert "audio" in r.json()["detail"].lower()


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


def _run_full_session(client, participant_id=None, phq9_item9=0, hamd_suicide_item=0):
    """Helper: create (or reuse) a participant, run a session through to scored."""
    if participant_id is None:
        r = client.post("/participants", json=PARTICIPANT_BODY)
        participant_id = r.json()["participant_id"]
    r = client.post("/sessions", json={"participant_id": participant_id})
    session_id = r.json()["session_id"]
    client.post(f"/sessions/{session_id}/scale-responses", json={
        "phq9_total": 10, "hamd_total": 15,
        "phq9_item9": phq9_item9, "hamd_suicide_item": hamd_suicide_item,
    })
    client.post(f"/sessions/{session_id}/audio",
                files={"file": ("session.wav", _wav_bytes(), "audio/wav")})
    client.post(f"/sessions/{session_id}/score")
    return participant_id, session_id


def test_clinical_notes_are_append_only_and_ordered(client):
    _, session_id = _run_full_session(client)

    r = client.post(f"/sessions/{session_id}/notes",
                     json={"author": "Dr. Nabirye", "note_text": "Initial impression: flat affect."})
    assert r.status_code == 200
    first_note = r.json()
    assert {"note_id", "author", "note_text", "created_at"}.issubset(first_note)

    r = client.post(f"/sessions/{session_id}/notes",
                     json={"author": "Dr. Nabirye", "note_text": "Follow-up: referred to counseling."})
    assert r.status_code == 200

    r = client.get(f"/sessions/{session_id}/notes")
    assert r.status_code == 200
    notes = r.json()
    assert len(notes) == 2
    assert notes[0]["note_id"] != notes[1]["note_id"]
    assert notes[0]["note_text"] == "Initial impression: flat affect."


def test_notes_unknown_session_404s(client):
    assert client.get("/sessions/doesnotexist/notes").status_code == 404
    assert client.post("/sessions/doesnotexist/notes",
                        json={"author": "x", "note_text": "y"}).status_code == 404


def test_review_workflow(client):
    _, session_id = _run_full_session(client)

    r = client.post(f"/sessions/{session_id}/review", json={
        "reviewer": "Dr. Okello", "agrees": False,
        "adjusted_phq9": 12.0, "adjusted_hamd": 18.0, "comment": "Clinical interview suggests higher severity.",
    })
    assert r.status_code == 200
    assert r.json()["agrees"] is False

    r = client.get(f"/sessions/{session_id}/review")
    assert r.status_code == 200
    assert r.json()["reviewer"] == "Dr. Okello"

    # Reflected in the enriched session list.
    r = client.get("/sessions")
    matching = [s for s in r.json() if s["session_id"] == session_id]
    assert matching and matching[0]["reviewed"] is True


def test_review_requires_scoring_first(client):
    r = client.post("/participants", json=PARTICIPANT_BODY)
    participant_id = r.json()["participant_id"]
    r = client.post("/sessions", json={"participant_id": participant_id})
    session_id = r.json()["session_id"]

    r = client.post(f"/sessions/{session_id}/review",
                     json={"reviewer": "Dr. Okello", "agrees": True})
    assert r.status_code == 400

    assert client.get(f"/sessions/{session_id}/review").status_code == 404


def test_list_sessions_triage_orders_risk_first(client):
    _, low_risk_session = _run_full_session(client, phq9_item9=0, hamd_suicide_item=0)
    _, flagged_session = _run_full_session(client, phq9_item9=2, hamd_suicide_item=0)

    r = client.get("/sessions")
    assert r.status_code == 200
    sessions = r.json()
    ids_in_order = [s["session_id"] for s in sessions]
    assert ids_in_order.index(flagged_session) < ids_in_order.index(low_risk_session)


def test_participant_sessions_trend(client):
    participant_id, session_1 = _run_full_session(client)
    _, session_2 = _run_full_session(client, participant_id=participant_id)

    r = client.get(f"/participants/{participant_id}/sessions")
    assert r.status_code == 200
    sessions = r.json()
    assert [s["session_id"] for s in sessions] == [session_1, session_2]  # chronological
    for s in sessions:
        assert s["phq9_pred"] is not None and s["hamd_pred"] is not None


def test_unknown_participant_sessions_404s(client):
    assert client.get("/participants/doesnotexist/sessions").status_code == 404


def test_get_single_session(client):
    _, session_id = _run_full_session(client, phq9_item9=1)
    r = client.get(f"/sessions/{session_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == session_id
    assert body["risk_flag"] is True
    assert body["phq9_pred"] is not None

    assert client.get("/sessions/doesnotexist").status_code == 404
