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
    for key in ("phq9_pred", "hamd_pred", "phq9_clinician", "hamd_clinician",
                "binary_pred", "risk_flag", "modality_attributions", "narrative",
                "subtype_differential", "evidence", "treatment_suggestions", "patient_summary"):
        assert key in result
    assert result["risk_flag"] is True  # carried through from scale-responses, not recomputed differently
    assert result["phq9_clinician"] == 15  # the clinician-entered total, surfaced alongside the AI estimate
    assert result["hamd_clinician"] == 20
    assert set(result["modality_attributions"]) == {"text", "audio", "metadata"}
    # No ANTHROPIC_API_KEY in the test environment -> all four LLM-based
    # fields are unavailable, not fabricated.
    assert result["subtype_differential"] is None
    assert result["evidence"] is None
    assert result["treatment_suggestions"] is None
    assert result["patient_summary"] is None

    r = client.get(f"/sessions/{session_id}/results")
    assert r.status_code == 200
    assert r.json() == result

    # Note drafting is also LLM-only -> 503 without a key, not a fabricated draft.
    r = client.post(f"/sessions/{session_id}/note-draft")
    assert r.status_code == 503


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


def test_patient_roster_orders_risk_first_and_counts_visits(client):
    """GET /participants powers the clinical roster: risk-flagged patients
    float to the top, and each patient shows their visit count + latest
    scores without the frontend needing per-patient follow-up fetches."""
    stable_patient, _ = _run_full_session(client, phq9_item9=0, hamd_suicide_item=0)
    flagged_patient, _ = _run_full_session(client, phq9_item9=2, hamd_suicide_item=0)
    # A second visit for the stable patient — visit_count should reflect both.
    _run_full_session(client, participant_id=stable_patient)

    r = client.get("/participants")
    assert r.status_code == 200
    roster = r.json()
    ids_in_order = [p["participant_id"] for p in roster]
    assert ids_in_order.index(flagged_patient) < ids_in_order.index(stable_patient)

    stable_entry = next(p for p in roster if p["participant_id"] == stable_patient)
    assert stable_entry["visit_count"] == 2
    assert stable_entry["risk_flag"] is False
    assert stable_entry["phq9_pred"] is not None
    assert "relapse_warning" in stable_entry and "flag" in stable_entry["relapse_warning"]
    assert "risk_trajectory" in stable_entry and "flag" in stable_entry["risk_trajectory"]

    flagged_entry = next(p for p in roster if p["participant_id"] == flagged_patient)
    assert flagged_entry["visit_count"] == 1
    assert flagged_entry["risk_flag"] is True


def test_get_single_participant(client):
    r = client.post("/participants", json=PARTICIPANT_BODY)
    participant_id = r.json()["participant_id"]

    r = client.get(f"/participants/{participant_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["participant_id"] == participant_id
    assert body["visit_count"] == 0
    assert body["last_visit_at"] is None
    assert body["risk_flag"] is None

    assert client.get("/participants/doesnotexist").status_code == 404


def test_subtype_differential_unavailable_without_api_key(monkeypatch):
    """No non-LLM fallback by design (see api/subtype_differential.py's
    docstring) — must return None, not a fabricated differential, when
    ANTHROPIC_API_KEY isn't set."""
    from api.subtype_differential import generate_differential

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert generate_differential("I feel low most days and can't sleep.") is None


def test_subtype_differential_unavailable_for_empty_transcript(monkeypatch):
    from api.subtype_differential import generate_differential

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    assert generate_differential("") is None
    assert generate_differential("   ") is None


def test_subtype_differential_validates_model_output():
    from api.subtype_differential import SUBTYPE_CRITERIA, _validate

    good = {name: {"likelihood": "possible", "rationale": "x"} for name in SUBTYPE_CRITERIA}
    validated = _validate(good)
    assert set(validated) == set(SUBTYPE_CRITERIA)

    with pytest.raises(ValueError):
        _validate({name: {"likelihood": "definitely"} for name in SUBTYPE_CRITERIA})  # bad enum

    with pytest.raises(ValueError):
        _validate({})  # missing entries


# ---- Feature 4/6: rule-based (not LLM) trend detection -------------------

def test_relapse_warning_flags_sustained_worsening():
    from api.trends import relapse_warning

    visits = [
        {"hamd_pred": 5.0}, {"hamd_pred": 8.0}, {"hamd_pred": 12.0},
    ]
    result = relapse_warning(visits)
    assert result["flag"] is True

    visits_improving = [
        {"hamd_pred": 20.0}, {"hamd_pred": 14.0}, {"hamd_pred": 8.0},
    ]
    assert relapse_warning(visits_improving)["flag"] is False

    assert relapse_warning([{"hamd_pred": 5.0}, {"hamd_pred": 6.0}])["flag"] is False  # too few visits


def test_relapse_warning_requires_meaningful_delta():
    from api.trends import relapse_warning

    # Non-improving but only a trivial delta -> should not alarm on noise.
    visits = [{"hamd_pred": 10.0}, {"hamd_pred": 10.5}, {"hamd_pred": 11.0}]
    assert relapse_warning(visits)["flag"] is False


def test_risk_trajectory_flags_repeating_pattern():
    from api.trends import risk_trajectory

    assert risk_trajectory([{"risk_flag": True}, {"risk_flag": False}, {"risk_flag": True}])["flag"] is True
    assert risk_trajectory([{"risk_flag": False}, {"risk_flag": False}, {"risk_flag": True}])["flag"] is False
    assert risk_trajectory([{"risk_flag": None}])["flag"] is False  # no scale responses yet


# ---- Feature 5: treatment-response overlay --------------------------------

def test_treatment_events_crud(client):
    r = client.post("/participants", json=PARTICIPANT_BODY)
    participant_id = r.json()["participant_id"]

    r = client.post(f"/participants/{participant_id}/treatments",
                     json={"event_type": "medication_change", "description": "Started sertraline 50mg"})
    assert r.status_code == 200
    event = r.json()
    assert event["event_type"] == "medication_change"
    assert "event_id" in event and "event_date" in event

    r = client.post(f"/participants/{participant_id}/treatments",
                     json={"event_type": "therapy_session", "description": "First CBT session"})
    assert r.status_code == 200

    r = client.get(f"/participants/{participant_id}/treatments")
    assert r.status_code == 200
    events = r.json()
    assert len(events) == 2
    assert events[0]["description"] == "Started sertraline 50mg"  # chronological


def test_treatment_events_unknown_patient_404s(client):
    assert client.get("/participants/doesnotexist/treatments").status_code == 404
    assert client.post("/participants/doesnotexist/treatments",
                        json={"event_type": "other", "description": "x"}).status_code == 404


# ---- Feature 8: natural-language caseload query ---------------------------

def test_caseload_query_unavailable_without_api_key(client):
    r = client.post("/query", json={"question": "which patients are flagged?"})
    assert r.status_code == 503


# ---- Feature 9: practice-level analytics ----------------------------------

def test_analytics_computes_deterministic_stats(client):
    _run_full_session(client, phq9_item9=0, hamd_suicide_item=0)
    _, flagged_session = _run_full_session(client, phq9_item9=2, hamd_suicide_item=0)

    r = client.get("/analytics")
    assert r.status_code == 200
    stats = r.json()
    assert stats["total_patients"] == 2
    assert stats["total_visits"] == 2
    assert stats["scored_visits"] == 2
    assert stats["referral_flag_rate"] == pytest.approx(0.5)
    assert set(stats["phq9_severity_distribution"]) == {
        "minimal (0-4)", "mild (5-9)", "moderate (10-14)",
        "moderately severe (15-19)", "severe (20-27)",
    }
    assert sum(stats["phq9_severity_distribution"].values()) == 2
    # No ANTHROPIC_API_KEY in the test environment -> narrative unavailable.
    assert stats["narrative"] is None


def test_analytics_handles_empty_practice(client):
    r = client.get("/analytics")
    assert r.status_code == 200
    stats = r.json()
    assert stats["total_patients"] == 0
    assert stats["scored_visits"] == 0
    assert stats["referral_flag_rate"] is None
    assert stats["caseness_rate"] is None


# ---- Features 1/2/7/10: remaining LLM-only modules, unavailable path -----

def test_note_draft_unavailable_without_scored_session(client):
    r = client.post("/participants", json=PARTICIPANT_BODY)
    participant_id = r.json()["participant_id"]
    r = client.post("/sessions", json={"participant_id": participant_id})
    session_id = r.json()["session_id"]

    r = client.post(f"/sessions/{session_id}/note-draft")
    assert r.status_code == 400  # not scored yet, distinct from the 503 "no LLM" case


def test_evidence_treatment_suggestions_patient_summary_unavailable(monkeypatch):
    from api.evidence import generate_evidence
    from api.treatment_suggestions import generate_suggestions
    from api.patient_summary import generate_patient_summary

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert generate_evidence("I feel low most days.", {"text": 0.8, "audio": 0.1, "metadata": 0.1}) is None
    assert generate_evidence("", {"text": 1.0}) is None  # empty transcript, no call needed
    assert generate_suggestions(5.0, 8.0, 6, 10, 0, False) is None
    assert generate_patient_summary(8.0, False) is None
