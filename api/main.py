"""
FastAPI backend — Phase 1 of the system roadmap (see the plan / migration doc).

Wraps the existing scoring pipeline behind HTTP endpoints without touching any
of its code: uploads get staged into the same on-disk session-folder shape
(`{id}_TRANSCRIPT.csv`, `{id}_AUDIO.*`, `{id}_METADATA.json`) that
sync.py/text_pipeline.py/audio_pipeline.py/metadata_pipeline.py already read,
under data/live/sessions/ instead of data/synthetic/sessions/.

Local dev:  uvicorn api.main:app --reload
Docker:     see docker/docker-compose.yml (api service)

api/storage.py is a local JSON stand-in for Firestore (Phase 4 swaps the
implementation only, not the call sites). api/genui.py generates the score
modal's narrative (Claude if ANTHROPIC_API_KEY is set, template otherwise).
"""
import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipelines.sync import build_segments
from src.pipelines.text_pipeline import BertTextEncoder, run_text_pipeline
from src.pipelines.audio_pipeline import Wav2Vec2AudioEncoder, run_audio_pipeline, _load_wav
from src.pipelines.metadata_pipeline import run_metadata_pipeline
from src.pipelines.asr_pipeline import WhisperASR, run_asr_pipeline
from src.fusion.aggregate import build_participant_vector
from src.fusion.model import FusionHead, HAMD_CASENESS_THRESHOLD
from src.xai.attribution import explain_participant
from src.safety.risk_flag import flag_risk
from api.storage import store
from api.genui import generate_narrative
from api.subtype_differential import generate_differential

LIVE_DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "live", "sessions")
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "weights", "fusion_head.npz")

app = FastAPI(title="Depression Detection API", version="0.1.0")

# Local dev only: the static score-modal demo page (web/index.html) is opened
# directly from disk or a plain static server, so it needs CORS to call this
# API. Phase 3's real Next.js app replaces this with a same-origin/proxy
# setup; tighten this before any real deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frozen encoders + fusion head, loaded once. Mock fallback if torch/transformers
# aren't installed (see each module's docstring) — safe for local dev.
_text_enc = BertTextEncoder()
_audio_enc = Wav2Vec2AudioEncoder()
_asr = WhisperASR()
_fusion_head = (
    FusionHead.load(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else FusionHead()
)


class ParticipantIn(BaseModel):
    age_band: str
    sex: str
    marital_status: str
    ethnicity: str
    residence: str
    education_level: str
    employment_status: str
    smartphone: str
    site: str


class SessionIn(BaseModel):
    participant_id: str


class ScaleResponsesIn(BaseModel):
    phq9_total: int
    hamd_total: int
    phq9_item9: int
    hamd_suicide_item: int


class ClinicalNoteIn(BaseModel):
    author: str
    note_text: str


class ReviewIn(BaseModel):
    reviewer: str
    agrees: bool
    adjusted_phq9: Optional[float] = None
    adjusted_hamd: Optional[float] = None
    comment: Optional[str] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_dir(session_id: str) -> str:
    return os.path.join(LIVE_DATA_ROOT, session_id)


def _require_session(session_id: str) -> dict:
    session = store.get("sessions", session_id)
    if session is None:
        raise HTTPException(404, f"session {session_id} not found")
    return session


def _enrich_session(session: dict) -> dict:
    """Join a session record with its prediction, if scored — used by the
    triage-queue dashboard and the participant trend view so the frontend
    doesn't need N+1 fetches."""
    prediction = store.get("predictions", session["session_id"])
    review = store.get("clinician_reviews", session["session_id"])
    return {
        **session,
        "phq9_pred": prediction["phq9_pred"] if prediction else None,
        "hamd_pred": prediction["hamd_pred"] if prediction else None,
        "binary_pred": prediction["binary_pred"] if prediction else None,
        "reviewed": review is not None,
    }


def _triage_sort_key(session: dict):
    # Risk-flagged sessions first, then by HAM-D severity (unscored sessions
    # sort after scored ones within each group), then oldest-first.
    return (
        0 if session.get("risk_flag") else 1,
        -(session["hamd_pred"] if session.get("hamd_pred") is not None else -1),
        session.get("created_at") or "",
    )


def _enrich_participant(participant: dict) -> dict:
    """Join a patient record with their visit history: visit count, most
    recent visit date, and that visit's risk_flag/scores — powers the
    patient roster so a clinician sees who needs attention without opening
    each patient individually."""
    visits = [s for s in store.list("sessions") if s["participant_id"] == participant["participant_id"]]
    visits.sort(key=lambda s: s.get("created_at") or "")
    latest = visits[-1] if visits else None
    latest_enriched = _enrich_session(latest) if latest else None
    return {
        **participant,
        "visit_count": len(visits),
        "last_visit_at": latest["created_at"] if latest else None,
        "risk_flag": latest_enriched["risk_flag"] if latest_enriched else None,
        "phq9_pred": latest_enriched["phq9_pred"] if latest_enriched else None,
        "hamd_pred": latest_enriched["hamd_pred"] if latest_enriched else None,
    }


def _patient_sort_key(patient: dict):
    # Same risk-first logic as _triage_sort_key, applied to each patient's
    # most recent visit. Patients with no visits yet sort last.
    return (
        0 if patient.get("risk_flag") else 1,
        -(patient["hamd_pred"] if patient.get("hamd_pred") is not None else -1),
        patient.get("last_visit_at") is None,
        patient.get("last_visit_at") or "",
    )


@app.post("/participants")
def create_participant(body: ParticipantIn):
    participant_id = uuid.uuid4().hex[:10]
    record = {"participant_id": participant_id, **body.model_dump()}
    store.set("participants", participant_id, record)
    return {"participant_id": participant_id}


@app.get("/participants")
def list_participants():
    """Patient roster: every registered patient enriched with visit count
    and their latest visit's risk_flag/scores, risk-flagged patients first.
    This is the clinical landing view — find or start a patient's next visit
    without already knowing a session link."""
    patients = [_enrich_participant(p) for p in store.list("participants")]
    patients.sort(key=_patient_sort_key)
    return patients


@app.get("/participants/{participant_id}")
def get_participant(participant_id: str):
    participant = store.get("participants", participant_id)
    if participant is None:
        raise HTTPException(404, f"participant {participant_id} not found")
    return _enrich_participant(participant)


@app.post("/sessions")
def create_session(body: SessionIn):
    participant = store.get("participants", body.participant_id)
    if participant is None:
        raise HTTPException(404, f"participant {body.participant_id} not found")

    session_id = uuid.uuid4().hex[:10]
    session_dir = _session_dir(session_id)
    os.makedirs(session_dir, exist_ok=True)

    metadata = {k: v for k, v in participant.items() if k != "participant_id"}
    metadata["participant_id"] = session_id  # pid used by metadata_pipeline is the session id
    with open(os.path.join(session_dir, f"{session_id}_METADATA.json"), "w") as f:
        json.dump(metadata, f)

    record = {
        "session_id": session_id,
        "participant_id": body.participant_id,
        "status": "created",
        "risk_flag": None,
        "created_at": _now_iso(),
    }
    store.set("sessions", session_id, record)
    return {"session_id": session_id}


@app.post("/sessions/{session_id}/scale-responses")
def submit_scale_responses(session_id: str, body: ScaleResponsesIn):
    _require_session(session_id)
    risk_flag = flag_risk(body.phq9_item9, body.hamd_suicide_item)
    store.set("scale_responses", session_id, {"session_id": session_id, **body.model_dump()})
    store.update("sessions", session_id, {"status": "scale_responses_recorded", "risk_flag": risk_flag})
    return {"risk_flag": risk_flag}


@app.post("/sessions/{session_id}/audio")
async def upload_audio(session_id: str, file: UploadFile):
    _require_session(session_id)
    session_dir = _session_dir(session_id)
    os.makedirs(session_dir, exist_ok=True)

    raw = await file.read()
    wav_path = os.path.join(session_dir, f"{session_id}_AUDIO.wav")
    with open(wav_path, "wb") as f:
        f.write(raw)

    waveform, sr = _load_wav(wav_path)
    duration_sec = len(waveform) / float(sr)
    with open(os.path.join(session_dir, f"{session_id}_AUDIO.meta.json"), "w") as f:
        json.dump({"sample_rate": sr, "duration_sec": duration_sec}, f)

    transcript_rows = run_asr_pipeline(waveform, sr, _asr)
    pd.DataFrame(transcript_rows).to_csv(
        os.path.join(session_dir, f"{session_id}_TRANSCRIPT.csv"), index=False
    )

    store.update("sessions", session_id, {"status": "audio_uploaded"})
    return {"duration_sec": duration_sec, "n_transcript_rows": len(transcript_rows)}


@app.post("/sessions/{session_id}/score")
def score_session(session_id: str):
    session = _require_session(session_id)
    scale = store.get("scale_responses", session_id)
    if scale is None:
        raise HTTPException(400, "scale responses not submitted yet")

    transcript_path = os.path.join(_session_dir(session_id), f"{session_id}_TRANSCRIPT.csv")
    if not os.path.exists(transcript_path):
        raise HTTPException(400, "no audio uploaded yet — POST /sessions/{id}/audio first")

    segments = build_segments(session_id, data_root=LIVE_DATA_ROOT)
    if not segments:
        raise HTTPException(400, "no transcript segments — upload audio first")

    text_embs = run_text_pipeline(segments, _text_enc)
    audio_embs = run_audio_pipeline(session_id, segments, _audio_enc, data_root=LIVE_DATA_ROOT)
    metadata_vec = run_metadata_pipeline(session_id, data_root=LIVE_DATA_ROOT)
    fusion_vec = build_participant_vector(text_embs, audio_embs, metadata_vec)

    scores = _fusion_head.forward(fusion_vec)
    binary_pred = int(scores["hamd"] >= HAMD_CASENESS_THRESHOLD)
    xai = explain_participant(fusion_vec, _fusion_head)
    risk_flag = flag_risk(scale["phq9_item9"], scale["hamd_suicide_item"])

    narrative = generate_narrative(
        scores["phq9"], scores["hamd"], binary_pred,
        xai["modality_attributions"], risk_flag, HAMD_CASENESS_THRESHOLD,
    )
    transcript_text = " ".join(seg["text"] for seg in segments if seg.get("text"))
    subtype_differential = generate_differential(transcript_text)

    result = {
        "session_id": session_id,
        "phq9_pred": round(float(scores["phq9"]), 3),
        "hamd_pred": round(float(scores["hamd"]), 3),
        # Clinician-administered totals, surfaced alongside the AI estimate
        # rather than replaced by it — the AI runs as a second opinion, not
        # a substitute for the clinician's own PHQ-9/HAM-D scoring.
        "phq9_clinician": scale["phq9_total"],
        "hamd_clinician": scale["hamd_total"],
        # None when unavailable (no ANTHROPIC_API_KEY) — see
        # api/subtype_differential.py's docstring for why there's no
        # non-LLM fallback here, unlike the narrative above.
        "subtype_differential": subtype_differential,
        "binary_pred": binary_pred,
        "risk_flag": risk_flag,
        "modality_attributions": xai["modality_attributions"],
        "narrative": narrative,
    }
    store.set("predictions", session_id, result)
    store.update("sessions", session_id, {"status": "scored"})
    return result


@app.get("/sessions/{session_id}/results")
def get_results(session_id: str):
    _require_session(session_id)
    result = store.get("predictions", session_id)
    if result is None:
        raise HTTPException(404, "not scored yet — POST /sessions/{id}/score first")
    return result


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Single enriched session — status + prediction (if scored) + review
    flag. Powers the session workspace page."""
    return _enrich_session(_require_session(session_id))


@app.get("/sessions")
def list_sessions():
    """Triage queue: every session enriched with its prediction, risk-flagged
    sessions first, then by severity. Powers the dashboard."""
    sessions = [_enrich_session(s) for s in store.list("sessions")]
    sessions.sort(key=_triage_sort_key)
    return sessions


@app.get("/participants/{participant_id}/sessions")
def list_participant_sessions(participant_id: str):
    """A participant's full session history, chronological — powers the
    longitudinal trend view (repeat PHQ-9/HAM-D administration over time)."""
    if store.get("participants", participant_id) is None:
        raise HTTPException(404, f"participant {participant_id} not found")
    sessions = [
        _enrich_session(s) for s in store.list("sessions")
        if s["participant_id"] == participant_id
    ]
    sessions.sort(key=lambda s: s.get("created_at") or "")
    return sessions


@app.post("/sessions/{session_id}/notes")
def add_clinical_note(session_id: str, body: ClinicalNoteIn):
    """Clinical notes are append-only, matching real clinical documentation
    practice — amend via a new note, never edit/delete history."""
    _require_session(session_id)
    notes = store.get("clinical_notes", session_id) or []
    note = {
        "note_id": uuid.uuid4().hex[:10],
        "author": body.author,
        "note_text": body.note_text,
        "created_at": _now_iso(),
    }
    notes.append(note)
    store.set("clinical_notes", session_id, notes)
    return note


@app.get("/sessions/{session_id}/notes")
def get_clinical_notes(session_id: str):
    _require_session(session_id)
    return store.get("clinical_notes", session_id) or []


@app.post("/sessions/{session_id}/review")
def submit_review(session_id: str, body: ReviewIn):
    """Clinician confirms or adjusts the model's prediction. Doubles as the
    raw data a future ICC clinical-validation pass would need."""
    _require_session(session_id)
    if store.get("predictions", session_id) is None:
        raise HTTPException(400, "session not scored yet — POST /sessions/{id}/score first")
    review = {**body.model_dump(), "session_id": session_id, "reviewed_at": _now_iso()}
    store.set("clinician_reviews", session_id, review)
    return review


@app.get("/sessions/{session_id}/review")
def get_review(session_id: str):
    _require_session(session_id)
    review = store.get("clinician_reviews", session_id)
    if review is None:
        raise HTTPException(404, "not reviewed yet")
    return review


@app.get("/health")
def health():
    return {"status": "ok"}
