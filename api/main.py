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
import wave
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipelines.sync import build_segments
from src.pipelines.text_pipeline import BertTextEncoder, run_text_pipeline
from src.pipelines.audio_pipeline import Wav2Vec2AudioEncoder, run_audio_pipeline, _load_wav
from src.pipelines.metadata_pipeline import run_metadata_pipeline
from src.pipelines.prosody_pipeline import run_prosody_pipeline
from src.pipelines.asr_pipeline import build_asr, run_asr_pipeline, is_probably_silence
from src.fusion.aggregate import build_participant_vector, FUSION_INPUT_DIM, ACOUSTIC
from src.fusion.model import FusionHead, HAMD_CASENESS_THRESHOLD, load_head
from src.xai.attribution import explain_participant
from src.safety.risk_flag import flag_risk
from api.storage import store
from api.genui import generate_narrative
from api.subtype_differential import generate_differential
from api.trends import relapse_warning, risk_trajectory
from api.evidence import generate_evidence
from api.treatment_suggestions import generate_suggestions
from api.patient_summary import generate_patient_summary
from api.note_draft import draft_note
from api.live_note import draft_live_note
from api.caseload_query import answer_query
from api.analytics_summary import generate_analytics_narrative

LIVE_DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "live", "sessions")
# fusion_head.npz is the served model and tracks DEP_ACOUSTIC's default
# (prosody, 808). The wav2vec2 comparison ships as fusion_head_1552.npz, so
# setting DEP_ACOUSTIC=wav2vec2 alone would load weights of the wrong width and
# fall back to an untrained head — pick the matching file automatically, and
# allow an explicit override for a locally refitted model.
_DEFAULT_WEIGHTS = ("fusion_head.npz" if os.environ.get("DEP_ACOUSTIC", "prosody").strip().lower()
                    != "wav2vec2" else "fusion_head_1552.npz")
WEIGHTS_PATH = os.environ.get("DEP_WEIGHTS") or os.path.join(
    os.path.dirname(__file__), "..", "outputs", "weights", _DEFAULT_WEIGHTS)

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
_asr = build_asr()  # DEP_ASR_PROVIDER: local (default) | openai | groq | deepgram
def _load_fusion_head():
    """Load trained weights, refusing any that do not match the served features.

    A head fitted on a different feature set will happily multiply a vector of
    the wrong length or, worse, the right length assembled from different
    features, and return plausible nonsense. Checking the dimension here is the
    only place that mismatch is cheap to catch. An untrained fallback is loud
    because its predictions are random — see docs/mvp-plan.md."""
    if not os.path.exists(WEIGHTS_PATH):
        print("[model] no trained weights at outputs/weights/fusion_head.npz; "
              "serving an UNTRAINED head. Severity scores are random. "
              "Run scripts/fit_final_model.py.")
        return FusionHead()
    head = load_head(WEIGHTS_PATH)
    dim = getattr(head, "input_dim", None) or int(head.W1.shape[1])
    if dim != FUSION_INPUT_DIM:
        print(f"[model] weights expect {dim} features but this build assembles "
              f"{FUSION_INPUT_DIM}; refusing to use them and serving an UNTRAINED "
              f"head. Severity scores are random until the feature sets agree.")
        return FusionHead()
    meta = getattr(head, "meta", {}) or {}
    print(f"[model] loaded {type(head).__name__}: {meta.get('feature_set', dim)} "
          f"| threshold {head.caseness_threshold:.2f} "
          f"| fitted {meta.get('fitted', 'unknown')}")
    return head


_fusion_head = _load_fusion_head()


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


class TreatmentEventIn(BaseModel):
    event_type: str  # "medication_change" | "therapy_session" | "other"
    description: str


class QueryIn(BaseModel):
    question: str


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
    recent visit date, that visit's risk_flag/scores, and rule-based (not
    LLM) trend flags — powers the patient roster so a clinician sees who
    needs attention without opening each patient individually."""
    visits = [s for s in store.list("sessions") if s["participant_id"] == participant["participant_id"]]
    visits.sort(key=lambda s: s.get("created_at") or "")
    enriched_visits = [_enrich_session(v) for v in visits]
    latest = enriched_visits[-1] if enriched_visits else None
    return {
        **participant,
        "visit_count": len(visits),
        "last_visit_at": latest["created_at"] if latest else None,
        "risk_flag": latest["risk_flag"] if latest else None,
        "phq9_pred": latest["phq9_pred"] if latest else None,
        "hamd_pred": latest["hamd_pred"] if latest else None,
        "relapse_warning": relapse_warning(enriched_visits),
        "risk_trajectory": risk_trajectory(enriched_visits),
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
    _write_transcript_csv(
        os.path.join(session_dir, f"{session_id}_TRANSCRIPT.csv"), transcript_rows
    )

    store.update("sessions", session_id, {"status": "audio_uploaded"})
    return {"duration_sec": duration_sec, "n_transcript_rows": len(transcript_rows)}


# --- Live streaming transcription -------------------------------------------
# The web app posts ~6s WAV chunks while recording rather than one file at the
# end, so the transcript and the AI note build up *during* the interview. Each
# chunk is transcribed on its own and its timestamps are offset by the audio
# already banked, so the accumulated rows stay in the same shape the batch path
# produces. Chunk-boundary words can be clipped — the tradeoff for not
# re-transcribing the whole session on stop, which this hardware can't do
# quickly. The whole-file /audio endpoint above stays as the no-mic fallback.

# Only redraft the running note once this much new transcript has arrived —
# regenerating on every chunk would burn tokens and hit rate limits for a
# note that has barely changed.
LIVE_NOTE_MIN_NEW_CHARS = 350


# The columns sync.load_transcript() expects. Written explicitly so a session
# with no recognised speech still produces a readable (header-only) CSV —
# real ASR returns zero segments for silence or non-speech audio, and
# pd.DataFrame([]).to_csv() would otherwise emit an unparseable empty file.
TRANSCRIPT_COLUMNS = ["start_time", "stop_time", "speaker", "value"]


def _write_transcript_csv(path: str, rows: list) -> None:
    pd.DataFrame(rows, columns=TRANSCRIPT_COLUMNS).to_csv(path, index=False)


def _live_doc(session_id: str) -> dict:
    return store.get("live_transcripts", session_id) or {
        "session_id": session_id,
        "rows": [],
        "duration_sec": 0.0,
        "sample_rate": None,
        "note": None,
        "note_at_chars": 0,
    }


def _transcript_text(rows: list) -> str:
    return " ".join(r["value"] for r in rows if r.get("value"))


@app.post("/sessions/{session_id}/audio/chunk")
async def upload_audio_chunk(session_id: str, file: UploadFile):
    _require_session(session_id)
    session_dir = _session_dir(session_id)
    os.makedirs(session_dir, exist_ok=True)

    raw = await file.read()
    tmp_path = os.path.join(session_dir, f"{session_id}_CHUNK.wav")
    with open(tmp_path, "wb") as f:
        f.write(raw)
    waveform, sr = _load_wav(tmp_path)
    os.remove(tmp_path)

    live = _live_doc(session_id)
    offset = float(live["duration_sec"])

    # Append raw samples so finalize() can assemble the full-session WAV that
    # the audio pipeline needs, without holding the whole take in memory.
    with open(os.path.join(session_dir, f"{session_id}_AUDIO.pcm"), "ab") as f:
        f.write(np.clip(waveform, -1.0, 1.0).astype("<f4").tobytes())

    # Transcribing silence costs as much CPU as transcribing speech (and
    # invites hallucinated text), so skip quiet chunks outright — this is what
    # keeps live transcription comfortably ahead of realtime on CPU.
    rows = [] if is_probably_silence(waveform) else run_asr_pipeline(waveform, sr, _asr)
    for row in rows:
        row["start_time"] = float(row["start_time"]) + offset
        row["stop_time"] = float(row["stop_time"]) + offset

    live["rows"].extend(rows)
    live["duration_sec"] = offset + len(waveform) / float(sr)
    live["sample_rate"] = sr

    text = _transcript_text(live["rows"])
    note_updated = False
    if len(text) - live["note_at_chars"] >= LIVE_NOTE_MIN_NEW_CHARS:
        note = draft_live_note(text)
        # Keep the previous draft on failure rather than blanking the panel.
        if note is not None:
            live["note"] = note
            note_updated = True
        live["note_at_chars"] = len(text)

    store.set("live_transcripts", session_id, live)
    store.update("sessions", session_id, {"status": "recording"})
    return {
        "new_rows": rows,
        "duration_sec": live["duration_sec"],
        "n_rows_total": len(live["rows"]),
        "note": live["note"],
        "note_updated": note_updated,
    }


@app.post("/sessions/{session_id}/audio/finalize")
def finalize_audio(session_id: str):
    _require_session(session_id)
    session_dir = _session_dir(session_id)
    live = _live_doc(session_id)
    pcm_path = os.path.join(session_dir, f"{session_id}_AUDIO.pcm")
    # Only the audio is required — a session where nothing intelligible was
    # said still finalizes (scoring then reports no transcript segments,
    # which is a clearer failure than refusing to close the recording).
    if not os.path.exists(pcm_path):
        raise HTTPException(400, "no streamed audio for this session")

    waveform = np.fromfile(pcm_path, dtype="<f4")
    sr = int(live["sample_rate"] or 16000)
    duration_sec = len(waveform) / float(sr)

    # Assemble the same artefacts the whole-file path writes, so scoring and
    # the audio pipeline are identical regardless of how the audio arrived.
    with wave.open(os.path.join(session_dir, f"{session_id}_AUDIO.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(waveform, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    os.remove(pcm_path)

    with open(os.path.join(session_dir, f"{session_id}_AUDIO.meta.json"), "w") as f:
        json.dump({"sample_rate": sr, "duration_sec": duration_sec}, f)
    _write_transcript_csv(
        os.path.join(session_dir, f"{session_id}_TRANSCRIPT.csv"), live["rows"]
    )

    store.update("sessions", session_id, {"status": "audio_uploaded"})
    return {"duration_sec": duration_sec, "n_transcript_rows": len(live["rows"])}


@app.get("/sessions/{session_id}/live")
def get_live_state(session_id: str):
    """Transcript + running note so far — lets a reloaded page catch up."""
    _require_session(session_id)
    live = _live_doc(session_id)
    return {
        "rows": live["rows"],
        "duration_sec": live["duration_sec"],
        "note": live["note"],
    }


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
    if ACOUSTIC == "prosody":
        # Participant-level already — aggregated across turns by the pipeline,
        # so it must not be pooled again.
        acoustic = run_prosody_pipeline(session_id, data_root=LIVE_DATA_ROOT)
    else:
        acoustic = run_audio_pipeline(session_id, segments, _audio_enc,
                                      data_root=LIVE_DATA_ROOT)
    metadata_vec = run_metadata_pipeline(session_id, data_root=LIVE_DATA_ROOT)
    fusion_vec = build_participant_vector(text_embs, acoustic, metadata_vec)

    scores = _fusion_head.forward(fusion_vec)
    # Calibrated cut point, not the clinical one. A regularised regression
    # shrinks toward the training mean, so at HAM-D >= 7 this classified almost
    # everyone as a case (specificity 0.19) despite ranking them well.
    threshold = _fusion_head.caseness_threshold
    binary_pred = int(scores["hamd"] >= threshold)
    xai = explain_participant(fusion_vec, _fusion_head)
    risk_flag = flag_risk(scale["phq9_item9"], scale["hamd_suicide_item"])

    narrative = generate_narrative(
        scores["phq9"], scores["hamd"], binary_pred,
        xai["modality_attributions"], risk_flag, threshold,
    )
    transcript_text = " ".join(seg["text"] for seg in segments if seg.get("text"))
    subtype_differential = generate_differential(transcript_text)
    evidence = generate_evidence(transcript_text, xai["modality_attributions"])
    treatment_suggestions = generate_suggestions(
        scores["phq9"], scores["hamd"], scale["phq9_total"], scale["hamd_total"],
        binary_pred, risk_flag, subtype_differential,
    )
    patient_summary = generate_patient_summary(scores["hamd"], risk_flag)

    result = {
        "session_id": session_id,
        "phq9_pred": round(float(scores["phq9"]), 3),
        "hamd_pred": round(float(scores["hamd"]), 3),
        # Clinician-administered totals, surfaced alongside the AI estimate
        # rather than replaced by it — the AI runs as a second opinion, not
        # a substitute for the clinician's own PHQ-9/HAM-D scoring.
        "phq9_clinician": scale["phq9_total"],
        "hamd_clinician": scale["hamd_total"],
        # All four below are None when unavailable (no ANTHROPIC_API_KEY) —
        # see each module's docstring for why there's no non-LLM fallback,
        # unlike the scores-only narrative above.
        "subtype_differential": subtype_differential,
        "evidence": evidence,
        "treatment_suggestions": treatment_suggestions,
        "patient_summary": patient_summary,
        "binary_pred": binary_pred,
        "risk_flag": risk_flag,
        "modality_attributions": xai["modality_attributions"],
        # Named prosodic measures driving this prediction, when the served
        # acoustic features are the interpretable ones. Absent under wav2vec2,
        # whose dimensions have no names to give.
        "acoustic_drivers": xai.get("acoustic_drivers"),
        "narrative": narrative,
        # Provenance: which model produced this, and where its decision fell.
        # Without it a stored prediction cannot be traced to the weights that
        # made it, and a later refit silently changes the meaning of old rows.
        "model": {
            "features": ACOUSTIC,
            "input_dim": FUSION_INPUT_DIM,
            "caseness_threshold": round(float(threshold), 3),
            "trained": bool(getattr(_fusion_head, "meta", None)),
            "fitted": (getattr(_fusion_head, "meta", {}) or {}).get("fitted"),
        },
    }
    store.set("predictions", session_id, result)
    store.update("sessions", session_id, {"status": "scored"})
    return result


@app.post("/sessions/{session_id}/note-draft")
def get_note_draft(session_id: str):
    """
    On-demand AI-drafted SOAP note (api/note_draft.py) — computed fresh each
    call, not stored, since a clinician may want to regenerate it after
    adding their own notes first. The clinician edits the draft client-side
    and submits it as a real note via the existing POST .../notes endpoint;
    this endpoint never writes to clinical_notes itself.
    """
    _require_session(session_id)
    prediction = store.get("predictions", session_id)
    if prediction is None:
        raise HTTPException(400, "session not scored yet — POST /sessions/{id}/score first")

    segments = build_segments(session_id, data_root=LIVE_DATA_ROOT)
    transcript_text = " ".join(seg["text"] for seg in segments if seg.get("text"))
    draft = draft_note(
        transcript_text,
        prediction["phq9_pred"], prediction["hamd_pred"],
        prediction["phq9_clinician"], prediction["hamd_clinician"],
        prediction["risk_flag"], prediction.get("subtype_differential"),
    )
    if draft is None:
        raise HTTPException(
            503,
            "Note drafting requires an LLM connection (ANTHROPIC_API_KEY) — "
            "not available in this local demo without a key.",
        )
    return draft


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


@app.post("/participants/{participant_id}/treatments")
def add_treatment_event(participant_id: str, body: TreatmentEventIn):
    """
    Treatment-response overlay data: medication/therapy changes recorded
    against a patient (not a single visit), so they can be plotted on the
    severity trend — "did the SSRI switch actually work" becomes a glance,
    not a chart review. Append-only, same convention as clinical notes.
    """
    if store.get("participants", participant_id) is None:
        raise HTTPException(404, f"participant {participant_id} not found")
    events = store.get("treatment_events", participant_id) or []
    event = {
        "event_id": uuid.uuid4().hex[:10],
        "event_type": body.event_type,
        "description": body.description,
        "event_date": _now_iso(),
    }
    events.append(event)
    store.set("treatment_events", participant_id, events)
    return event


@app.get("/participants/{participant_id}/treatments")
def list_treatment_events(participant_id: str):
    if store.get("participants", participant_id) is None:
        raise HTTPException(404, f"participant {participant_id} not found")
    events = store.get("treatment_events", participant_id) or []
    return sorted(events, key=lambda e: e.get("event_date") or "")


@app.post("/query")
def query_caseload(body: QueryIn):
    """
    Natural-language caseload query (api/caseload_query.py) over the full
    patient roster — "which patients haven't improved in 3 visits."
    """
    roster = [_enrich_participant(p) for p in store.list("participants")]
    result = answer_query(body.question, roster)
    if result is None:
        raise HTTPException(
            503,
            "Caseload queries require an LLM connection (ANTHROPIC_API_KEY) — "
            "not available in this local demo without a key.",
        )
    return result


@app.get("/analytics")
def get_analytics():
    """
    Practice-level analytics: a population view for a clinician managing a
    caseload, not just one patient's chart at a time. Numbers are always
    computed deterministically here; only the narrative on top
    (api/analytics_summary.py) depends on an LLM connection.
    """
    participants = store.list("participants")
    all_sessions = store.list("sessions")
    scored = []
    for s in all_sessions:
        prediction = store.get("predictions", s["session_id"])
        if prediction:
            scored.append(prediction)

    # PHQ-9 severity bands (standard clinical convention: 0-4 minimal,
    # 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe).
    bands = {"minimal (0-4)": 0, "mild (5-9)": 0, "moderate (10-14)": 0,
             "moderately severe (15-19)": 0, "severe (20-27)": 0}
    for p in scored:
        v = p["phq9_pred"]
        if v < 5:
            bands["minimal (0-4)"] += 1
        elif v < 10:
            bands["mild (5-9)"] += 1
        elif v < 15:
            bands["moderate (10-14)"] += 1
        elif v < 20:
            bands["moderately severe (15-19)"] += 1
        else:
            bands["severe (20-27)"] += 1

    n_scored = len(scored)
    stats = {
        "total_patients": len(participants),
        "total_visits": len(all_sessions),
        "scored_visits": n_scored,
        "referral_flag_rate": round(sum(1 for p in scored if p["risk_flag"]) / n_scored, 3) if n_scored else None,
        "caseness_rate": round(sum(1 for p in scored if p["binary_pred"]) / n_scored, 3) if n_scored else None,
        "phq9_severity_distribution": bands,
    }
    stats["narrative"] = generate_analytics_narrative(stats)
    return stats


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
