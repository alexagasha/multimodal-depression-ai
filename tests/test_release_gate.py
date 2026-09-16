"""
The API fails closed: no severity score from anything but the released model.

No clinical data - weights, manifests and encoders here are all synthetic.
"""
import io
import json
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

import api.main as api_main
from api import readiness
from scripts import tag_release


# ---------------------------------------------------------------- release pin
def _manifest(tmp_path, entries, name="model-2026.09.1"):
    path = tmp_path / "RELEASE.json"
    path.write_text(json.dumps({"release": name, "weights": entries}))
    return str(path)


def test_weights_matching_the_manifest_validate(tmp_path):
    w = tmp_path / "fusion_head.npz"
    w.write_bytes(b"released weights")
    m = _manifest(tmp_path, [{"file": "fusion_head.npz", "sha256": readiness.sha256_file(w)}])
    rel = readiness.load_release(str(w), m)
    assert rel["validated"] and rel["release"] == "model-2026.09.1"


def test_changed_weights_do_not_validate_even_with_the_same_name(tmp_path):
    w = tmp_path / "fusion_head.npz"
    w.write_bytes(b"released weights")
    m = _manifest(tmp_path, [{"file": "fusion_head.npz", "sha256": readiness.sha256_file(w)}])
    w.write_bytes(b"refitted later")
    rel = readiness.load_release(str(w), m)
    assert not rel["validated"]
    assert "do not match release model-2026.09.1" in rel["problem"]


def test_a_byte_identical_copy_elsewhere_still_validates(tmp_path):
    """Matching is by content, so serving the released file via DEP_WEIGHTS works."""
    w = tmp_path / "fusion_head.npz"
    w.write_bytes(b"released weights")
    m = _manifest(tmp_path, [{"file": "fusion_head.npz", "sha256": readiness.sha256_file(w)}])
    copy = tmp_path / "elsewhere.npz"
    copy.write_bytes(b"released weights")
    assert readiness.load_release(str(copy), m)["validated"]


def test_no_manifest_means_never_released(tmp_path):
    w = tmp_path / "fusion_head.npz"
    w.write_bytes(b"x")
    rel = readiness.load_release(str(w), str(tmp_path / "missing.json"))
    assert not rel["validated"] and "no release manifest" in rel["problem"]


def test_missing_weights_file(tmp_path):
    rel = readiness.load_release(str(tmp_path / "nope.npz"), str(tmp_path / "RELEASE.json"))
    assert not rel["validated"] and "no weights file" in rel["problem"]


# ---------------------------------------------------------------- components
def _enc(model):
    return types.SimpleNamespace(model=model, model_name="some-model")


READY_RELEASE = {"validated": True, "release": "model-2026.09.1"}


def _report(**over):
    args = dict(text_enc=_enc(object()), audio_enc=None, acoustic="prosody",
                fusion_problem=None, asr=types.SimpleNamespace(backend="faster-whisper"),
                release=READY_RELEASE)
    args.update(over)
    return readiness.readiness_report(readiness.component_checks(**args))


def test_everything_real_and_released_is_ready(monkeypatch):
    monkeypatch.delenv("DEP_ASR_PROVIDER", raising=False)
    rep = _report()
    assert rep["scoring_ready"] and rep["blockers"] == []


@pytest.mark.parametrize("over, blocker", [
    ({"text_enc": _enc(None)}, "text_encoder"),
    ({"acoustic": "wav2vec2", "audio_enc": _enc(None)}, "acoustic_encoder"),
    ({"fusion_problem": "no trained weights"}, "fusion_head"),
    ({"asr": types.SimpleNamespace(backend="mock")}, "transcriber"),
    ({"release": {"validated": False, "problem": "never released"}}, "release"),
])
def test_each_fallback_blocks_scoring(monkeypatch, over, blocker):
    monkeypatch.delenv("DEP_ASR_PROVIDER", raising=False)
    rep = _report(**over)
    assert not rep["scoring_ready"]
    assert any(b.startswith(blocker + ":") for b in rep["blockers"])


def test_unhonoured_asr_provider_warns_but_does_not_block(monkeypatch):
    """Local Whisper still produces real transcripts; the wrong provider is a
    configuration problem to surface, not a reason to stop scoring."""
    monkeypatch.setenv("DEP_ASR_PROVIDER", "groq")
    rep = _report(asr=types.SimpleNamespace(backend="faster-whisper"))
    assert rep["scoring_ready"]
    assert any("DEP_ASR_PROVIDER=groq" in w for w in rep["warnings"])


def test_override_is_read_at_call_time(monkeypatch):
    monkeypatch.delenv(readiness.ALLOW_ENV, raising=False)
    assert not readiness.allow_unvalidated()
    monkeypatch.setenv(readiness.ALLOW_ENV, "1")
    assert readiness.allow_unvalidated()


# ---------------------------------------------------------------- the API
PARTICIPANT = {"age_band": "26-35", "sex": "female", "marital_status": "married",
               "ethnicity": "bantu", "residence": "urban", "education_level": "secondary",
               "employment_status": "employed", "smartphone": "yes", "site": "butabika"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_main.store, "root", str(tmp_path / "store"))
    monkeypatch.setattr(api_main, "LIVE_DATA_ROOT", str(tmp_path / "sessions"))
    monkeypatch.setattr(api_main._asr, "model", None)
    monkeypatch.setattr(api_main._asr, "backend", "mock")
    monkeypatch.delenv(readiness.ALLOW_ENV, raising=False)
    return TestClient(api_main.app)


def _visit(client, item9=0):
    pid = client.post("/participants", json=PARTICIPANT).json()["participant_id"]
    sid = client.post("/sessions", json={"participant_id": pid}).json()["session_id"]
    client.post(f"/sessions/{sid}/scale-responses", json={
        "phq9_total": 10, "hamd_total": 15, "phq9_item9": item9, "hamd_suicide_item": 0})
    buf = io.BytesIO()
    wavfile.write(buf, 16000, (np.random.randn(16000 * 40) * 3000).astype(np.int16))
    buf.seek(0)
    client.post(f"/sessions/{sid}/audio", files={"file": ("v.wav", buf, "audio/wav")})
    return sid


def test_scoring_is_refused_on_mock_components(client):
    sid = _visit(client)
    r = client.post(f"/sessions/{sid}/score")
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "not running the validated model release" in detail
    assert "transcriber" in detail          # the mocked ASR is named as a reason


def test_safety_path_keeps_working_while_scoring_is_refused(client):
    """Refusing to score must never take the clinical record down with it."""
    sid = _visit(client, item9=2)
    assert client.post(f"/sessions/{sid}/score").status_code == 503
    assert client.get(f"/sessions/{sid}").json()["risk_flag"] is True
    assert client.post(f"/sessions/{sid}/notes",
                       json={"author": "Dr. N", "note_text": "Seen."}).status_code == 200
    assert client.post(f"/sessions/{sid}/risk-assessment", json={
        "assessor": "Dr. N", "ideation": "passive", "intent": False, "plan": False,
        "means_access": False, "prior_attempts": False, "disposition": "urgent_follow_up",
        "clinical_reasoning": "Passive ideation, no plan; strong family support.",
    }).status_code == 200


def test_health_reports_why_scoring_is_blocked(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["scoring_ready"] is False
    assert h["blockers"] and h["allow_unvalidated"] is False


def test_development_override_scores_but_stamps_the_result(client, monkeypatch):
    sid = _visit(client)
    monkeypatch.setenv(readiness.ALLOW_ENV, "1")
    r = client.post(f"/sessions/{sid}/score")
    assert r.status_code == 200
    model = r.json()["model"]
    assert model["validated"] is False
    assert model["unvalidated_reasons"]
    assert model["release"] is None


def test_released_installation_scores_and_records_its_release(client, monkeypatch):
    sid = _visit(client)
    monkeypatch.setattr(api_main, "_RELEASE", {
        "validated": True, "release": "model-2026.09.1", "sha256": "ab" * 32})
    monkeypatch.setattr(api_main, "_readiness", lambda: readiness.readiness_report([
        readiness._check("release", True, "weights match release model-2026.09.1")]))
    r = client.post(f"/sessions/{sid}/score")
    assert r.status_code == 200
    model = r.json()["model"]
    assert model["validated"] is True
    assert model["release"] == "model-2026.09.1"
    assert model["weights_sha256"] == "ab" * 8
    # and it is what gets stored, so the score stays traceable later
    assert client.get(f"/sessions/{sid}/results").json()["model"]["release"] == "model-2026.09.1"


# ---------------------------------------------------------------- tag_release
def _weights(path, threshold=10.5):
    from src.fusion.model import LinearHead
    LinearHead(W=np.zeros((2, 808)), b=np.zeros(2), decision_threshold=threshold,
               meta={"fitted": "2026-09-01T00:00:00", "n_participants": 135,
                     "feature_set": "text(768)+prosody(24)+metadata(16)"}).save(str(path))


def test_manifest_written_by_the_script_validates_the_same_weights(tmp_path):
    w = tmp_path / "fusion_head.npz"
    _weights(w)
    manifest = tag_release.build_manifest("model-2026.09.1", "first field release", [str(w)])
    (tmp_path / "RELEASE.json").write_text(json.dumps(manifest))

    entry = manifest["weights"][0]
    assert entry["feature_dim"] == 808 and entry["n_participants"] == 135
    assert readiness.load_release(str(w), str(tmp_path / "RELEASE.json"))["validated"]
    assert tag_release.verify_weights(manifest, str(tmp_path)) == []

    _weights(w, threshold=9.0)        # refit
    assert tag_release.verify_weights(manifest, str(tmp_path))
    assert not readiness.load_release(str(w), str(tmp_path / "RELEASE.json"))["validated"]


def test_release_names_are_constrained(tmp_path):
    w = tmp_path / "fusion_head.npz"
    _weights(w)
    with pytest.raises(SystemExit):
        tag_release.build_manifest("final-v2-really", "", [str(w)])
