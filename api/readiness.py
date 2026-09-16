"""
Fail closed: refuse to produce a severity score from anything but the real,
released model.

Every ML component in this system degrades gracefully, on purpose. The text
and audio encoders fall back to deterministic mock embeddings when torch or
transformers are missing; the fusion head falls back to untrained weights when
its file is absent or the wrong width; the transcriber falls back to
placeholder text when no speech-recognition backend loads. That is what lets
the test suite and a laptop demo run without a GPU stack.

On a field machine the same behaviour is dangerous. A mock embedding is still a
well-formed vector, the vector still produces a plausible score, and nothing on
the screen distinguishes it from a real one. So scoring consults this module on
every request and refuses (HTTP 503) unless every blocking check passes:

    text_encoder      the real text model loaded
    acoustic_encoder  the real audio model loaded, when one is served
    fusion_head       trained weights loaded, with matching dimensions
    transcriber       a real speech-recognition backend, not placeholder text
    release           the weights on disk are byte-identical to a tagged
                      release in outputs/weights/RELEASE.json

Only scoring fails closed. Clinical notes, the risk assessment, scale responses
and the referral flag keep working, because the safety path must never depend
on the model (src/safety/risk_flag.py).

DEP_ALLOW_UNVALIDATED=1 lets a development machine score anyway. Every result
produced that way is stamped validated=false with the reasons, and the
interface says so. It is an explicit, visible override, never a silent default.
"""
import hashlib
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_MANIFEST = os.path.join(REPO, "outputs", "weights", "RELEASE.json")
ALLOW_ENV = "DEP_ALLOW_UNVALIDATED"


def allow_unvalidated() -> bool:
    """Read on every call, not at import, so it cannot be stale."""
    return os.environ.get(ALLOW_ENV, "").strip().lower() in ("1", "true", "yes")


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_release(weights_path, manifest_path=DEFAULT_MANIFEST) -> dict:
    """Which tagged release the served weights belong to, matched by content.

    Matching is on SHA-256, not filename, so a byte-identical copy served via
    DEP_WEIGHTS still validates while a refitted file with the familiar name
    does not. Returns validated=False with a plain-language problem otherwise.
    """
    out = {"validated": False, "release": None, "problem": None,
           "weights_path": weights_path, "sha256": None, "entry": None}
    if not os.path.exists(weights_path):
        out["problem"] = f"no weights file at {weights_path}"
        return out
    out["sha256"] = sha256_file(weights_path)
    if not os.path.exists(manifest_path):
        out["problem"] = ("no release manifest (outputs/weights/RELEASE.json), so these "
                          "weights have never been released for use")
        return out
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        out["problem"] = f"release manifest is unreadable ({type(e).__name__})"
        return out

    name = manifest.get("release")
    match = next((w for w in manifest.get("weights", [])
                  if w.get("sha256") == out["sha256"]), None)
    if match is None:
        out["problem"] = (f"the weights being served do not match release {name} - "
                          f"they were changed or refitted after it was cut "
                          f"(sha256 {out['sha256'][:16]})")
        return out
    out.update(validated=True, release=name, entry=match)
    return out


def _check(name, ok, detail, blocking=True):
    return {"name": name, "ok": bool(ok), "blocking": blocking, "detail": detail}


def component_checks(text_enc, audio_enc, acoustic, fusion_problem, asr, release) -> list:
    checks = []

    text_name = getattr(text_enc, "model_name", "text model")
    text_ok = getattr(text_enc, "model", None) is not None
    checks.append(_check(
        "text_encoder", text_ok,
        f"text model {text_name} loaded" if text_ok else
        f"text model {text_name} did not load, so text features would be mock "
        f"embeddings - install torch and transformers"))

    if acoustic == "wav2vec2":
        audio_name = getattr(audio_enc, "model_name", "audio model")
        audio_ok = audio_enc is not None and getattr(audio_enc, "model", None) is not None
        checks.append(_check(
            "acoustic_encoder", audio_ok,
            f"audio model {audio_name} loaded" if audio_ok else
            f"audio model {audio_name} did not load, so acoustic features would be "
            f"mock embeddings"))
    else:
        checks.append(_check(
            "acoustic_encoder", True,
            "prosodic features are computed from the waveform; no model to load"))

    checks.append(_check(
        "fusion_head", fusion_problem is None,
        "trained weights loaded with matching dimensions" if fusion_problem is None
        else f"{fusion_problem} - the head is untrained and its scores are random"))

    backend = getattr(asr, "backend", "unknown")
    checks.append(_check(
        "transcriber", backend != "mock",
        f"transcription via {backend}" if backend != "mock" else
        "no speech-recognition backend loaded, so transcripts would be placeholder text"))

    requested = os.environ.get("DEP_ASR_PROVIDER", "local").strip().lower()
    if requested != "local":
        checks.append(_check(
            "transcriber_provider", backend == requested,
            f"hosted transcription via {requested}" if backend == requested else
            f"DEP_ASR_PROVIDER={requested} was requested but {backend} is in use - "
            f"check that provider's API key",
            blocking=False))

    checks.append(_check(
        "release", release.get("validated"),
        f"weights match release {release.get('release')}" if release.get("validated")
        else release.get("problem") or "weights are not part of a release"))
    return checks


def readiness_report(checks) -> dict:
    blockers = [f"{c['name']}: {c['detail']}" for c in checks if c["blocking"] and not c["ok"]]
    warnings = [f"{c['name']}: {c['detail']}" for c in checks if not c["blocking"] and not c["ok"]]
    return {"scoring_ready": not blockers, "blockers": blockers, "warnings": warnings,
            "allow_unvalidated": allow_unvalidated(), "checks": checks}
