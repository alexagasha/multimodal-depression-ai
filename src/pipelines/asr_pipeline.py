"""
ASR pipeline: raw session audio -> TRANSCRIPT.csv-shaped rows.

Bridges live-recorded audio (web app) into the existing pipeline, which has
always assumed a transcript already exists (previously produced by "a local
firm" per methodology.txt §3.6). Live sessions need transcription in-app.

Same real-model-with-mock-fallback convention as text_pipeline.BertTextEncoder
and audio_pipeline.Wav2Vec2AudioEncoder: loads real Whisper when available,
otherwise falls back to a deterministic mock so the API/tests run without the
(large) whisper dependency installed.

Supports both whole-session transcription (the original batch path) and short
streaming chunks (the web app's live captions, which post ~6s of audio at a
time — see api/main.py's /audio/chunk endpoint). Speaker attribution MVP: the
caller (web app) supplies a speaker label per chunk (push-to-talk toggle)
rather than this module doing diarization; single-speaker audio defaults to
"Participant".

Backend preference: faster-whisper (CTranslate2) over openai-whisper. On the
CPU-only target hardware faster-whisper is several times quicker and, unlike
openai-whisper, needs no torch install at all — which is what makes live
chunked transcription feasible without a GPU. Falls back to openai-whisper if
that's what's installed, then to a deterministic mock so the API/tests run on
a box with no ASR deps at all.

Provider is selected by DEP_ASR_PROVIDER. "local" runs faster-whisper on this
machine; the hosted providers in CloudASR.CONFIG exist because small local
models handle accented clinical English poorly, and no amount of CPU tuning
fixes that on a GPU-less laptop. All providers return the same
[{"start", "end", "text"}] contract, so nothing downstream changes when you
switch — which also means you can A/B them on the same recording.
"""
import io
import os
import wave

import numpy as np

from src.pipelines.sync import SEGMENT_LEN_SEC

# Default ASR model — override with the DEP_ASR_MODEL env var.
# base.en (English-only) is the accuracy/speed sweet spot for real-time
# chunked transcription on a CPU-only machine; the project is English-only, so
# the multilingual checkpoints cost speed and accuracy for nothing. Drop to
# tiny.en if chunks can't keep up with the recording.
DEFAULT_ASR_MODEL = "base.en"
# int8 quantization ~halves CPU time again at a small accuracy cost.
DEFAULT_COMPUTE_TYPE = "int8"
# Greedy decoding (1) is ~2x quicker than the usual beam search but does cost
# accuracy. Raise via DEP_ASR_BEAM_SIZE on hardware with headroom — at beam 5
# this CPU falls behind realtime, so live chunking would build a backlog.
DEFAULT_BEAM_SIZE = 1

def wav_bytes(waveform: np.ndarray, sample_rate: int) -> bytes:
    """float32 mono samples -> in-memory 16-bit WAV, for the cloud providers."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes((np.clip(waveform, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return buf.getvalue()

# Steers spelling/vocabulary toward the clinical register instead of the
# general-web English the model was trained on. Whisper conditions on this as
# if it were preceding speech, so it measurably improves recognition of the
# terms this app actually cares about.
CLINICAL_PROMPT = (
    "Clinical psychiatric interview covering depression symptoms: PHQ-9, HAM-D, "
    "low mood, anhedanoia, insomnia, hypersomnia, appetite, fatigue, "
    "concentration, hopelessness, worthlessness, guilt, suicidal ideation, "
    "self-harm, anxiety, panic, sertraline, fluoxetine, amitriptyline, "
    "counselling, psychotherapy, referral."
)

# Whisper reliably invents these when fed silence or noise. They are never
# meaningful clinical content, so they are dropped outright rather than
# entering a patient's transcript.
HALLUCINATION_PHRASES = frozenset(
    [
        "thank you.",
        "thank you",
        "thanks for watching!",
        "thanks for watching.",
        "please subscribe",
        "you",
        "bye.",
        "bye bye.",
        ".",
        "...",
        "[blank_audio]",
        "(upbeat music)",
        "subtitles by the amara.org community",
    ]
)


def _is_hallucination(text: str) -> bool:
    return text.strip().lower() in HALLUCINATION_PHRASES


class WhisperASR:
    """
    Frozen transcription model. transcribe() contract:
    (waveform: np.ndarray, sample_rate: int) -> list[{"start", "end", "text"}].

    Loads faster-whisper when available, else openai-whisper, else falls back
    to a deterministic mock transcript so the pipeline/tests run on a box with
    no ASR deps installed.
    """

    def __init__(self, model_name=None):
        self.model = None
        self.backend = "mock"
        self.use_vad = False
        self.model_name = model_name or os.environ.get("DEP_ASR_MODEL", DEFAULT_ASR_MODEL)
        compute_type = os.environ.get("DEP_ASR_COMPUTE_TYPE", DEFAULT_COMPUTE_TYPE)
        self.beam_size = int(os.environ.get("DEP_ASR_BEAM_SIZE", DEFAULT_BEAM_SIZE))
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.model_name, device="cpu", compute_type=compute_type)
            self.backend = "faster-whisper"
            # faster-whisper's VAD runs Silero through onnxruntime, which
            # doesn't load everywhere (older CPUs / missing runtime DLLs).
            # Probe once rather than letting every transcribe() raise; when
            # it's unavailable, callers gate on signal energy instead
            # (see is_probably_silence).
            try:
                import onnxruntime  # noqa: F401
                self.use_vad = True
            except Exception as vad_err:
                print(f"[asr] VAD unavailable ({type(vad_err).__name__}); transcribing without it.")
        except Exception as fw_err:
            try:
                import whisper
                # openai-whisper has no ".en"-suffixed multilingual aliases for
                # every size, but the plain names always exist.
                self.model = whisper.load_model(self.model_name)
                self.backend = "openai-whisper"
            except Exception as w_err:
                self.model = None
                print(
                    f"[asr] no real ASR backend available "
                    f"(faster-whisper: {type(fw_err).__name__}, whisper: {type(w_err).__name__}); "
                    f"using mock transcription."
                )
        if self.model is not None:
            print(f"[asr] using {self.backend} model={self.model_name}")

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> list:
        if waveform.size == 0:
            return []
        if self.model is None:
            return self._mock_transcribe(waveform, sample_rate)
        if self.backend == "faster-whisper":
            return self._transcribe_faster(waveform)
        result = self.model.transcribe(waveform, fp16=False)
        return [
            {"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip()}
            for seg in result.get("segments", [])
        ]

    def _transcribe_faster(self, waveform: np.ndarray) -> list:
        # faster-whisper expects float32 mono at 16kHz, which is what
        # _load_wav already produces.
        segments, _info = self.model.transcribe(
            waveform.astype(np.float32),
            beam_size=self.beam_size,
            vad_filter=self.use_vad,
            initial_prompt=CLINICAL_PROMPT,
            # Chunks are transcribed independently, so carrying decoder state
            # across them has nothing valid to condition on and sends Whisper
            # into repetition loops. Off is correct for streaming.
            condition_on_previous_text=False,
            # Hallucination guards: drop a segment when the model itself
            # reports it was probably silence, when its average token
            # confidence is poor, or when the text is degenerately repetitive.
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )
        rows = []
        for seg in segments:
            text = seg.text.strip()
            if not text or _is_hallucination(text):
                continue
            # Belt and braces: faster-whisper still emits some segments whose
            # own no_speech_prob says there was nothing there.
            if getattr(seg, "no_speech_prob", 0.0) > 0.8:
                continue
            rows.append({"start": float(seg.start), "end": float(seg.end), "text": text})
        return rows

    @staticmethod
    def _mock_transcribe(waveform: np.ndarray, sample_rate: int) -> list:
        """Deterministic placeholder: one segment per SEGMENT_LEN_SEC window."""
        duration = len(waveform) / float(sample_rate)
        n_windows = max(1, int(duration // SEGMENT_LEN_SEC) + 1)
        return [
            {
                "start": i * SEGMENT_LEN_SEC,
                "end": min((i + 1) * SEGMENT_LEN_SEC, duration),
                "text": f"[mock transcript segment {i}]",
            }
            for i in range(n_windows)
        ]


class CloudASR:
    """
    Hosted ASR, same transcribe() contract as WhisperASR.

    Exists because small local models handle accented clinical English poorly
    and no CPU tuning fixes that on a GPU-less machine. Each provider returns
    segment-level timings, which the streaming path needs to offset onto
    absolute session time.

    Every provider here sends interview audio off the machine — that is the
    trade being made, and it needs to be reflected in participant consent.
    """

    #: provider -> (env var holding the API key, default model)
    CONFIG = {
        "openai": ("OPENAI_API_KEY", "whisper-1"),
        "groq": ("GROQ_API_KEY", "whisper-large-v3-turbo"),
        "deepgram": ("DEEPGRAM_API_KEY", "nova-3"),
    }

    def __init__(self, provider: str, model_name=None):
        self.backend = provider
        self.provider = provider
        key_env, default_model = self.CONFIG[provider]
        self.api_key = os.environ.get(key_env)
        self.model_name = model_name or os.environ.get("DEP_ASR_MODEL") or default_model
        self.timeout = float(os.environ.get("DEP_ASR_TIMEOUT", "60"))
        if not self.api_key:
            raise RuntimeError(f"{key_env} is not set")
        print(f"[asr] using {provider} model={self.model_name}")

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> list:
        if waveform.size == 0:
            return []
        import httpx

        audio = wav_bytes(waveform, sample_rate)
        try:
            if self.provider == "deepgram":
                rows = self._deepgram(httpx, audio)
            else:
                rows = self._openai_compatible(httpx, audio)
        except Exception as e:
            # A failed chunk costs a few seconds of captions, not the
            # recording — the audio is still banked locally either way.
            print(f"[asr] {self.provider} request failed ({type(e).__name__}); skipping chunk.")
            return []
        return [r for r in rows if r["text"] and not _is_hallucination(r["text"])]

    def _openai_compatible(self, httpx, audio: bytes) -> list:
        """OpenAI and Groq share the /audio/transcriptions request shape."""
        base = (
            "https://api.openai.com/v1"
            if self.provider == "openai"
            else "https://api.groq.com/openai/v1"
        )
        resp = httpx.post(
            f"{base}/audio/transcriptions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": ("chunk.wav", audio, "audio/wav")},
            data={
                "model": self.model_name,
                # verbose_json is what carries per-segment start/end times.
                "response_format": "verbose_json",
                "language": "en",
                "prompt": CLINICAL_PROMPT,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        segments = body.get("segments")
        if not segments:
            # Some models answer with plain text only; keep it as one segment
            # rather than dropping the speech.
            text = (body.get("text") or "").strip()
            return [{"start": 0.0, "end": 0.0, "text": text}] if text else []
        return [
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": (s.get("text") or "").strip(),
            }
            for s in segments
        ]

    def _deepgram(self, httpx, audio: bytes) -> list:
        resp = httpx.post(
            "https://api.deepgram.com/v1/listen",
            headers={
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav",
            },
            params={
                "model": self.model_name,
                "language": "en",
                "smart_format": "true",
                "punctuate": "true",
                # Utterance-level output is the closest match to Whisper's
                # segments, and carries the timings the segmenter needs.
                "utterances": "true",
            },
            content=audio,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        utterances = body.get("results", {}).get("utterances") or []
        if utterances:
            return [
                {
                    "start": float(u.get("start", 0.0)),
                    "end": float(u.get("end", 0.0)),
                    "text": (u.get("transcript") or "").strip(),
                }
                for u in utterances
            ]
        channels = body.get("results", {}).get("channels") or []
        alts = channels[0].get("alternatives") if channels else None
        text = (alts[0].get("transcript") or "").strip() if alts else ""
        return [{"start": 0.0, "end": 0.0, "text": text}] if text else []


def build_asr(provider: str | None = None):
    """
    Returns the configured transcriber. DEP_ASR_PROVIDER selects between
    "local" (faster-whisper on this machine) and the hosted providers in
    CloudASR.CONFIG. Falls back to local — loudly, never silently — when a
    cloud provider is requested but its API key is missing, so a
    misconfiguration degrades instead of taking the app down mid-clinic.
    """
    provider = (provider or os.environ.get("DEP_ASR_PROVIDER", "local")).lower()
    if provider in CloudASR.CONFIG:
        try:
            return CloudASR(provider)
        except Exception as e:
            print(f"[asr] {provider} unavailable ({e}); falling back to local.")
    elif provider != "local":
        print(f"[asr] unknown DEP_ASR_PROVIDER={provider!r}; falling back to local.")
    return WhisperASR()


# Below this RMS a chunk is treated as silence and skipped. Whisper spends
# just as long on a silent chunk as a spoken one (and is prone to
# hallucinating text into the quiet), so this is the cheap stand-in for VAD
# when onnxruntime isn't available — it keeps live chunking ahead of realtime.
SILENCE_RMS_THRESHOLD = 0.005


def is_probably_silence(waveform: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    if waveform.size == 0:
        return True
    return float(np.sqrt(np.mean(np.square(waveform)))) < threshold


def run_asr_pipeline(waveform: np.ndarray, sample_rate: int, asr: WhisperASR,
                      speaker: str = "Participant") -> list:
    """
    Returns rows in the exact shape sync.load_transcript()/build_segments()
    expects: start_time, stop_time, speaker, value.
    """
    segments = asr.transcribe(waveform, sample_rate)
    return [
        {"start_time": seg["start"], "stop_time": seg["end"],
         "speaker": speaker, "value": seg["text"]}
        for seg in segments
    ]


if __name__ == "__main__":
    sr = 16000
    dummy = (np.random.randn(sr * 45) * 0.01).astype(np.float32)  # 45s of "audio"
    asr = WhisperASR()
    rows = run_asr_pipeline(dummy, sr, asr)
    for r in rows:
        print(r)
