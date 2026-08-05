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
"""
import os

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
        # _load_wav already produces. beam_size=1 (greedy) roughly halves CPU
        # time versus the default beam search — the difference that makes
        # real-time chunking viable without a GPU.
        segments, _info = self.model.transcribe(
            waveform.astype(np.float32), beam_size=1, vad_filter=self.use_vad
        )
        return [
            {"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()}
            for seg in segments
            if seg.text.strip()
        ]

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
