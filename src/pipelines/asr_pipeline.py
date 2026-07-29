"""
ASR pipeline: raw session audio -> TRANSCRIPT.csv-shaped rows.

Bridges live-recorded audio (web app) into the existing pipeline, which has
always assumed a transcript already exists (previously produced by "a local
firm" per methodology.txt §3.6). Live sessions need transcription in-app.

Same real-model-with-mock-fallback convention as text_pipeline.BertTextEncoder
and audio_pipeline.Wav2Vec2AudioEncoder: loads real Whisper when available,
otherwise falls back to a deterministic mock so the API/tests run without the
(large) whisper dependency installed.

MVP: transcribes the whole session audio in one pass and buckets segments into
fixed SEGMENT_LEN_SEC windows (matching sync.SEGMENT_LEN_SEC) — not full
real-time streaming captions. Speaker attribution MVP: the caller (web app)
supplies a speaker label per chunk (push-to-talk toggle) rather than this
module doing diarization; single-speaker audio defaults to "Participant".
"""
import os

import numpy as np

from src.pipelines.sync import SEGMENT_LEN_SEC

# Default ASR model — override with the DEP_ASR_MODEL env var.
DEFAULT_ASR_MODEL = "small"  # matches the Whisper checkpoint already cached on Colab


class WhisperASR:
    """
    Frozen transcription model. transcribe() contract:
    (waveform: np.ndarray, sample_rate: int) -> list[{"start", "end", "text"}].

    Loads real Whisper when the `whisper` package is available; otherwise
    falls back to a deterministic mock transcript so the pipeline/tests run
    on a box with no ASR deps installed.
    """

    def __init__(self, model_name=None):
        self.model = None
        self.model_name = model_name or os.environ.get("DEP_ASR_MODEL", DEFAULT_ASR_MODEL)
        try:
            import whisper
            self.model = whisper.load_model(self.model_name)
        except Exception as e:  # whisper missing or load failed -> mock
            self.model = None
            print(f"[asr] real model unavailable ({type(e).__name__}); using mock transcription.")

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> list:
        if waveform.size == 0:
            return []
        if self.model is None:
            return self._mock_transcribe(waveform, sample_rate)
        result = self.model.transcribe(waveform, fp16=False)
        return [
            {"start": float(seg["start"]), "end": float(seg["end"]), "text": seg["text"].strip()}
            for seg in result.get("segments", [])
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
