"""
Audio pipeline: raw waveform -> per-30s-segment slice -> frozen Wav2Vec2 embedding.

No hand-engineered MFCC/Mel-spectrogram/spectral-contrast features needed —
Wav2Vec2 takes raw waveform and learns its own features. No augmentation,
since the backbone is frozen and used purely as a feature extractor.

MULTILINGUAL: the study spans ~3 languages (English / Luganda / Luo). The
English-only wav2vec2-base is a poor fit; the real SWAP target should be a
multilingual model (e.g. facebook/wav2vec2-large-xlsr-53) — see __init__.
"""
import hashlib
import json
import os

import numpy as np

EMBED_DIM = 768  # matches wav2vec2-base hidden size
TARGET_SR = 16000

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic", "sessions")


def _load_wav(path):
    """Read a WAV as mono float32 in [-1, 1], plus its sample rate."""
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        from scipy.io import wavfile
        sr, raw = wavfile.read(path)
        if np.issubdtype(raw.dtype, np.integer):
            audio = raw.astype(np.float32) / float(np.iinfo(raw.dtype).max)
        else:
            audio = raw.astype(np.float32)
    if audio.ndim > 1:                      # stereo -> mono
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def load_audio(pid, data_root=DATA_ROOT):
    """Load participant audio. Supports synthetic .npy (+meta) and real .wav (E-DAIC)."""
    session_dir = os.path.join(data_root, str(pid))
    npy = os.path.join(session_dir, f"{pid}_AUDIO.npy")
    wav = os.path.join(session_dir, f"{pid}_AUDIO.wav")
    if os.path.exists(npy):
        audio = np.load(npy)
        with open(os.path.join(session_dir, f"{pid}_AUDIO.meta.json")) as f:
            return audio, json.load(f)["sample_rate"]
    if os.path.exists(wav):
        return _load_wav(wav)
    raise FileNotFoundError(f"No {pid}_AUDIO.npy or {pid}_AUDIO.wav in {session_dir}")


def resample_if_needed(audio, sr, target_sr=TARGET_SR):
    # SWAP: real resampling, e.g. librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    if sr == target_sr:
        return audio
    ratio = target_sr / sr
    new_len = int(len(audio) * ratio)
    return np.interp(np.linspace(0, len(audio), new_len), np.arange(len(audio)), audio).astype(np.float32)


def light_denoise(audio):
    # SWAP: real denoising, e.g. noisereduce.reduce_noise(y=audio, sr=TARGET_SR)
    return audio  # placeholder no-op for synthetic data


def slice_segment(audio, sr, start_sec, end_sec):
    start_idx = int(start_sec * sr)
    end_idx = int(end_sec * sr)
    return audio[start_idx:end_idx]


# Default audio encoder. wav2vec2-base is 768-d (== EMBED_DIM) and works for the
# English E-DAIC smoke test. For the multilingual Uganda data switch to a model
# like facebook/wav2vec2-large-xlsr-53 — but that is 1024-d, so you must bump
# EMBED_DIM here and AUDIO_DIM in fusion/aggregate.py to 1024 to match.
DEFAULT_AUDIO_MODEL = "facebook/wav2vec2-base-960h"  # 768-d


class Wav2Vec2AudioEncoder:
    """
    Frozen audio encoder. encode() contract: np.ndarray waveform -> np.ndarray[EMBED_DIM].

    Loads a real HuggingFace Wav2Vec2 model when torch + transformers are
    available (e.g. Colab GPU); otherwise falls back to deterministic mock
    embeddings so the pipeline/tests run on a CPU-only box with no ML deps.
    """

    def __init__(self, model_name=None, device=None):
        self.model = None
        self.processor = None
        self._torch = None
        self.model_name = model_name or os.environ.get("DEP_AUDIO_MODEL", DEFAULT_AUDIO_MODEL)
        try:
            import torch
            from transformers import AutoProcessor, Wav2Vec2Model
            self._torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Wav2Vec2Model.from_pretrained(self.model_name).eval().to(self.device)
        except Exception as e:  # transformers/torch missing or load failed -> mock
            self.model = None
            print(f"[audio] real model unavailable ({type(e).__name__}); using mock embeddings.")
        if self.model is not None:
            hidden = getattr(self.model.config, "hidden_size", EMBED_DIM)
            if hidden != EMBED_DIM:
                raise ValueError(
                    f"{self.model_name} hidden size {hidden} != EMBED_DIM {EMBED_DIM}. "
                    f"Update EMBED_DIM here and AUDIO_DIM in fusion/aggregate.py.")

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.size == 0:
            return np.zeros(EMBED_DIM, dtype=np.float32)
        if self.model is None:
            return self._mock_encode(waveform)
        torch = self._torch
        inputs = self.processor(waveform, sampling_rate=TARGET_SR,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        vec = out.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
        return vec.astype(np.float32)

    @staticmethod
    def _mock_encode(waveform: np.ndarray) -> np.ndarray:
        seed = int(hashlib.sha256(waveform.tobytes()[:1000]).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.normal(size=EMBED_DIM).astype(np.float32)


def run_audio_pipeline(pid, segments, encoder: Wav2Vec2AudioEncoder, data_root=DATA_ROOT):
    """segments: output of sync.build_segments(). Returns list[np.ndarray]."""
    audio, sr = load_audio(pid, data_root)
    audio = resample_if_needed(audio, sr)
    audio = light_denoise(audio)

    embeddings = []
    for seg in segments:
        clip = slice_segment(audio, TARGET_SR, seg["start"], seg["end"])
        embeddings.append(encoder.encode(clip))
    return embeddings


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from sync import build_segments

    segs = build_segments(300)
    enc = Wav2Vec2AudioEncoder()
    embs = run_audio_pipeline(300, segs, enc)
    print(f"{len(embs)} audio segment embeddings, dim={embs[0].shape if embs else None}")
