"""
Audio pipeline: raw waveform -> per-30s-segment slice -> frozen Wav2Vec2 embedding.

No hand-engineered MFCC/Mel-spectrogram/spectral-contrast features needed —
Wav2Vec2 takes raw waveform and learns its own features. No augmentation,
since the backbone is frozen and used purely as a feature extractor.
"""
import hashlib
import json
import os

import numpy as np

EMBED_DIM = 768  # matches wav2vec2-base hidden size
TARGET_SR = 16000

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "synthetic", "sessions")


def load_audio(pid, data_root=DATA_ROOT):
    session_dir = os.path.join(data_root, str(pid))
    audio = np.load(os.path.join(session_dir, f"{pid}_AUDIO.npy"))
    with open(os.path.join(session_dir, f"{pid}_AUDIO.meta.json")) as f:
        meta = json.load(f)
    return audio, meta["sample_rate"]


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


class Wav2Vec2AudioEncoder:
    """Frozen audio encoder. encode() contract: np.ndarray waveform -> np.ndarray[EMBED_DIM]."""

    def __init__(self):
        # SWAP: load real model here, e.g.
        #   from transformers import Wav2Vec2Processor, Wav2Vec2Model
        #   self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        #   self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval()
        pass

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.size == 0:
            return np.zeros(EMBED_DIM, dtype=np.float32)
        # SWAP: replace this block with real Wav2Vec2 forward pass + mean-pooled
        # last_hidden_state, e.g.:
        #   inputs = self.processor(waveform, sampling_rate=TARGET_SR, return_tensors="pt")
        #   with torch.no_grad():
        #       out = self.model(**inputs)
        #   return out.last_hidden_state.mean(dim=1).squeeze().numpy()
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
