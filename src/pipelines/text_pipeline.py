"""
Text pipeline: transcript -> per-segment text -> frozen BERT embedding.

No stop-word removal, stemming, or TF-IDF — frozen BERT learns its own
representation from near-raw text, so preprocessing here is intentionally
minimal (contraction expansion + whitespace cleanup only).
"""
import hashlib
import re

import numpy as np

EMBED_DIM = 768  # matches bert-base hidden size

CONTRACTIONS = {
    "don't": "do not", "can't": "cannot", "won't": "will not",
    "i'm": "i am", "it's": "it is", "i've": "i have", "didn't": "did not",
}


def expand_contractions(text):
    for c, full in CONTRACTIONS.items():
        text = re.sub(c, full, text, flags=re.IGNORECASE)
    return text


def clean_text(text):
    text = expand_contractions(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class BertTextEncoder:
    """Frozen text encoder. encode() contract: str -> np.ndarray[EMBED_DIM]."""

    def __init__(self):
        # SWAP: load real model here, e.g.
        #   from transformers import AutoTokenizer, AutoModel
        #   self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        #   self.model = AutoModel.from_pretrained("bert-base-uncased").eval()
        pass

    def encode(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(EMBED_DIM, dtype=np.float32)
        # SWAP: replace this block with real BERT forward pass + mean-pooled
        # last_hidden_state, e.g.:
        #   tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        #   with torch.no_grad():
        #       out = self.model(**tokens)
        #   return out.last_hidden_state.mean(dim=1).squeeze().numpy()
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        return rng.normal(size=EMBED_DIM).astype(np.float32)


def run_text_pipeline(segments, encoder: BertTextEncoder):
    """segments: output of sync.build_segments(). Returns list[np.ndarray]."""
    embeddings = []
    for seg in segments:
        cleaned = clean_text(seg["text"])
        embeddings.append(encoder.encode(cleaned))
    return embeddings


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from sync import build_segments

    segs = build_segments(300)
    enc = BertTextEncoder()
    embs = run_text_pipeline(segs, enc)
    print(f"{len(embs)} text segment embeddings, dim={embs[0].shape if embs else None}")
