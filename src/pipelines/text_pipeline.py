"""
Text pipeline: transcript -> per-segment text -> frozen embedding.

MULTILINGUAL: the study collects data in ~3 languages (assumed English /
Luganda / Luo). The real encoder must therefore be a multilingual model
(e.g. XLM-RoBERTa or an African-language model such as AfriBERTa) rather than
English bert-base — see the SWAP notes below. The participant's preferred
language is available as the `language` field in metadata if needed for
routing/adapters.

No stop-word removal, stemming, or TF-IDF — the frozen encoder learns its own
representation from near-raw text, so preprocessing here is intentionally
minimal (English contraction expansion + whitespace cleanup only; the
contraction step is a harmless no-op for Luganda/Luo text).
"""
import hashlib
import os
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


# Default text encoder — multilingual, so it covers the ~3 study languages.
# Override with the DEP_TEXT_MODEL env var (must have hidden_size == EMBED_DIM).
DEFAULT_TEXT_MODEL = "xlm-roberta-base"  # 768-d, multilingual


class BertTextEncoder:
    """
    Frozen text encoder. encode() contract: str -> np.ndarray[EMBED_DIM].

    Loads a real HuggingFace model when torch + transformers are available
    (e.g. on Colab GPU); otherwise falls back to deterministic mock embeddings
    so the pipeline and test suite still run on a CPU-only box with no ML deps.
    """

    def __init__(self, model_name=None, device=None):
        self.model = None
        self.tokenizer = None
        self._torch = None
        self.model_name = model_name or os.environ.get("DEP_TEXT_MODEL", DEFAULT_TEXT_MODEL)
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).eval().to(self.device)
        except Exception as e:  # transformers/torch missing or load failed -> mock
            self.model = None
            print(f"[text] real model unavailable ({type(e).__name__}); using mock embeddings.")
        if self.model is not None:
            hidden = getattr(self.model.config, "hidden_size", EMBED_DIM)
            if hidden != EMBED_DIM:
                raise ValueError(
                    f"{self.model_name} hidden size {hidden} != EMBED_DIM {EMBED_DIM}. "
                    f"Update EMBED_DIM/TEXT_DIM/FUSION_INPUT_DIM to match this model.")

    def encode(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros(EMBED_DIM, dtype=np.float32)
        if self.model is None:
            return self._mock_encode(text)
        torch = self._torch
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model(**tokens)
        vec = out.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
        return vec.astype(np.float32)

    @staticmethod
    def _mock_encode(text: str) -> np.ndarray:
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
