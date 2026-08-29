"""
Lightweight fusion head: concatenated participant vector -> two severity scores.

Multi-task output (agreed): the tool collects both PHQ-9 (self-report) and
HAM-D (interviewer-rated), so the head predicts BOTH.

Phase 1 architecture (numpy-only, no torch dependency):
  Linear(D -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 64) -> ReLU -> Linear(64 -> 2)
  outputs: [phq9, hamd]

This is a deterministic forward-pass stub with random weights that proves the
shape plumbing is correct end-to-end. Weights are not trained here — training
happens in train.py once real (or larger synthetic) data is ready.

SWAP instructions for real PyTorch model:
  See the FusionHeadTorch class at the bottom of this file (final layer -> 2).

Outputs: phq9 in [0, 27], hamd in [0, 44], clamped at inference.
Binary caseness (the primary classification label) is HAM-D-derived — the
interviewer-rated score is the clinical gold standard. See HAMD_CASENESS_THRESHOLD.
"""
import json
import os

import numpy as np

from .aggregate import FUSION_INPUT_DIM

HIDDEN1 = 256
HIDDEN2 = 64
OUTPUT_DIM = 2  # [phq9, hamd]

PHQ9_RANGE = (0.0, 27.0)
HAMD_RANGE = (0.0, 44.0)  # 11 items x 0-4; paper form misprints /52

# Binary caseness ("depressed") is derived from HAM-D (clinician-rated gold standard).
# Standard HAM-D-17 caseness is >=8 on the /52 scale; scaled to this 11-item /44
# form: 8 * 44/52 ~= 7. Confirm against a validated cutoff for the reduced scale.
HAMD_CASENESS_THRESHOLD = 7.0


def relu(x):
    return np.maximum(0, x)


class FusionHead:
    """
    Numpy-only fusion MLP. Weights are random at init (untrained).
    Replace with FusionHeadTorch when ready to train for real.
    """

    def __init__(self, input_dim=FUSION_INPUT_DIM, seed=0, decision_threshold=None):
        # Decision threshold for caseness, in PREDICTED-HAM-D space rather than
        # clinical HAM-D space. A regression head shrinks toward the training
        # mean, so its outputs occupy a narrower range than the instrument does:
        # with 83% of this cohort above 7 and a mean of 13.3, predictions rarely
        # fall below the clinical cutoff and specificity collapses to zero even
        # at AUC 0.83. The calibrated value comes from src/eval/calibrate.py,
        # which picks it on training folds only, and is stored with the weights.
        # None -> fall back to the clinical cutoff (previous behaviour).
        self.decision_threshold = decision_threshold
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / HIDDEN1)
        scale3 = np.sqrt(2.0 / HIDDEN2)
        self.W1 = rng.normal(0, scale1, (HIDDEN1, input_dim)).astype(np.float32)
        self.b1 = np.zeros(HIDDEN1, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (HIDDEN2, HIDDEN1)).astype(np.float32)
        self.b2 = np.zeros(HIDDEN2, dtype=np.float32)
        self.W3 = rng.normal(0, scale3, (OUTPUT_DIM, HIDDEN2)).astype(np.float32)
        self.b3 = np.zeros(OUTPUT_DIM, dtype=np.float32)

    def forward(self, x: np.ndarray) -> dict:
        """x: shape (FUSION_INPUT_DIM,) -> {"phq9": float, "hamd": float}."""
        h1 = relu(self.W1 @ x + self.b1)
        h2 = relu(self.W2 @ h1 + self.b2)
        out = (self.W3 @ h2 + self.b3).flatten()
        return {
            "phq9": float(np.clip(out[0], *PHQ9_RANGE)),
            "hamd": float(np.clip(out[1], *HAMD_RANGE)),
        }

    @property
    def caseness_threshold(self) -> float:
        """The cut point actually used, calibrated if one was trained."""
        return (HAMD_CASENESS_THRESHOLD if self.decision_threshold is None
                else float(self.decision_threshold))

    def predict_label(self, x: np.ndarray) -> int:
        """Binary caseness from predicted HAM-D, at the calibrated threshold."""
        return int(self.forward(x)["hamd"] >= self.caseness_threshold)

    # --- persistence: save / load trained weights as a numpy .npz ---
    def get_weights(self) -> dict:
        w = {"W1": self.W1, "b1": self.b1, "W2": self.W2,
             "b2": self.b2, "W3": self.W3, "b3": self.b3}
        if self.decision_threshold is not None:
            w["decision_threshold"] = np.array(float(self.decision_threshold))
        return w

    def set_weights(self, w: dict):
        for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(self, k, np.asarray(w[k], dtype=np.float32))
        if "decision_threshold" in w:
            self.decision_threshold = float(np.asarray(w["decision_threshold"]))
        return self

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, **self.get_weights())

    @classmethod
    def load(cls, path):
        d = np.load(path)
        in_dim = int(d["W1"].shape[1])
        if in_dim != FUSION_INPUT_DIM:
            print(f"[FusionHead.load] note: weights input_dim={in_dim} "
                  f"but current FUSION_INPUT_DIM={FUSION_INPUT_DIM}")
        keys = ["W1", "b1", "W2", "b2", "W3", "b3"]
        if "decision_threshold" in d.files:
            keys.append("decision_threshold")
        m = cls(input_dim=in_dim).set_weights({k: d[k] for k in keys})
        if m.decision_threshold is None:
            print("[FusionHead.load] no calibrated threshold in weights; using the "
                  "clinical cutoff, which yields near-zero specificity on this cohort. "
                  "Run src/eval/calibrate.py and save the threshold with the weights.")
        return m


# ---------------------------------------------------------------------------
# Training: the PyTorch version of this head lives in src/fusion/train.py.
# It trains an identical MLP, then exports the weights back into this numpy
# FusionHead via .save()/.load() — so inference stays numpy-only (no torch).
# ---------------------------------------------------------------------------


class LinearHead:
    """Ridge regression as a single affine map: y = Wx + b.

    WHY THIS EXISTS
        Evaluation over the whole cohort (src/eval/benchmark.py) found ridge
        beat the MLP above on every feature set examined — 0.826 vs 0.695 AUC on
        prosodic features, 0.791 vs 0.673 on linguistic — which is what a
        414,000-parameter network does when fitted to ~108 participants per
        fold. This head is therefore the one to serve.

    THE STANDARDISER IS FOLDED IN
        The evaluated model is a StandardScaler followed by Ridge. Rather than
        ship the scaler separately and risk it drifting out of sync with the
        weights, both collapse into one affine transform:

            W' = W / scale
            b' = b - W @ (mean / scale)

        so inference is a single matrix multiply with no preprocessing state.
        scripts/fit_final_model.py asserts this reproduces the sklearn pipeline
        before exporting.

    Interface matches FusionHead exactly, so api/main.py needs no special case.
    """

    def __init__(self, W=None, b=None, input_dim=None, decision_threshold=None,
                 meta=None):
        d = input_dim if input_dim is not None else FUSION_INPUT_DIM
        self.W = (np.zeros((OUTPUT_DIM, d), dtype=np.float32)
                  if W is None else np.asarray(W, dtype=np.float32))
        self.b = (np.zeros(OUTPUT_DIM, dtype=np.float32)
                  if b is None else np.asarray(b, dtype=np.float32))
        self.decision_threshold = decision_threshold
        self.meta = meta or {}          # provenance: date, seed, feature set

    @property
    def input_dim(self) -> int:
        return int(self.W.shape[1])

    def forward(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=np.float32).ravel()
        if x.shape[0] != self.input_dim:
            raise ValueError(
                f"feature vector has {x.shape[0]} dimensions but these weights "
                f"expect {self.input_dim}. The served feature set must match the "
                f"one the model was fitted on.")
        out = (self.W @ x + self.b).ravel()
        return {
            "phq9": float(np.clip(out[0], *PHQ9_RANGE)),
            "hamd": float(np.clip(out[1], *HAMD_RANGE)),
        }

    @property
    def caseness_threshold(self) -> float:
        return (HAMD_CASENESS_THRESHOLD if self.decision_threshold is None
                else float(self.decision_threshold))

    def predict_label(self, x: np.ndarray) -> int:
        return int(self.forward(x)["hamd"] >= self.caseness_threshold)

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """Per-feature contribution to the HAM-D prediction: W[hamd] * x.

        Exact for a linear model rather than approximated, which is what makes
        the attribution shown to a clinician trustworthy."""
        x = np.asarray(x, dtype=np.float32).ravel()
        return (self.W[1] * x).astype(np.float32)

    # --- persistence ---
    def get_weights(self) -> dict:
        w = {"W": self.W, "b": self.b}
        if self.decision_threshold is not None:
            w["decision_threshold"] = np.array(float(self.decision_threshold))
        if self.meta:
            w["meta_json"] = np.array(json.dumps(self.meta))
        return w

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, **self.get_weights())

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        meta = {}
        if "meta_json" in d.files:
            try:
                meta = json.loads(str(d["meta_json"]))
            except Exception:
                meta = {}
        m = cls(W=d["W"], b=d["b"], meta=meta,
                decision_threshold=(float(d["decision_threshold"])
                                    if "decision_threshold" in d.files else None))
        if m.decision_threshold is None:
            print("[LinearHead.load] no calibrated threshold in weights; falling "
                  "back to the clinical cutoff, which gives near-zero specificity "
                  "on this cohort. See src/eval/calibrate.py.")
        return m


def load_head(path):
    """Load whichever head a weights file contains.

    Lets api/main.py stay agnostic while the MLP weights format remains
    supported for older exports."""
    d = np.load(path, allow_pickle=False)
    return LinearHead.load(path) if "W" in d.files else FusionHead.load(path)


if __name__ == "__main__":
    model = FusionHead()
    dummy = np.random.randn(FUSION_INPUT_DIM).astype(np.float32)
    scores = model.forward(dummy)
    label = model.predict_label(dummy)
    print(f"PHQ-9={scores['phq9']:.2f}  HAM-D={scores['hamd']:.2f}  binary={label}")
