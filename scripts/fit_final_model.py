"""
Fit the model that the application will actually serve, and export its weights.

Everything up to now has been evaluation: src/eval/benchmark.py fits models
inside cross-validation to estimate how well they generalise, and throws each
one away. Nothing has ever been fitted on the whole cohort and saved, which is
why api/main.py has been falling back to an untrained FusionHead and serving
random severity scores.

WHAT THIS DOES
    1. Loads the cached feature vectors and the prosodic feature table.
    2. Assembles the served feature set: text (768) + prosody (24) +
       demographic (16) = 808, in that order. The order is part of the
       contract with src/fusion/aggregate.py and must not change without
       refitting.
    3. Fits StandardScaler + Ridge(alpha) on all analysed participants.
    4. Folds the scaler into the ridge coefficients so inference is one affine
       map, and ASSERTS the folded form reproduces the sklearn pipeline before
       going further.
    5. Derives the caseness threshold by the Youden criterion.
    6. Exports outputs/weights/fusion_head.npz with provenance.

ON THE THRESHOLD
    The threshold shipped here is fitted on all available participants, whereas
    the performance reported in the thesis came from thresholds fitted per
    training fold. Both are correct for their purpose — the reported figure
    estimates generalisation, the shipped threshold should use all the data —
    but they are not the same number and the model card should say so.

USAGE
    python scripts/fit_final_model.py --cache fusion_vectors.npz \
        --prosody prosody.csv --qc "<kobo>/sessions/QC.csv"
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.eval.benchmark import attach_prosody, TEXT_SLICE, META_SLICE
from src.eval.calibrate import youden_threshold, sensitivity_threshold, binary_scores
from src.fusion.model import LinearHead, HAMD_CASENESS_THRESHOLD

TEXT_DIM, PROSODY_DIM, META_DIM = 768, 24, 16
SERVED_DIM = TEXT_DIM + PROSODY_DIM + META_DIM      # 808


def sha256(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(n), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--prosody", required=True)
    ap.add_argument("--alpha", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rule", default="youden", choices=["youden", "sens"])
    ap.add_argument("--target_sens", type=float, default=0.90)
    ap.add_argument("--out", default=os.path.join("outputs", "weights", "fusion_head.npz"))
    ap.add_argument("--feature_set", default="served", choices=["served", "legacy"],
                    help="served = text+prosody+metadata (808), the evaluated best; "
                         "legacy = text+wav2vec2+metadata (1552), matching what "
                         "api/main.py currently assembles. Both are exported during "
                         "the migration described in docs/mvp-plan.md.")
    a = ap.parse_args()

    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    d = np.load(a.cache, allow_pickle=True)
    X_all, Y, pids = d["X"].astype(np.float64), d["Y"].astype(np.float64), d["pids"]
    P = attach_prosody(pids, a.prosody, drop_duration=True)

    if a.feature_set == "served":
        X = np.hstack([X_all[:, TEXT_SLICE], P, X_all[:, META_SLICE]])
        expect, fs_name = SERVED_DIM, "text(768)+prosody(24)+metadata(16)"
    else:
        # The cache is already [text | wav2vec2 | metadata]; this is what the
        # scoring path assembles today, before the Day 2-3 migration.
        X = X_all
        expect, fs_name = X_all.shape[1], "text(768)+wav2vec2(768)+metadata(16)"
    assert X.shape[1] == expect, f"expected {expect} features, got {X.shape[1]}"
    print(f"fitting '{a.feature_set}' on {X.shape[0]} participants x {X.shape[1]} features")

    y_bin = (Y[:, 1] >= HAMD_CASENESS_THRESHOLD).astype(int)
    print(f"caseness: {y_bin.sum()} cases / {(1 - y_bin).sum()} non-cases")

    # ---- fit ----
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=a.alpha))
    pipe.fit(X, Y)
    scaler, ridge = pipe.named_steps["standardscaler"], pipe.named_steps["ridge"]

    # ---- fold the scaler into the coefficients ----
    # sklearn computes  y = C @ ((x - mean)/scale) + i
    # which is          y = (C/scale) @ x + (i - C @ (mean/scale))
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    W = ridge.coef_ / scale
    b = ridge.intercept_ - ridge.coef_ @ (scaler.mean_ / scale)

    pred_sklearn = pipe.predict(X)
    pred_folded = X @ W.T + b
    err = float(np.abs(pred_sklearn - pred_folded).max())
    print(f"folded-form max deviation from sklearn pipeline: {err:.3e}")
    assert err < 1e-4, ("folding the standardiser into the weights changed the "
                        "predictions; do not export this model")

    # ---- threshold, on the same data the weights were fitted to ----
    scores = pred_folded[:, 1]
    thr = (youden_threshold(y_bin, scores) if a.rule == "youden"
           else sensitivity_threshold(y_bin, scores, a.target_sens))
    at = binary_scores(y_bin, (scores >= thr).astype(int))
    auc_apparent = roc_auc_score(y_bin, scores)

    print(f"\nthreshold ({a.rule}): {thr:.3f}  (clinical cutoff is "
          f"{HAMD_CASENESS_THRESHOLD})")
    print("apparent performance on the fitting data — OPTIMISTIC, not a "
          "generalisation estimate:")
    print(f"  AUC {auc_apparent:.3f} | sens {at['sens']:.3f} | spec {at['spec']:.3f} "
          f"| bal acc {at['bal_acc']:.3f} | PPV {at['ppv']:.3f} | NPV {at['npv']:.3f}")
    print("  cross-validated figures are in outputs/benchmark*.csv and are the "
          "ones to report.")

    # ---- export ----
    meta = {
        "fitted": dt.datetime.now().isoformat(timespec="seconds"),
        "n_participants": int(X.shape[0]),
        "feature_set": fs_name,
        "feature_dim": int(X.shape[1]),
        "model": f"ridge(alpha={a.alpha}), standardiser folded into weights",
        "threshold_rule": a.rule,
        "threshold": float(thr),
        "cache_sha256": sha256(a.cache),
        "prosody_sha256": sha256(a.prosody),
        "seed": a.seed,
        "note": ("Threshold fitted on all participants; reported performance uses "
                 "per-fold thresholds. Apparent performance above is optimistic."),
    }
    head = LinearHead(W=W, b=b, decision_threshold=float(thr), meta=meta)
    head.save(a.out)
    print(f"\nwrote {a.out}")
    print(json.dumps(meta, indent=2))

    # ---- verify the exported file round-trips ----
    back = LinearHead.load(a.out)
    r = back.forward(X[0])
    ref = pred_folded[0]
    assert abs(r["hamd"] - np.clip(ref[1], 0, 44)) < 1e-3, "round-trip mismatch"
    assert abs(back.caseness_threshold - thr) < 1e-9
    print(f"round-trip OK — participant {pids[0]}: PHQ-9 {r['phq9']:.1f}, "
          f"HAM-D {r['hamd']:.1f}, caseness "
          f"{int(r['hamd'] >= back.caseness_threshold)}")


if __name__ == "__main__":
    main()
