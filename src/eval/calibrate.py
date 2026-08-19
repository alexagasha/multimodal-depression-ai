"""
Decision-threshold calibration for the caseness head.

THE PROBLEM
    Ridge shrinks its predictions toward the training mean. With 83% of the
    cohort above HAM-D 7 and a mean of 13.3, predicted scores cluster in the
    low teens and essentially never fall below the clinical cutoff. The result
    is AUC 0.826 — the ranking is good — alongside specificity 0.000, because
    every participant lands on the "case" side of a threshold inherited from
    the instrument rather than from the model's own output distribution.

    So the model is not failing to discriminate. It is being asked to make its
    decision at a point its predictions never reach.

THE FIX, AND WHY IT IS DONE THIS WAY
    Pick the threshold from the model's predicted distribution instead of the
    clinical scale. The only honest way to do that is inside the CV loop:
    fit on the training fold, predict the TRAINING fold, choose the threshold
    there, then apply it unchanged to the held-out fold. Choosing it on test
    predictions would leak the answer and inflate every number below.

    Two criteria are offered because they answer different clinical questions:
      youden   maximise sensitivity + specificity - 1; the balanced operating
               point, appropriate when both error types cost similarly.
      sens@X   the threshold that achieves at least X sensitivity on the
               training fold; appropriate for screening, where a missed case
               is worse than a false alarm.

    A direct classifier (logistic regression with balanced class weights) is
    included as a comparison, since regression-then-threshold is not obviously
    the right design when caseness is the target.

USAGE
    python src/eval/calibrate.py --cache fusion_vectors.npz \
        --prosody prosody.csv --qc QC.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.eval.benchmark import (attach_prosody, attach_duration, load_cache,
                                TEXT_SLICE, AUDIO_SLICE, META_SLICE)
from src.fusion.model import HAMD_CASENESS_THRESHOLD


# --------------------------------------------------------------------------
def youden_threshold(y_bin, scores):
    """Threshold maximising sensitivity + specificity - 1."""
    cand = np.unique(scores)
    if len(cand) < 2:
        return float(np.median(scores))
    best, best_j = float(cand[0]), -np.inf
    for t in cand:
        pred = (scores >= t).astype(int)
        tp = np.sum((y_bin == 1) & (pred == 1)); fn = np.sum((y_bin == 1) & (pred == 0))
        tn = np.sum((y_bin == 0) & (pred == 0)); fp = np.sum((y_bin == 0) & (pred == 1))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1
        if j > best_j:
            best_j, best = j, float(t)
    return best


def sensitivity_threshold(y_bin, scores, target=0.90):
    """Highest threshold that still achieves `target` sensitivity.

    Highest, not lowest: among all cut points meeting the sensitivity floor we
    want the one that rules out the most non-cases, which is the whole value of
    a screening threshold."""
    cand = np.unique(scores)[::-1]
    chosen = float(np.min(scores))
    for t in cand:
        pred = (scores >= t).astype(int)
        tp = np.sum((y_bin == 1) & (pred == 1)); fn = np.sum((y_bin == 1) & (pred == 0))
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        if sens >= target:
            chosen = float(t)
            break
    return chosen


def binary_scores(y_bin, pred_bin):
    tp = int(np.sum((y_bin == 1) & (pred_bin == 1))); fn = int(np.sum((y_bin == 1) & (pred_bin == 0)))
    tn = int(np.sum((y_bin == 0) & (pred_bin == 0))); fp = int(np.sum((y_bin == 0) & (pred_bin == 1)))
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {"sens": sens, "spec": spec, "ppv": ppv, "npv": npv,
            "bal_acc": np.nanmean([sens, spec]),
            "accuracy": (tp + tn) / max(len(y_bin), 1)}


# --------------------------------------------------------------------------
def run_view(X, Y, mode, folds=5, repeats=5, seed=0, target=0.90, classifier=False):
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    y_bin = (Y[:, 1] >= HAMD_CASENESS_THRESHOLD).astype(int)
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    rows = []
    for tr, te in cv.split(X, y_bin):
        if classifier:
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=5000, class_weight="balanced", C=0.05))
            m.fit(X[tr], y_bin[tr])
            s_tr = m.predict_proba(X[tr])[:, 1]
            s_te = m.predict_proba(X[te])[:, 1]
        else:
            m = make_pipeline(StandardScaler(), Ridge(alpha=1000.0))
            m.fit(X[tr], Y[tr])
            s_tr = np.asarray(m.predict(X[tr]))[:, 1]
            s_te = np.asarray(m.predict(X[te]))[:, 1]

        if mode == "clinical":
            t = HAMD_CASENESS_THRESHOLD
        elif mode == "youden":
            t = youden_threshold(y_bin[tr], s_tr)      # train fold only
        else:
            t = sensitivity_threshold(y_bin[tr], s_tr, target)

        r = binary_scores(y_bin[te], (s_te >= t).astype(int))
        r["threshold"] = t
        r["auc"] = (roc_auc_score(y_bin[te], s_te)
                    if len(set(y_bin[te])) > 1 and np.std(s_te) > 1e-12 else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Calibrate the caseness decision threshold.")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--prosody", default=None)
    ap.add_argument("--qc", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target_sens", type=float, default=0.90)
    ap.add_argument("--out", default=os.path.join("outputs", "calibration.csv"))
    a = ap.parse_args()

    X, Y, pids = load_cache(a.cache)
    y_bin = (Y[:, 1] >= HAMD_CASENESS_THRESHOLD).astype(int)
    print(f"caseness: {y_bin.sum()} cases / {(1-y_bin).sum()} non-cases\n")

    views = {"text (768d)": X[:, TEXT_SLICE]}
    if a.prosody and os.path.exists(a.prosody):
        P = attach_prosody(pids, a.prosody, drop_duration=True)
        views["prosody (24d)"] = P
        views["text + prosody"] = np.hstack([X[:, TEXT_SLICE], P])
    if a.qc and os.path.exists(a.qc):
        views["duration only"] = attach_duration(pids, a.qc)

    modes = [("clinical", "ridge, HAM-D>=7"),
             ("youden", "ridge, Youden"),
             (f"sens@{a.target_sens:g}", f"ridge, sens>={a.target_sens:g}")]

    cols = ["sens", "spec", "bal_acc", "ppv", "npv", "accuracy", "auc"]
    hdr = f"{'view':<17}{'threshold rule':<22}" + " ".join(f"{c:>13}" for c in cols)
    print("=" * len(hdr))
    print(f"{a.repeats}x{a.folds}-fold CV — threshold chosen on the TRAINING fold only")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    out_rows = []
    for vname, Xv in views.items():
        for mode, label in modes:
            df = run_view(Xv, Y, mode.split("@")[0] if "@" in mode else mode,
                          a.folds, a.repeats, a.seed, a.target_sens, classifier=False)
            cells = [f"{df[c].mean():.3f}±{df[c].std():.3f}" for c in cols]
            print(f"{vname:<17}{label:<22}" + " ".join(f"{c:>13}" for c in cells))
            out_rows.append({"view": vname, "rule": label,
                             **{f"{c}_mean": df[c].mean() for c in cols}})
        # Direct classifier: predicts caseness rather than thresholding a
        # regression. Its threshold is also picked on the training fold, since
        # a balanced-weight model's natural 0.5 cut is not necessarily optimal.
        df = run_view(Xv, Y, "youden", a.folds, a.repeats, a.seed,
                      a.target_sens, classifier=True)
        cells = [f"{df[c].mean():.3f}±{df[c].std():.3f}" for c in cols]
        print(f"{vname:<17}{'logistic, balanced':<22}" + " ".join(f"{c:>13}" for c in cells))
        out_rows.append({"view": vname, "rule": "logistic, balanced",
                         **{f"{c}_mean": df[c].mean() for c in cols}})
        print("-" * len(hdr))

    print("\nReading this:")
    print("  'HAM-D>=7' is the current behaviour — specificity near zero means the")
    print("  model calls everyone a case, not that it cannot discriminate.")
    print("  Youden gives the balanced operating point; sens@0.9 is the screening")
    print("  setting, trading specificity for catching cases.")
    print("  PPV/NPV are the numbers a clinician actually experiences at an 83% base rate.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(out_rows).to_csv(a.out, index=False)
    print(f"\nwritten -> {a.out}")


if __name__ == "__main__":
    main()
