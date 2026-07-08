"""
Evaluation harness: accuracy, F1 (weighted/macro), RMSE, MAE.

Classification (accuracy + F1 + confusion matrix) is computed on HAM-D-derived
caseness — the interviewer-rated score is the clinical gold standard
(see model.HAMD_CASENESS_THRESHOLD). Regression error (RMSE/MAE) is reported
per target (PHQ-9 and HAM-D).

Acceptance target: accuracy >= 0.85 on HAM-D caseness.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.fusion.model import HAMD_CASENESS_THRESHOLD

ACCURACY_TARGET = 0.85


def _regression_errors(df, target):
    y_true = df[f"{target}_true"].values.astype(float)
    y_pred = df[f"{target}_pred"].values.astype(float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return round(rmse, 4), round(mae, 4)


def compute_metrics(df: pd.DataFrame, verbose=True) -> dict:
    """
    df must contain columns:
        phq9_true, phq9_pred, hamd_true, hamd_pred, binary_true, binary_pred
    (binary_* are HAM-D-derived caseness, produced upstream by run_pipeline.)
    """
    y_true_bin = df["binary_true"].values
    y_pred_bin = df["binary_pred"].values

    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    f1_weighted = f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    phq9_rmse, phq9_mae = _regression_errors(df, "phq9")
    hamd_rmse, hamd_mae = _regression_errors(df, "hamd")

    metrics = {
        "accuracy": round(accuracy, 4),
        "accuracy_target_met": bool(accuracy >= ACCURACY_TARGET),
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "phq9_rmse": phq9_rmse,
        "phq9_mae": phq9_mae,
        "hamd_rmse": hamd_rmse,
        "hamd_mae": hamd_mae,
        "n_participants": len(df),
        "caseness_threshold": HAMD_CASENESS_THRESHOLD,
    }

    if verbose:
        print("=" * 54)
        print("EVALUATION RESULTS  (classification = HAM-D caseness)")
        print("=" * 54)
        print(f"Participants:        {metrics['n_participants']}")
        print(f"Accuracy:            {metrics['accuracy']}  "
              f"(target >= {ACCURACY_TARGET}: {'PASS' if metrics['accuracy_target_met'] else 'FAIL'})")
        print(f"F1 (weighted):       {metrics['f1_weighted']}")
        print(f"F1 (macro):          {metrics['f1_macro']}")
        print(f"PHQ-9 RMSE / MAE:    {metrics['phq9_rmse']} / {metrics['phq9_mae']}")
        print(f"HAM-D RMSE / MAE:    {metrics['hamd_rmse']} / {metrics['hamd_mae']}")
        print()
        print(f"Classification report (HAM-D caseness, threshold >= {HAMD_CASENESS_THRESHOLD}):")
        print(classification_report(y_true_bin, y_pred_bin, labels=[0, 1],
              target_names=["Non-case", "Case"], zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]))
        print("=" * 54)

    return metrics


if __name__ == "__main__":
    pred_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "outputs", "predictions", "predictions.csv"
    )
    if os.path.exists(pred_path):
        df = pd.read_csv(pred_path)
        compute_metrics(df)
    else:
        print("No predictions file found. Run src/fusion/run_pipeline.py first.")
