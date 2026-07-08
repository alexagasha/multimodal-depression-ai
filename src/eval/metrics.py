"""
Evaluation harness: F1 (weighted), RMSE, MAE — matching the reporting format
of the Phenomics (Zhang et al. 2024) and MDPI Computation (Nykoniuk et al. 2025)
benchmark papers so results are directly comparable.

Also produces a per-class breakdown and a simple confusion matrix printout.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)

DEPRESSION_THRESHOLD = 10.0


def compute_metrics(df: pd.DataFrame, verbose=True) -> dict:
    """
    df must contain columns: phq8_true, phq8_pred, binary_true, binary_pred

    Returns dict with all key metrics.
    """
    y_true_bin = df["binary_true"].values
    y_pred_bin = df["binary_pred"].values
    y_true_cont = df["phq8_true"].values
    y_pred_cont = df["phq8_pred"].values

    f1_weighted = f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0)
    rmse = float(np.sqrt(np.mean((y_true_cont - y_pred_cont) ** 2)))
    mae = float(mean_absolute_error(y_true_cont, y_pred_cont))

    metrics = {
        "f1_weighted": round(f1_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "n_participants": len(df),
        "depression_threshold": DEPRESSION_THRESHOLD,
    }

    if verbose:
        print("=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Participants:      {metrics['n_participants']}")
        print(f"F1 (weighted):     {metrics['f1_weighted']}")
        print(f"F1 (macro):        {metrics['f1_macro']}")
        print(f"RMSE (severity):   {metrics['rmse']}")
        print(f"MAE  (severity):   {metrics['mae']}")
        print()
        print("Classification report:")
        print(classification_report(y_true_bin, y_pred_bin,
              target_names=["Non-depressed", "Depressed"], zero_division=0))
        print("Confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(y_true_bin, y_pred_bin))
        print("=" * 50)

    return metrics


if __name__ == "__main__":
    import os
    pred_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "outputs", "predictions", "predictions.csv"
    )
    if os.path.exists(pred_path):
        df = pd.read_csv(pred_path)
        compute_metrics(df)
    else:
        print("No predictions file found. Run src/fusion/run_pipeline.py first.")
