# Multilingual Fake News Detector — evaluate.py
"""Offline evaluation script for the Multilingual Fake News Detector.

Usage:
    python evaluate.py --csv path/to/test.csv

The CSV must contain columns: text, language, label (0=Real, 1=Fake).
Reports overall metrics plus a per-language breakdown for English, Urdu,
and Spanish, and writes evaluation_results.json next to the script.
"""

import argparse
import json
import sys
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from model_loader import LABELS, SUPPORTED_LANGUAGES, predict

REQUIRED_COLUMNS = {"text", "language", "label"}
RESULTS_PATH = "evaluation_results.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Multilingual Fake News Detector on a labelled CSV."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to a CSV with columns: text, language, label (0=Real, 1=Fake).",
    )
    return parser.parse_args()


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the evaluation CSV and validate its schema.

    Args:
        csv_path: Path to the labelled CSV file.

    Returns:
        A pandas DataFrame with the required columns.

    Raises:
        ValueError: If required columns are missing or labels are out of range.
    """
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if not df["label"].isin([0, 1]).all():
        raise ValueError("label column must contain only 0 (Real) or 1 (Fake).")
    return df


def run_predictions(df: pd.DataFrame) -> list:
    """Run the model on every row of the dataset.

    Args:
        df: The loaded dataset.

    Returns:
        A list of predicted label indices (0=Real, 1=Fake) in row order.
    """
    predictions = []
    for text in df["text"].astype(str).tolist():
        try:
            result = predict(text)
        except (ValueError, RuntimeError) as exc:
            print(f"Warning: prediction failed for one row: {exc}", file=sys.stderr)
            predictions.append(-1)
            continue
        predictions.append(LABELS.index(result["prediction"]))
    return predictions


def compute_overall_metrics(y_true, y_pred) -> dict:
    """Compute overall accuracy, precision, recall, and F1 (macro + weighted)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_per_language_metrics(df: pd.DataFrame) -> dict:
    """Compute accuracy and F1 broken down per supported language."""
    per_lang = {}
    for language in SUPPORTED_LANGUAGES:
        subset = df[df["language"] == language]
        if subset.empty:
            per_lang[language] = {"samples": 0, "accuracy": None, "f1_macro": None}
            continue
        valid = subset[subset["pred"] != -1]
        per_lang[language] = {
            "samples": int(len(subset)),
            "accuracy": float(accuracy_score(valid["label"], valid["pred"])),
            "f1_macro": float(f1_score(valid["label"], valid["pred"], average="macro", zero_division=0)),
        }
    return per_lang


def main() -> None:
    """Entry point for the evaluation CLI."""
    args = parse_args()

    print(f"Loading dataset from {args.csv} ...")
    df = load_dataset(args.csv)
    print(f"Loaded {len(df)} rows.")

    start = time.time()
    df["pred"] = run_predictions(df)
    elapsed = time.time() - start

    valid_df = df[df["pred"] != -1]
    y_true = valid_df["label"].tolist()
    y_pred = valid_df["pred"].tolist()

    overall = compute_overall_metrics(y_true, y_pred)
    per_language = compute_per_language_metrics(df)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    print("\n=== Overall metrics ===")
    for k, v in overall.items():
        print(f"{k:>20}: {v:.4f}")

    print("\n=== Per-language breakdown ===")
    for lang, metrics in per_language.items():
        print(f"\n{lang} ({metrics['samples']} samples):")
        if metrics["accuracy"] is None:
            print("  no samples")
            continue
        print(f"  accuracy : {metrics['accuracy']:.4f}")
        print(f"  f1_macro : {metrics['f1_macro']:.4f}")

    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    print(f"          pred_Real  pred_Fake")
    print(f"true_Real  {cm[0][0]:>9}  {cm[0][1]:>9}")
    print(f"true_Fake  {cm[1][0]:>9}  {cm[1][1]:>9}")

    print(f"\nTotal samples evaluated: {len(valid_df)} (skipped: {len(df) - len(valid_df)})")
    print(f"Time taken: {elapsed:.2f}s")

    results = {
        "overall": overall,
        "per_language": per_language,
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": cm,
        },
        "total_samples": int(len(df)),
        "evaluated_samples": int(len(valid_df)),
        "skipped_samples": int(len(df) - len(valid_df)),
        "elapsed_seconds": round(elapsed, 2),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}.")


if __name__ == "__main__":
    main()
