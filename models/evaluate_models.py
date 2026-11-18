"""
models/evaluate_models.py
Schema constants and metric functions shared by training and evaluation.

Positive class = the candidate arc IS the correct road (label == 1).
Classification metrics use average="binary" (positive class only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ----------------------------------------------------------------------------
# Agreed feature-table schema (see work plan, section 5). If Amrit changes a
# column name, change it HERE only.
# ----------------------------------------------------------------------------
TARGET = "label"
META_COLS = ["trip_id", "gps_index", "candidate_arc_id"]  # never model inputs
#
# Column names follow Amrit's features/candidate_generation.py and
# features/feature_extraction.py. Units: metres, degrees, seconds.
FEATURES = [
    "distance_m",              # GPS point -> candidate arc, metres
    "bearing_difference_deg",  # wrapped angle gap 0..180; NaN when car is ~stationary
    "speed_mps",               # NaN where the time step is invalid
    "heading_change_deg",      # NaN for the first two points of a trip
    "segment_length_m",        # candidate arc length, metres
]
# Optional extra, enabled with --with-projection (agree with Amrit first).
OPTIONAL_FEATURES = ["projection_fraction"]  # where along the arc the GPS point projects (0..1)
REQUIRED_COLUMNS = META_COLS + FEATURES + [TARGET]

THRESHOLD = 0.5  # decision threshold for precision/recall/F1/accuracy


def top1_matching_accuracy(meta: pd.DataFrame, y_true, proba) -> tuple[float, int]:
    """
    For each GPS observation, pick the candidate with the highest predicted
    probability and check whether it is the true one.

    Only observations that actually have a positive candidate are counted
    (if the correct arc was never retrieved, no model could pick it; that is a
    candidate-generation miss, not a model error).

    Returns (accuracy, number_of_observations_counted).
    """
    d = meta[["trip_id", "gps_index"]].reset_index(drop=True).copy()
    d["y"] = np.asarray(y_true)
    d["p"] = np.asarray(proba)
    keys = ["trip_id", "gps_index"]

    has_pos = d.groupby(keys)["y"].transform("max") == 1
    d = d[has_pos]
    if d.empty:
        return float("nan"), 0

    best = d.groupby(keys)["p"].idxmax()
    return float(d.loc[best, "y"].mean()), int(len(best))


def compute_metrics(y_true, proba, meta: pd.DataFrame, threshold: float = THRESHOLD) -> dict:
    """Metrics for one validation fold."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-15, 1 - 1e-15)
    pred = (proba >= threshold).astype(int)

    top1, n_obs = top1_matching_accuracy(meta, y_true, proba)
    both_classes = len(np.unique(y_true)) == 2

    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, average="binary", pos_label=1, zero_division=0),
        "recall": recall_score(y_true, pred, average="binary", pos_label=1, zero_division=0),
        "f1": f1_score(y_true, pred, average="binary", pos_label=1, zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "roc_auc": roc_auc_score(y_true, proba) if both_classes else float("nan"),
        "top1_accuracy": top1,
        "top1_n_obs": n_obs,
        "val_rows": int(len(y_true)),
        "val_positive_rate": float(y_true.mean()),
    }