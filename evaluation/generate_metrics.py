"""
Shared evaluation metrics (owner: Himansh — integration & evaluation).

All functions work on the agreed candidate table: one row per
(trip_id, gps_index, candidate_arc_id), with `label` = 1 for the reference arc.

Usage (summarise the model comparison table):
    python -m evaluation.generate_metrics
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR  # noqa: E402

POINT_KEYS = ["trip_id", "gps_index"]


def top1_predictions(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """For each GPS observation keep the candidate with the highest score."""
    df = df.reset_index(drop=True)   # concatenated trips repeat index labels; idxmax needs unique ones
    idx = df.groupby(POINT_KEYS, sort=False)[score_col].idxmax()
    return df.loc[idx].reset_index(drop=True)


def top1_accuracy(df: pd.DataFrame, score_col: str = "score") -> dict:
    """Observation-level Top-1 accuracy.

    - top1_accuracy_all: denominator = every GPS point that has candidates
      (a point whose reference arc was not generated counts as wrong).
    - top1_accuracy_covered: denominator = only points whose reference arc is
      among the candidates (measures the ranker alone, not candidate generation).
    """
    top = top1_predictions(df, score_col)
    has_pos = df.groupby(POINT_KEYS, sort=False)["label"].max().rename("has_pos")
    top = top.merge(has_pos, left_on=POINT_KEYS, right_index=True)
    correct = top["label"] == 1
    covered = top["has_pos"] == 1
    return {
        "n_points": int(len(top)),
        "top1_accuracy_all": float(correct.mean()) if len(top) else float("nan"),
        "top1_accuracy_covered": float(correct[covered].mean()) if covered.any() else float("nan"),
    }


def candidate_miss_rate(df: pd.DataFrame, n_gps_points: int | None = None) -> dict:
    """Share of GPS points whose reference arc is absent from the candidate set.

    If n_gps_points is given, points that received zero candidates are counted
    as misses too (they never appear in the candidate table).
    """
    per_point = df.groupby(POINT_KEYS, sort=False)["label"].max()
    n_with_cands = len(per_point)
    n_missing_label = int((per_point == 0).sum())
    n_total = n_gps_points if n_gps_points is not None else n_with_cands
    n_no_cands = max(n_total - n_with_cands, 0)
    return {
        "n_gps_points": int(n_total),
        "points_without_candidates": int(n_no_cands),
        "points_reference_not_in_candidates": n_missing_label,
        "candidate_miss_rate": (n_missing_label + n_no_cands) / n_total if n_total else float("nan"),
    }


def route_continuity(pred_arcs, arcs: pd.DataFrame) -> float:
    """Share of consecutive predicted-arc changes that are physically connected.

    A step arc_a -> arc_b counts as continuous if it stays on the same arc or if
    arc_a ends where arc_b starts (to_node(a) == from_node(b)). Direction matters
    because .arcs are directed.
    """
    pred = [a for a in pred_arcs if a is not None and not pd.isna(a)]
    changes = [(a, b) for a, b in zip(pred[:-1], pred[1:]) if a != b]
    if not changes:
        return 1.0
    to_node = arcs["to_node"].to_numpy()
    from_node = arcs["from_node"].to_numpy()
    ok = sum(to_node[int(a)] == from_node[int(b)] for a, b in changes)
    return ok / len(changes)


def summarise_comparison(path: Path = RESULTS_DIR / "model_comparison.csv") -> pd.DataFrame:
    """Load the model comparison table written by models/train_models.py and
    save a Markdown copy for the report."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run models/train_models.py first")
    df = pd.read_csv(path)
    md_path = path.with_suffix(".md")
    try:
        text = df.to_markdown(index=False, floatfmt=".4f")   # needs `tabulate`
    except ImportError:
        text = df.to_string(index=False)
    md_path.write_text(text)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Summarise results/model_comparison.csv")
    ap.add_argument("--file", type=Path, default=RESULTS_DIR / "model_comparison.csv")
    args = ap.parse_args()
    table = summarise_comparison(args.file)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(table.to_string(index=False))
    print(f"\nMarkdown copy written to {args.file.with_suffix('.md')}")
