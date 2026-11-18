"""
models/train_models.py
Trains and compares five candidate-arc classifiers with trip-wise validation.

Input: the labelled feature table from features/build_training_data.py, one row
per (GPS observation, candidate arc), with columns:
    trip_id, gps_index, candidate_arc_id,
    distance_m, bearing_difference_deg, speed_mps, heading_change_deg,
    segment_length_m, label   (+ optional projection_fraction)

    python -m models.train_models --data results/training_data.csv
    python -m models.train_models --data results/training_data.csv --subset-trips 20
    python -m models.train_models --data results/training_data.csv --with-projection
    python -m models.train_models --selftest        # code test on SYNTHETIC data only

Run from the project root (PathSnap/).

Leakage protection:
  * GroupKFold with groups = trip_id (no trip appears in both train and val)
  * trip_id / gps_index / candidate_arc_id are never passed to the model
  * absolute compass bearings and timestamps are not features
  * imputer and scaler live inside the Pipeline, so they are fitted on the
    training fold only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from models.evaluate_models import (FEATURES, META_COLS, OPTIONAL_FEATURES,
                                        TARGET, compute_metrics)
except ImportError:  # running as `python models/train_models.py`
    from evaluate_models import (FEATURES, META_COLS, OPTIONAL_FEATURES,
                                 TARGET, compute_metrics)

SEED = 42
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "log_loss", "roc_auc", "top1_accuracy"]


# ----------------------------------------------------------------------------
# Data loading and validation
# ----------------------------------------------------------------------------
def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[error] feature table not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def validate_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Fail clearly on schema problems; do not silently fix real data issues."""
    required = META_COLS + features + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"[error] missing columns vs agreed schema: {missing}\n"
                 f"        columns found: {list(df.columns)}")

    df = df.copy()
    bad_labels = set(df[TARGET].dropna().unique()) - {0, 1, True, False}
    if bad_labels or df[TARGET].isna().any():
        sys.exit(f"[error] label must be 0/1 with no NaN. Found invalid: {bad_labels}, "
                 f"NaN count: {int(df[TARGET].isna().sum())}")
    df[TARGET] = df[TARGET].astype(int)

    # inf is invalid; convert to NaN (counted and reported) so the imputer handles it
    feats = df[features].apply(pd.to_numeric, errors="coerce")
    n_inf = int(np.isinf(feats.to_numpy(dtype=float)).sum())
    if n_inf:
        print(f"[warn] {n_inf} non-finite feature values found; treated as NaN and imputed")
        feats = feats.replace([np.inf, -np.inf], np.nan)
    df[features] = feats

    print("[info] NaN fraction per feature (expected: bearing NaN when stationary, "
          "heading_change NaN at trip start):")
    print(df[features].isna().mean().round(4).to_string())

    dup = int(df.duplicated(subset=META_COLS).sum())
    if dup:
        print(f"[warn] {dup} duplicate (trip_id, gps_index, candidate_arc_id) rows")

    if df[TARGET].nunique() < 2:
        sys.exit("[error] label has a single class; cannot train a classifier")

    # How many GPS points even have a correct candidate? (upper bound for top-1)
    per_obs = df.groupby(["trip_id", "gps_index"])[TARGET].max()
    print(f"[info] GPS observations: {len(per_obs):,}; with a true candidate: "
          f"{per_obs.mean():.1%} (top-1 accuracy is measured on these only)")
    return df


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def build_models() -> dict[str, Pipeline]:
    def imp():
        # median fill + a 0/1 "was missing" flag, so the model can tell
        # "bearing unknown (car stationary)" from a real value
        return SimpleImputer(strategy="median", add_indicator=True)

    models: dict[str, Pipeline] = {
        "logistic_regression": Pipeline([
            ("imputer", imp()), ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)),
        ]),
        "knn": Pipeline([
            ("imputer", imp()), ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
        ]),
        "decision_tree": Pipeline([
            ("imputer", imp()),
            ("clf", DecisionTreeClassifier(max_depth=8, min_samples_leaf=20,
                                           class_weight="balanced", random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("imputer", imp()),
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                           class_weight="balanced", n_jobs=-1, random_state=SEED)),
        ]),
    }

    try:
        from xgboost import XGBClassifier
        models["xgboost"] = Pipeline([
            ("imputer", imp()),
            ("clf", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                  eval_metric="logloss", n_jobs=-1, random_state=SEED,
                                  scale_pos_weight=1.0)),  # reset per fold from the train fold
        ])
    except ImportError:
        print("[warn] xgboost not installed; using sklearn HistGradientBoosting as the "
              "documented boosting alternative (model name: hist_gradient_boosting)")
        models["hist_gradient_boosting"] = Pipeline([
            ("imputer", imp()),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                   class_weight="balanced", random_state=SEED)),
        ])
    return models


def config_string(model: Pipeline) -> str:
    return json.dumps(model.named_steps["clf"].get_params(), default=str, sort_keys=True)


def set_imbalance(model: Pipeline, y_train: pd.Series) -> None:
    """XGBoost has no class_weight; set scale_pos_weight = negatives / positives."""
    if "clf__scale_pos_weight" in model.get_params():
        pos = int(y_train.sum())
        model.set_params(clf__scale_pos_weight=(len(y_train) - pos) / max(pos, 1))


# ----------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------
def run_cv(df: pd.DataFrame, models: dict[str, Pipeline], n_splits: int, features: list[str]):
    X, y, groups = df[features], df[TARGET], df["trip_id"]
    gkf = GroupKFold(n_splits=n_splits)
    rows = []

    for name, template in models.items():
        print(f"\n=== {name} ===")
        for fold, (tr, va) in enumerate(gkf.split(X, y, groups=groups)):
            # hard guarantee: no trip on both sides
            assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[va])), "trip leakage across split!"

            y_tr = y.iloc[tr]
            if y_tr.nunique() < 2:
                print(f"  fold {fold}: train fold has a single class, skipped")
                continue

            model = clone(template)
            set_imbalance(model, y_tr)

            t0 = time.perf_counter()
            model.fit(X.iloc[tr], y_tr)
            fit_s = time.perf_counter() - t0

            proba = model.predict_proba(X.iloc[va])[:, 1]
            m = compute_metrics(y.iloc[va], proba, df.iloc[va][META_COLS])
            m.update(model=name, fold=fold, train_time_s=fit_s,
                     train_trips=int(groups.iloc[tr].nunique()),
                     val_trips=int(groups.iloc[va].nunique()))
            rows.append(m)
            print(f"  fold {fold}: f1={m['f1']:.3f} recall={m['recall']:.3f} "
                  f"logloss={m['log_loss']:.3f} top1={m['top1_accuracy']:.3f} ({fit_s:.1f}s)")
    return pd.DataFrame(rows)


def summarise(folds: pd.DataFrame, models: dict[str, Pipeline], scope: str,
              n_rows: int, n_trips: int, n_splits: int, features: list[str]) -> pd.DataFrame:
    g = folds.groupby("model")
    out = g[METRIC_COLS + ["train_time_s"]].mean()
    std = g[METRIC_COLS].std().add_suffix("_std")
    out = out.join(std).reset_index()
    out["n_folds"] = out["model"].map(g.size())
    out["n_rows"] = n_rows
    out["n_trips"] = n_trips
    out["features"] = ",".join(features)
    out["cv"] = f"GroupKFold(n_splits={n_splits}, groups=trip_id)"
    out["metric_averaging"] = "binary (positive class = correct arc); mean over folds"
    out["scope"] = scope
    out["config"] = out["model"].map(lambda m: config_string(models[m]))
    return out


# ----------------------------------------------------------------------------
# Synthetic data (CODE TEST ONLY; never report these numbers)
# ----------------------------------------------------------------------------
def make_synthetic(n_trips: int = 12, n_gps: int = 40, n_cand: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for t in range(n_trips):
        for g in range(n_gps):
            true_k = rng.integers(n_cand)
            stationary = rng.random() < 0.1  # bearing undefined when ~stationary
            for k in range(n_cand):
                is_true = int(k == true_k)
                rows.append({
                    "trip_id": f"synthetic_{t}", "gps_index": g,
                    "candidate_arc_id": t * 1000 + g * 10 + k,
                    "distance_m": abs(rng.normal(5 if is_true else 25, 8)),
                    "bearing_difference_deg": np.nan if (stationary or g == 0)
                    else abs(rng.normal(10 if is_true else 60, 25)),
                    "speed_mps": np.nan if g == 0 else abs(rng.normal(8, 3)),
                    "heading_change_deg": np.nan if g < 2 else abs(rng.normal(0, 15)),
                    "segment_length_m": abs(rng.normal(80, 30)),
                    "projection_fraction": rng.random(),
                    "label": is_true,
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("results/training_data.csv"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--model-dir", type=Path, default=Path("models"))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--subset-trips", type=int, default=None,
                    help="use only the first N trips; results are labelled SUBSET")
    ap.add_argument("--with-projection", action="store_true",
                    help="also use projection_fraction as a feature")
    ap.add_argument("--select-by", default="top1_accuracy",
                    choices=["top1_accuracy", "f1", "recall", "log_loss"])
    ap.add_argument("--selftest", action="store_true",
                    help="run on SYNTHETIC data into results/_selftest; never overwrites real outputs")
    args = ap.parse_args()

    features = FEATURES + (OPTIONAL_FEATURES if args.with_projection else [])

    if args.selftest:
        print("*** SELF-TEST: synthetic data, code check only. Do NOT report these numbers. ***")
        df = make_synthetic()
        scope = "SYNTHETIC_CODE_TEST_DO_NOT_REPORT"
        args.results_dir = args.results_dir / "_selftest"
        args.model_dir = args.results_dir / "model"
    else:
        df = load_table(args.data)
        scope = "full"

    df = validate_table(df, features)

    if args.subset_trips and not args.selftest:
        keep = sorted(df["trip_id"].unique())[: args.subset_trips]
        df = df[df["trip_id"].isin(keep)]
        scope = f"SUBSET ({len(keep)} trips)"

    n_trips = df["trip_id"].nunique()
    if n_trips < 2:
        sys.exit("[error] need at least 2 trips for trip-wise validation")
    n_splits = min(args.n_splits, n_trips)
    print(f"\n[info] rows={len(df):,} trips={n_trips} positive_rate={df[TARGET].mean():.3f} "
          f"splits={n_splits} scope={scope}")
    print(f"[info] features: {features}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    models = build_models()
    folds = run_cv(df, models, n_splits, features)
    if folds.empty:
        sys.exit("[error] no folds completed")

    table = summarise(folds, models, scope, len(df), n_trips, n_splits, features)
    table["timestamp"] = datetime.now().isoformat(timespec="seconds")

    # choose provisional best
    asc = args.select_by == "log_loss"
    best_row = table.sort_values(args.select_by, ascending=asc, na_position="last").iloc[0]
    best_name = best_row["model"]
    table["selected"] = table["model"] == best_name
    table = table.sort_values(args.select_by, ascending=asc).reset_index(drop=True)

    table.to_csv(args.results_dir / "model_comparison.csv", index=False)
    folds.to_csv(args.results_dir / "model_comparison_per_fold.csv", index=False)

    # refit the selected model on all rows and save it
    final = clone(models[best_name])
    set_imbalance(final, df[TARGET])
    final.fit(df[features], df[TARGET])
    if not hasattr(final, "predict_proba"):
        sys.exit("[error] selected model has no predict_proba; HMM emission needs probabilities")

    joblib.dump(final, args.model_dir / "best_model.pkl")
    meta = {
        "model": best_name, "features": features, "target": TARGET,
        "threshold": 0.5, "selected_by": args.select_by, "scope": scope,
        "cv": f"GroupKFold(n_splits={n_splits}, groups=trip_id)",
        "metrics_cv_mean": {m: float(best_row[m]) for m in METRIC_COLS},
        "config": json.loads(config_string(final)),
        "created": table["timestamp"].iloc[0],
        "usage": "model.predict_proba(df[features])[:, 1]  # probability the arc is correct",
    }
    (args.model_dir / "best_model_features.json").write_text(json.dumps(meta, indent=2))

    # append-only experiment log
    with open(args.results_dir / "experiment_log.jsonl", "a") as f:
        f.write(json.dumps({"timestamp": meta["created"], "scope": scope, "n_rows": len(df),
                            "n_trips": n_trips, "features": features, "cv": meta["cv"],
                            "selected": best_name,
                            "summary": table[["model"] + METRIC_COLS].round(4).to_dict("records")}) + "\n")

    show = ["model", "f1", "precision", "recall", "log_loss", "top1_accuracy", "train_time_s", "selected"]
    print("\n=== comparison (mean over folds) ===")
    print(table[show].round(4).to_string(index=False))
    print(f"\n[done] selected: {best_name} (by {args.select_by})")
    print(f"[done] wrote {args.results_dir/'model_comparison.csv'}, {args.model_dir/'best_model.pkl'}, "
          f"{args.model_dir/'best_model_features.json'}")


if __name__ == "__main__":
    main()