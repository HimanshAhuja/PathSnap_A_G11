"""
Plots built ONLY from real run outputs (owner: Himansh).

    python -m evaluation.generate_plots models      # from results/model_comparison.csv
    python -m evaluation.generate_plots features --file results/features_train.parquet
    python -m evaluation.generate_plots importance  # from models/best_model.pkl

Figures are saved to results/plots/.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

METRIC_ALIASES = {   # first matching column name in model_comparison.csv is used
    "Macro F1 (CV)": ["cv_macro_f1", "cv_f1_macro", "cv_f1"],
    "Macro F1 (student test)": ["test_macro_f1", "macro_f1", "f1_macro", "f1"],
    "Top-1 accuracy": ["test_top1_accuracy", "top1_accuracy", "top1"],
    "Accuracy": ["test_accuracy", "accuracy"],
    "Log loss": ["test_log_loss", "log_loss", "logloss"],
    "Inference time (s)": ["inference_time_s", "inference_time"],
}


def _save(fig, name):
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path.relative_to(config.PROJECT_ROOT)}")


def plot_model_comparison(csv=config.RESULTS_DIR / "model_comparison.csv"):
    df = pd.read_csv(csv)
    name_col = next(c for c in ["model", "model_name", "name"] if c in df.columns)
    panels = [(title, next((c for c in cols if c in df.columns), None)) for title, cols in METRIC_ALIASES.items()]
    panels = [(t, c) for t, c in panels if c is not None and df[c].notna().any()]
    if not panels:
        raise ValueError(f"no known metric columns in {csv}: {list(df.columns)}")
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4), squeeze=False)
    for ax, (title, col) in zip(axes[0], panels):
        vals = df[col].astype(float)
        ax.barh(df[name_col], vals, color="#4C72B0")
        for y, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(v, y, f" {v:.3f}", va="center", fontsize=8)
        ax.set_title(title)
        ax.invert_yaxis()
    _save(fig, "model_comparison.png")


def plot_feature_distributions(path):
    feats = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    cols = [c for c in config.REQUIRED_FEATURES if c in feats.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.5), squeeze=False)
    for ax, c in zip(axes[0], cols):
        v = feats[c].replace([np.inf, -np.inf], np.nan)
        hi = v.quantile(0.99)
        for lab, colour, name in [(0, "#C44E52", "other candidates"), (1, "#55A868", "reference arc")]:
            ax.hist(v[(feats["label"] == lab)].clip(upper=hi).dropna(), bins=40, density=True,
                    alpha=0.55, color=colour, label=name)
        ax.set_title(c)
    axes[0][0].legend(fontsize=8)
    _save(fig, "feature_distributions.png")

    cpp = feats.groupby(["trip_id", "gps_index"]).size()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(cpp, bins=range(1, int(cpp.max()) + 2), color="#4C72B0")
    ax.set_xlabel("candidates per GPS observation")
    ax.set_ylabel("GPS observations")
    _save(fig, "candidates_per_point.png")


def plot_feature_importance():
    import joblib
    obj = joblib.load(config.BEST_MODEL_PATH)
    model = obj["model"] if isinstance(obj, dict) else obj
    cols = obj.get("feature_columns") if isinstance(obj, dict) else getattr(model, "feature_names_in_", None)
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        imp, label = est.feature_importances_, "importance"
    elif hasattr(est, "coef_"):
        imp, label = np.abs(est.coef_).ravel(), "|coefficient| (scaled features)"
    else:
        print("model exposes no feature importances — skipped")
        return
    cols = list(cols) if cols is not None else [f"f{i}" for i in range(len(imp))]
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(6, 0.35 * len(cols) + 1.5))
    ax.barh(np.array(cols)[order], imp[order], color="#4C72B0")
    ax.set_xlabel(label)
    ax.set_title(f"Feature importance — {type(est).__name__}")
    _save(fig, "feature_importance.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["models", "features", "importance"])
    ap.add_argument("--file", type=Path)
    a = ap.parse_args()
    if a.what == "models":
        plot_model_comparison(a.file or config.RESULTS_DIR / "model_comparison.csv")
    elif a.what == "features":
        plot_feature_distributions(a.file or config.RESULTS_DIR / "features_train.parquet")
    else:
        plot_feature_importance()
