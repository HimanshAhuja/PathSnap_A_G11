"""
Shared project configuration: data paths and the agreed feature-table schema.

Every module should import paths and column names from here instead of
hard-coding them, so the whole team stays on the same interface.
"""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _find_data_root() -> Path:
    """Locate the folder that contains train/ and student_test/.

    Order of preference:
      1. PATHSNAP_DATA environment variable
      2. <repo>/data
      3. ../Student folder (repo placed next to the supplied materials)
    """
    candidates = []
    if os.environ.get("PATHSNAP_DATA"):
        candidates.append(Path(os.environ["PATHSNAP_DATA"]).expanduser())
    candidates.append(PROJECT_ROOT / "data")
    candidates.append(PROJECT_ROOT.parent / "Student folder")
    for c in candidates:
        # must contain real trip folders, not just a placeholder train/ directory
        if (c / "train").is_dir() and any(p.is_dir() for p in (c / "train").iterdir()):
            return c
    # Fall back to the first option so error messages show a sensible path.
    return candidates[0]


DATA_ROOT = _find_data_root()
SPLITS = ("train", "student_test")   # evaluator-testing data is never stored here

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Agreed feature-table schema (team work plan, section 5)
# One row = one (GPS observation, candidate arc) pair.
# ---------------------------------------------------------------------------
ID_COLUMNS = ["trip_id", "gps_index", "candidate_arc_id"]   # metadata, never model inputs
LABEL_COLUMN = "label"                                      # 1 = candidate is the reference arc
REQUIRED_FEATURES = [
    "distance",             # GPS-to-candidate distance, metres
    "bearing_difference",   # wrapped |movement bearing - arc bearing|, degrees [0, 180]
    "speed",                # m/s estimated from neighbouring observations
    "heading_change",       # change in GPS movement heading, degrees [0, 180]
    "segment_length",       # candidate arc length, metres
]
REQUIRED_COLUMNS = ID_COLUMNS + REQUIRED_FEATURES + [LABEL_COLUMN]


def feature_columns(df):
    """Model input columns = every numeric column that is not an ID or the label.

    Lets the feature owner add extra features without the model code changing.
    """
    excluded = set(ID_COLUMNS) | {LABEL_COLUMN, "split"}
    return [c for c in df.columns
            if c not in excluded and df[c].dtype.kind in "biuf"]
