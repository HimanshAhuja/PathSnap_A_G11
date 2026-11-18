"""
End-to-end pipeline validation (owner: Himansh — integration & evaluation).

Runs ONE real trip through every stage, logs row counts and timings, and runs
data-integrity checks. Stages whose module is not written yet are reported as
PENDING instead of crashing, so this script is useful from the first minute.

Interface contract each module must follow (see README "Interfaces"):
  preprocessing.load_data.load_trip(split, trip_id) -> dict
      {"trip_id", "split", "track": DataFrame[lon, lat, t],
       "nodes": DataFrame[lon, lat], "arcs": DataFrame[from_node, to_node],
       "route": DataFrame[arc_id] or None}
  preprocessing.clean_data.clean_track(track) -> (clean_track_df, report_dict)
      clean_track_df keeps a `gps_index` column = original row number
  features.build_training_data.build_trip_features(trip) -> DataFrame
      columns >= config.REQUIRED_COLUMNS, one row per (gps point, candidate arc)
  models/best_model.pkl  (joblib) -> fitted sklearn estimator/Pipeline with predict_proba
      (feature names read from feature_names_in_), or {"model": ..., "feature_columns": [...]}

Usage (from the project root):
    python -m evaluation.validate_pipeline --split train --trip 00000046
    python -m evaluation.validate_pipeline --features-file results/training_data.csv
"""
from __future__ import annotations
import argparse
import importlib
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from evaluation.generate_metrics import candidate_miss_rate, top1_accuracy  # noqa: E402
from evaluation.load_adapter import load_trip_normalised  # noqa: E402


class Report:
    """Collects stage results and check outcomes, prints them, saves JSON."""

    def __init__(self):
        self.stages, self.checks = [], []

    def stage(self, name, status, seconds=None, rows=None, note=""):
        self.stages.append(dict(stage=name, status=status, seconds=seconds, rows=rows, note=note))
        t = f"{seconds:6.2f}s" if seconds is not None else "      -"
        r = f"{rows:>9,}" if rows is not None else "        -"
        print(f"  [{status:<7}] {name:<22} {t}  rows={r}  {note}")

    def check(self, name, passed, detail=""):
        self.checks.append(dict(check=name, passed=bool(passed), detail=detail))
        print(f"    {'PASS' if passed else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))

    def warn(self, name, ok, detail=""):
        """Expected data issue: reported and logged, but does not fail the run."""
        self.checks.append(dict(check=name, passed=True, warning=not ok, detail=detail))
        print(f"    {'PASS' if ok else 'WARN'}  {name}" + (f"  — {detail}" if detail else ""))

    @property
    def ok(self):
        return all(c["passed"] for c in self.checks) and all(s["status"] != "ERROR" for s in self.stages)

    def save(self, tag):
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.LOGS_DIR / f"validate_{tag}_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(dict(stages=self.stages, checks=self.checks, ok=self.ok), indent=2, default=str))
        return path


def _optional_import(module, func):
    """Return module.func, or None if the module/function does not exist yet."""
    try:
        return getattr(importlib.import_module(module), func)
    except ModuleNotFoundError as e:
        if e.name == module or module.startswith(f"{e.name}."):
            return None   # the team member's file simply does not exist yet
        raise             # a real missing dependency inside their file: surface it
    except AttributeError:
        return None


def _run(rep, name, fn, *args):
    t0 = time.perf_counter()
    try:
        out = fn(*args)
        return out, time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        rep.stage(name, "ERROR", time.perf_counter() - t0, note=f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_trip(rep, trip):
    print("  Raw-data checks")
    track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip.get("route")
    rep.check("track has lon/lat/t columns", {"lon", "lat", "t"} <= set(track.columns), str(list(track.columns)))
    finite = np.isfinite(track[["lon", "lat", "t"]].to_numpy()).all(axis=1)
    rep.check("track values finite", finite.all(), f"{(~finite).sum()} non-finite rows")
    in_range = track["lon"].between(-180, 180) & track["lat"].between(-90, 90)
    rep.check("track lon/lat in valid range", in_range.all(), f"{(~in_range).sum()} out-of-range rows")
    n_nodes = len(nodes)
    bad_arc = ~(arcs["from_node"].between(0, n_nodes - 1) & arcs["to_node"].between(0, n_nodes - 1))
    rep.check("arc endpoints index valid nodes", not bad_arc.any(), f"{bad_arc.sum()} bad arcs of {len(arcs)}")
    if route is not None:
        bad_route = ~route["arc_id"].between(0, len(arcs) - 1)
        rep.check("route arc_ids index valid arcs", not bad_route.any(), f"{bad_route.sum()} bad of {len(route)}")
        r = route["arc_id"].to_numpy()
        if len(r) > 1 and not bad_route.any():
            joined = arcs["to_node"].to_numpy()[r[:-1]] == arcs["from_node"].to_numpy()[r[1:]]
            rep.check("consecutive route arcs connected", joined.all(),
                      f"{joined.mean():.1%} of {len(joined)} steps connected")
    dt = np.diff(track["t"].to_numpy())
    rep.warn("timestamps strictly increasing", (dt > 0).all(),
              f"{(dt <= 0).sum()} non-increasing steps; median gap {np.median(dt) if len(dt) else 0:.0f}s, "
              f"max gap {dt.max() if len(dt) else 0:.0f}s")


def check_features(rep, feats, n_arcs=None, n_gps=None):
    print("  Feature-table checks")
    missing = [c for c in config.REQUIRED_COLUMNS if c not in feats.columns]
    rep.check("required schema columns present", not missing, f"missing: {missing}" if missing else "")
    if missing:
        return
    fcols = config.feature_columns(feats)
    leaked = [c for c in fcols if c in config.ID_COLUMNS]
    rep.check("no ID columns used as features", not leaked, str(leaked) if leaked else f"{len(fcols)} features: {fcols}")
    vals = feats[fcols].to_numpy(dtype=float)
    bad = ~np.isfinite(vals)
    per_col = {c: int(n) for c, n in zip(fcols, bad.sum(axis=0)) if n}
    rep.check("no NaN/inf in features", not bad.any(), str(per_col) if per_col else "")
    dup = feats.duplicated(subset=config.ID_COLUMNS).sum()
    rep.check("no duplicate (trip, gps_index, arc) rows", dup == 0, f"{dup} duplicates")
    rep.check("labels are 0/1", feats["label"].isin([0, 1]).all(), str(sorted(int(v) for v in feats["label"].unique())[:5]))
    pos = feats.groupby(["trip_id", "gps_index"])["label"].sum()
    rep.check("at most one positive per GPS point", (pos <= 1).all(), f"{(pos > 1).sum()} points with >1 positive")
    rep.check("distance >= 0 (metres)", (feats["distance"] >= 0).all(),
              f"median {feats['distance'].median():.1f} m, max {feats['distance'].max():.1f} m")
    rep.check("bearing_difference within [0,180]", feats["bearing_difference"].between(0, 180).all())
    if n_arcs is not None:
        bad_ids = ~feats["candidate_arc_id"].between(0, n_arcs - 1)
        rep.check("candidate_arc_id indexes a real arc", not bad_ids.any(), f"{bad_ids.sum()} invalid")
    if n_gps is not None:
        bad_gps = ~feats["gps_index"].between(0, n_gps - 1)
        rep.check("gps_index within track length", not bad_gps.any(), f"{bad_gps.sum()} invalid")
    cpp = feats.groupby(["trip_id", "gps_index"]).size()
    miss = candidate_miss_rate(feats, n_gps)
    rep.warn("candidate sets non-empty", miss["points_without_candidates"] == 0,
             f"{miss['points_without_candidates']} GPS points with zero candidates (GPS outliers beyond "
             f"the search radius — counted in the miss rate, handled later by the HMM); "
              f"candidates/point median {cpp.median():.0f}, max {cpp.max()}")
    print(f"    INFO  candidate miss rate = {miss['candidate_miss_rate']:.2%} "
          f"(reference arc absent for {miss['points_reference_not_in_candidates']} points)")
    print(f"    INFO  label balance = {feats['label'].mean():.2%} positive rows")


def run_model(rep, feats):
    if not config.BEST_MODEL_PATH.exists() or config.BEST_MODEL_PATH.stat().st_size == 0:
        rep.stage("model prediction", "PENDING", note=f"{config.BEST_MODEL_PATH.name} not trained/saved yet")
        return
    import joblib
    t0 = time.perf_counter()
    obj = joblib.load(config.BEST_MODEL_PATH)
    model = obj["model"] if isinstance(obj, dict) else obj
    cols = obj.get("feature_columns") if isinstance(obj, dict) else None
    if not cols and hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)   # sklearn models fitted on a DataFrame remember their columns
    cols = cols or config.feature_columns(feats)
    scores = model.predict_proba(feats[cols])[:, 1]
    secs = time.perf_counter() - t0
    scored = feats.assign(score=scores)
    m = top1_accuracy(scored)
    rep.stage("model prediction", "OK", secs, len(scored),
              f"Top-1 (all points) {m['top1_accuracy_all']:.3f}, "
              f"Top-1 (covered) {m['top1_accuracy_covered']:.3f}")


# ---------------------------------------------------------------------------
def validate_trip(split, trip_id):
    rep = Report()
    print(f"\nPathSnap pipeline validation — {split}/{trip_id}  (data root: {config.DATA_ROOT})")
    print("Stages")

    load_trip = _optional_import("preprocessing.load_data", "load_trip")
    if load_trip is None:
        rep.stage("load_trip", "PENDING", note="preprocessing/load_data.py: load_trip() not written yet")
        return rep
    trip, secs = _run(rep, "load_trip", load_trip_normalised, load_trip, split, trip_id)
    if trip is None:
        return rep
    rep.stage("load_trip", "OK", secs, len(trip["track"]),
              f"nodes={len(trip['nodes']):,} arcs={len(trip['arcs']):,} "
              f"route={len(trip['route']) if trip.get('route') is not None else 'n/a'}")
    check_trip(rep, trip)

    clean_track = _optional_import("preprocessing.clean_data", "clean_track")
    if clean_track is None:
        rep.stage("clean_track", "PENDING", note="preprocessing/clean_data.py: clean_track() not written yet")
    else:
        out, secs = _run(rep, "clean_track", clean_track, trip["track"])
        if out is not None:
            cleaned, cleaning_report = out
            rep.stage("clean_track", "OK", secs, len(cleaned),
                      f"removed {len(trip['track']) - len(cleaned)} rows; {cleaning_report}")
            trip["track_clean"] = cleaned

    build = _optional_import("features.build_training_data", "build_trip_features")
    if build is None:
        rep.stage("build_trip_features", "PENDING", note="features/build_training_data.py not written yet")
        return rep
    feats, secs = _run(rep, "build_trip_features", build, trip)
    if feats is None:
        return rep
    rep.stage("build_trip_features", "OK", secs, len(feats),
              f"{feats['gps_index'].nunique() if 'gps_index' in feats else '?'} GPS points with candidates")
    check_features(rep, feats, n_arcs=len(trip["arcs"]), n_gps=len(trip["track"]))
    run_model(rep, feats)
    return rep


def validate_features_file(path):
    rep = Report()
    print(f"\nValidating saved feature table {path}")
    t0 = time.perf_counter()
    feats = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    rep.stage("read feature table", "OK", time.perf_counter() - t0, len(feats),
              f"{feats['trip_id'].nunique() if 'trip_id' in feats else '?'} trips")
    check_features(rep, feats)
    run_model(rep, feats)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=config.SPLITS)
    ap.add_argument("--trip", default="00000046", help="trip id, e.g. 00000046")
    ap.add_argument("--features-file", type=Path, help="validate a saved feature table instead")
    args = ap.parse_args()

    report = validate_features_file(args.features_file) if args.features_file else validate_trip(args.split, args.trip)
    log = report.save(args.features_file.stem if args.features_file else f"{args.split}_{args.trip}")
    n_fail = sum(not c["passed"] for c in report.checks)
    pending = [s["stage"] for s in report.stages if s["status"] == "PENDING"]
    if not report.ok:
        verdict = f"FAILED — {n_fail} check(s) failed / stage errors present"
    elif pending:
        verdict = f"INCOMPLETE — passed so far, pending stages: {pending}"
    else:
        verdict = "ALL STAGES RAN AND ALL CHECKS PASSED"
    print(f"\nResult: {verdict}")
    print(f"Log saved to {log.relative_to(config.PROJECT_ROOT)}")
    sys.exit(0 if report.ok else 1)
