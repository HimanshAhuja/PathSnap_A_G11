"""PathSnap inference pipeline for ONE trip (used by the FastAPI server).

Prediction uses ONLY .track, .nodes and .arcs. The reference .route is read afterwards,
server-side, to score the prediction; it is never passed to the models or returned to the UI.
"""
from __future__ import annotations
import json, logging, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features.build_training_data import (FEATURE_RENAME, align_candidates_to_route,  # noqa: E402
                                          generate_trip_features, load_trip_data)

log = logging.getLogger("pathsnap")
FM = ROOT / "evaluation" / "final_model"
exec(open(FM / "nb_funcs.py").read())          # notebook feature, metric and HMM functions, unchanged
exec(open(FM / "context_features.py").read())  # stage-2 context features, unchanged
RAW = {"test": {}}                              # read by prepare_trips / route_continuity / context_features
RES = ROOT / "results"


class FinalModel:
    def __init__(self):
        t0 = time.perf_counter()
        self.clf = joblib.load(RES / "final_model.joblib")
        self.ranker = joblib.load(RES / "lgbm_ranker.joblib")
        s2 = joblib.load(RES / "stage2_model.joblib")
        self.s2, self.s2_features = s2["model"], s2["features"]
        self.hp = json.load(open(RES / "final_hmm_params.json"))
        log.info("Loaded final model (LightGBM + ranker + stage-2) and HMM parameters in %.1fs", time.perf_counter() - t0)

    @staticmethod
    def _softmax_by_point(df, s):
        d = pd.DataFrame({"t": df.trip_id.values, "g": df.gps_index.values, "s": s})
        e = np.exp(d.s - d.groupby(["t", "g"]).s.transform("max"))
        return (e / e.groupby([d.t, d.g]).transform("sum")).to_numpy()

    def run(self, trip_dir: Path) -> dict:
        timings, t_all = {}, time.perf_counter()
        tid = trip_dir.name

        def step(name, t0):
            timings[name] = round(time.perf_counter() - t0, 3)
            log.info("[%s] %-34s %.2fs", tid, name, timings[name])

        # 1. load raw files and keep the reference route aside (scoring only)
        t0 = time.perf_counter()
        trip = load_trip_data(trip_dir)
        route = trip.pop("route")
        track = trip["track"].to_numpy()
        lon0, lat0 = np.median(track[:, 0]), np.median(track[:, 1])
        kx = 6371000.0 * np.pi / 180 * np.cos(np.radians(lat0)); ky = 6371000.0 * np.pi / 180
        nodes = trip["nodes"]
        RAW["test"] = {tid: dict(frm=trip["arcs"][:, 0], to=trip["arcs"][:, 1], t=track[:, 2], n_gps=len(track),
                                 P=np.c_[(track[:, 0] - lon0) * kx, (track[:, 1] - lat0) * ky],
                                 N=np.c_[(nodes[:, 0] - lon0) * kx, (nodes[:, 1] - lat0) * ky])}
        step("load raw trip", t0)

        # 2. candidate generation + base features (no route)
        t0 = time.perf_counter()
        feats, index = generate_trip_features(trip)
        df = feats.rename(columns=FEATURE_RENAME)[META + BASE_FEATURES].copy()
        df["trip_id"] = tid
        df[TARGET] = 0                                  # placeholder: the models never see labels
        df = df.sort_values(["trip_id", "gps_index", "candidate_arc_id"]).reset_index(drop=True)
        step("candidates + base features", t0)

        # 3. engineered features
        t0 = time.perf_counter()
        df = add_geometry_features(add_features(df, RAW["test"]), RAW["test"])
        step("engineered features (32)", t0)

        # 4. three probability sources, averaged
        t0 = time.perf_counter()
        p1 = self.clf.predict_proba(df[FEATURES])[:, 1]
        pr = self._softmax_by_point(df, self.ranker.predict(df[FEATURES]))
        ctx = context_features(df, p1, "test")
        p2 = self.s2.predict_proba(pd.concat([df, ctx.add_prefix("s1_")], axis=1)[self.s2_features])[:, 1]
        p = np.mean([p1, pr, p2], axis=0)
        step("classifiers (3 models)", t0)

        # 5. HMM / Viterbi
        t0 = time.perf_counter()
        pred = decode(prepare_trips(df, p, "test"), self.hp)[["gps_index", "candidate_arc_id"]]
        clf_only = top1_choice(df, p)[["gps_index", "candidate_arc_id"]]
        step("HMM / Viterbi decoding", t0)

        # 6. scoring against the server-side reference (after prediction)
        t0 = time.perf_counter()
        trip["route"] = route
        labelled, _ = align_candidates_to_route(feats, trip, index)
        truth = labelled.loc[labelled.label == 1, ["gps_index", "candidate_arc_id"]].rename(columns={"candidate_arc_id": "true_arc"})
        route_set = set(route.tolist())
        def score(ch):
            m = ch.merge(truth, on="gps_index", how="inner")
            c = ch.assign(trip_id=tid)
            return {"top1_accuracy": float((m.candidate_arc_id == m.true_arc).mean()) if len(m) else None,
                    "route_continuity": float(route_continuity(c, "test")),
                    "route_arc_accuracy": float(np.mean([a in route_set for a in ch.candidate_arc_id]))}
        metrics = {"final_model_with_hmm": score(pred), "classifiers_only": score(clf_only),
                   "gps_points": int(len(track)), "points_with_candidates": int(df.gps_index.nunique()),
                   "evaluated_points": int(len(truth)),
                   "candidate_miss_rate": float(1 - len(truth) / len(track))}
        step("scoring vs reference (server-side)", t0)

        # 7. response: GPS track + predicted route geometry (no reference route, no labels)
        arcs_in_order = pred.candidate_arc_id.loc[pred.candidate_arc_id.ne(pred.candidate_arc_id.shift())].tolist()
        A = trip["arcs"]
        route_lines = [[[float(nodes[A[a, 0], 1]), float(nodes[A[a, 0], 0])], [float(nodes[A[a, 1], 1]), float(nodes[A[a, 1], 0])]]
                       for a in arcs_in_order]
        timings["total"] = round(time.perf_counter() - t_all, 3)
        log.info("[%s] DONE  top1=%.4f  continuity=%.4f  route-arc=%.4f  total %.2fs", tid,
                 metrics["final_model_with_hmm"]["top1_accuracy"], metrics["final_model_with_hmm"]["route_continuity"],
                 metrics["final_model_with_hmm"]["route_arc_accuracy"], timings["total"])
        return {"trip_id": tid, "metrics": metrics, "timings_s": timings,
                "gps": [[float(la), float(lo)] for lo, la in track[:, :2]],
                "predicted_arcs": [{"gps_index": int(g), "arc_id": int(a)} for g, a in zip(pred.gps_index, pred.candidate_arc_id)],
                "predicted_route": route_lines}
