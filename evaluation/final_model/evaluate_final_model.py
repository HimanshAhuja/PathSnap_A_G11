"""Run the FINAL PathSnap model (frozen, no retraining, no tuning) on a set of raw trips and report metrics.

Final model = average of three probability sources, decoded with the continuity-aware HMM:
  1. tuned LightGBM classifier            results/final_model.joblib
  2. LightGBM ranker (softmax per point)  results/lgbm_ranker.joblib
  3. stage-2 context LightGBM             results/stage2_model.joblib (uses stage-1 = model 1 probabilities)
  HMM/Viterbi parameters                  results/final_hmm_params.json (tuned on training trips only)

Usage: python evaluate_final_model.py <candidate_csv> <raw_split_dir> <results_dir> <out_dir> <name>
"""
import os, sys, json, time, glob
from collections import defaultdict
import numpy as np, pandas as pd, joblib
from sklearn.metrics import accuracy_score

csv_path, raw_dir, res_dir, out_dir, name = sys.argv[1:6]
os.makedirs(out_dir, exist_ok=True)
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nb_funcs.py")).read())  # notebook functions, unchanged

def load_raw_dir(d0):
    out = {}
    for d in sorted(glob.glob(os.path.join(d0, "*"))):
        if not os.path.isdir(d): continue
        tid = os.path.basename(d)
        arcs = np.loadtxt(os.path.join(d, f"{tid}.arcs"), dtype=np.int64, ndmin=2)
        nodes = np.loadtxt(os.path.join(d, f"{tid}.nodes"), dtype=float, ndmin=2)
        track = np.loadtxt(os.path.join(d, f"{tid}.track"), dtype=float, ndmin=2)
        route = np.loadtxt(os.path.join(d, f"{tid}.route"), dtype=np.int64, ndmin=1)
        lon0, lat0 = np.median(track[:, 0]), np.median(track[:, 1])
        kx = 6371000.0 * np.pi / 180 * np.cos(np.radians(lat0)); ky = 6371000.0 * np.pi / 180
        out[tid.zfill(8)] = dict(frm=arcs[:, 0], to=arcs[:, 1], t=track[:, 2], n_gps=len(track), route=route,
                                 P=np.c_[(track[:, 0] - lon0) * kx, (track[:, 1] - lat0) * ky],
                                 N=np.c_[(nodes[:, 0] - lon0) * kx, (nodes[:, 1] - lat0) * ky])
    return out

t0 = time.perf_counter()
RAW = {"test": load_raw_dir(raw_dir)}
test_raw = load_table(csv_path)
test = add_geometry_features(add_features(test_raw, RAW["test"]), RAW["test"])
assert len(test) == len(test_raw)

# ---- the three frozen probability sources
clf = joblib.load(os.path.join(res_dir, "final_model.joblib"))
p1 = clf.predict_proba(test[FEATURES])[:, 1]
ranker = joblib.load(os.path.join(res_dir, "lgbm_ranker.joblib"))
def softmax_by_point(df, s):
    d = pd.DataFrame({"t": df.trip_id.values, "g": df.gps_index.values, "s": s})
    e = np.exp(d.s - d.groupby(["t", "g"]).s.transform("max"))
    return (e / e.groupby([d.t, d.g]).transform("sum")).to_numpy()
p_r = softmax_by_point(test, ranker.predict(test[FEATURES]))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "context_features.py")).read())
s2 = joblib.load(os.path.join(res_dir, "stage2_model.joblib"))
ctx = context_features(test, p1, "test")
test2 = pd.concat([test, ctx.add_prefix("s1_")], axis=1)
p2 = s2["model"].predict_proba(test2[s2["features"]])[:, 1]
p_c = np.mean([p1, p_r, p2], axis=0)
hp = json.load(open(os.path.join(res_dir, "final_hmm_params.json")))

final = decode(prepare_trips(test, p_c, "test"), hp)
clf_only = top1_choice(test, p_c)
nearest = top1_choice(test, -test["distance"].to_numpy())
route_sets = {t: set(RAW["test"][t]["route"].tolist()) for t in RAW["test"]}
def rep(method, ch):
    return {"method": method, "top1_accuracy": ch[TARGET].mean(), "route_continuity": route_continuity(ch, "test"),
            "route_arc_accuracy": np.mean([a in route_sets[t] for t, a in zip(ch.trip_id, ch.candidate_arc_id)])}
res = pd.DataFrame([rep("nearest candidate (no ML)", nearest), rep("combined classifiers only (no HMM)", clf_only),
                    rep("FINAL: combined classifiers + continuity-aware HMM", final)])
res["row_level_accuracy"] = [np.nan, accuracy_score(test[TARGET], (p_c >= 0.5).astype(int)), np.nan]
# candidate miss rate: GPS points whose true arc was not among the generated candidates (not in the table)
n_gps = sum(r["n_gps"] for r in RAW["test"].values()); n_pts = test.groupby(POINT).ngroups
per_trip = final.groupby("trip_id").agg(points=(TARGET, "size"), top1=(TARGET, "mean")).reset_index()
per_trip["route_arc_accuracy"] = [np.mean([a in route_sets[t] for a in g.candidate_arc_id]) for t, g in final.groupby("trip_id")]
per_trip["route_continuity"] = [route_continuity(g, "test") for _, g in final.groupby("trip_id")]
print(f"[{name}] trips={test.trip_id.nunique()} candidate rows={len(test):,} labelled GPS points={n_pts:,} "
      f"of {n_gps:,} GPS points ({1 - n_pts / n_gps:.2%} without the true arc among candidates)")
print(res.round(4).to_string(index=False)); print(per_trip.round(4).to_string(index=False))
print(f"total time {time.perf_counter() - t0:.1f}s")
res.to_csv(os.path.join(out_dir, f"{name}_results.csv"), index=False)
per_trip.to_csv(os.path.join(out_dir, f"{name}_per_trip.csv"), index=False)
final.to_csv(os.path.join(out_dir, f"{name}_predictions.csv"), index=False)
