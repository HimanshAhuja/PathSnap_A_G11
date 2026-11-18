"""Folium map of one trip: GPS points, nearby road network, reference route and the final model's predicted route.

Usage: python evaluation/plot_prediction_map.py <raw_trip_dir> <predictions_csv> <out_html>
predictions_csv = output of evaluation/final_model/evaluate_final_model.py (trip_id, gps_index, candidate_arc_id, label)
"""
import os, sys
import numpy as np, pandas as pd, folium

trip_dir, pred_csv, out_html = sys.argv[1:4]
tid = os.path.basename(os.path.normpath(trip_dir))
track = np.loadtxt(os.path.join(trip_dir, f"{tid}.track"), ndmin=2)
nodes = np.loadtxt(os.path.join(trip_dir, f"{tid}.nodes"), ndmin=2)
arcs = np.loadtxt(os.path.join(trip_dir, f"{tid}.arcs"), dtype=np.int64, ndmin=2)
route = np.loadtxt(os.path.join(trip_dir, f"{tid}.route"), dtype=np.int64, ndmin=1)
pred = pd.read_csv(pred_csv, dtype={"trip_id": str})
pred = pred[pred.trip_id.str.zfill(8) == tid].sort_values("gps_index")

def line(a):  # arc -> [(lat, lon), (lat, lon)]
    f, t = arcs[a]
    return [(nodes[f, 1], nodes[f, 0]), (nodes[t, 1], nodes[t, 0])]

lat0, lat1 = track[:, 1].min() - 0.002, track[:, 1].max() + 0.002
lon0, lon1 = track[:, 0].min() - 0.002, track[:, 0].max() + 0.002
m = folium.Map(location=[track[:, 1].mean(), track[:, 0].mean()], zoom_start=15, tiles="OpenStreetMap", control_scale=True)
m.fit_bounds([[lat0, lon0], [lat1, lon1]])

net = folium.FeatureGroup(name="Road network (arcs near the trip)")
inside = ((nodes[arcs[:, 0], 1] > lat0) & (nodes[arcs[:, 0], 1] < lat1) & (nodes[arcs[:, 0], 0] > lon0) & (nodes[arcs[:, 0], 0] < lon1))
for a in np.where(inside)[0]:
    folium.PolyLine(line(a), color="#9e9e9e", weight=1.5, opacity=0.6).add_to(net)
net.add_to(m)

ref = folium.FeatureGroup(name="Reference route (ground truth)")
for a in route:
    folium.PolyLine(line(a), color="#1f77b4", weight=7, opacity=0.55).add_to(ref)
ref.add_to(m)

prd = folium.FeatureGroup(name="Predicted route (final model + HMM)")
for a in pred.candidate_arc_id.unique():
    folium.PolyLine(line(int(a)), color="#ff7f0e", weight=3, opacity=0.95).add_to(prd)
prd.add_to(m)

gps = folium.FeatureGroup(name="GPS observations")
for i, (lon, lat, t) in enumerate(track):
    folium.CircleMarker((lat, lon), radius=2, color="#212121", fill=True, fill_opacity=0.8, weight=0,
                        tooltip=f"gps_index {i}, t={t:.0f}s").add_to(gps)
gps.add_to(m)

wrong = folium.FeatureGroup(name="Points matched to a wrong arc")
for r in pred[pred.label == 0].itertuples():
    lon, lat, _ = track[r.gps_index]
    folium.CircleMarker((lat, lon), radius=4, color="#d62728", fill=True, fill_opacity=0.9, weight=0,
                        tooltip=f"gps_index {r.gps_index}: predicted arc {r.candidate_arc_id}").add_to(wrong)
wrong.add_to(m)

acc = pred.label.mean()
legend = f"""<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;padding:8px 10px;border:1px solid #999;font:12px sans-serif">
<b>Trip {tid}</b> &middot; Top-1 {acc:.1%} ({len(pred)} points)<br>
<span style="color:#1f77b4">&#9644;</span> Reference route &nbsp; <span style="color:#ff7f0e">&#9644;</span> Predicted route<br>
<span style="color:#212121">&#9679;</span> GPS point &nbsp; <span style="color:#d62728">&#9679;</span> wrong arc &nbsp; <span style="color:#9e9e9e">&#9644;</span> road network</div>"""
m.get_root().html.add_child(folium.Element(legend))
folium.LayerControl(collapsed=False).add_to(m)
m.save(out_html)
print("saved", out_html, f"top1={acc:.4f}")
