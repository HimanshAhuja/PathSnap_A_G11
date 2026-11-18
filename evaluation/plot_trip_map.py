"""
Interactive Folium map of one trip (owner: Himansh).

Layers: road network (grey), raw GPS observations (blue), reference route
(green) and — optionally — the candidate arcs of one GPS observation (orange,
reference candidate highlighted) taken from a saved feature table.

    python -m evaluation.plot_trip_map --split train --trip 00000046
    python -m evaluation.plot_trip_map --split train --trip 00000046 \
        --features-file results/training_data.csv --gps-index 120

Output: results/plots/map_<split>_<trip>.html
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from branca.element import Element

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def read_trip(split, trip_id):
    """Use the team loader if it exists; otherwise read the four raw files directly."""
    try:
        from preprocessing.load_data import load_trip
        from evaluation.load_adapter import load_trip_normalised
        return load_trip_normalised(load_trip, split, trip_id)
    except ImportError:
        d = config.DATA_ROOT / split / trip_id
        rd = lambda ext, names: pd.read_csv(d / f"{trip_id}.{ext}", sep=r"\s+", header=None, names=names)
        route_path = d / f"{trip_id}.route"
        return dict(trip_id=trip_id, split=split,
                    track=rd("track", ["lon", "lat", "t"]),
                    nodes=rd("nodes", ["lon", "lat"]),
                    arcs=rd("arcs", ["from_node", "to_node"]),
                    route=rd("route", ["arc_id"]) if route_path.exists() else None)


def arc_lines(arc_ids, arcs, nodes):
    """[[(lat, lon), (lat, lon)], ...] for the given arc ids (Folium wants lat first)."""
    a = arcs.loc[arc_ids]
    f, t = nodes.loc[a["from_node"]].to_numpy(), nodes.loc[a["to_node"]].to_numpy()
    return [[(fl[1], fl[0]), (tl[1], tl[0])] for fl, tl in zip(f, t)]


def build_map(trip, features=None, gps_index=None, margin_deg=0.003):
    track, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip.get("route")
    lon_min, lon_max = track["lon"].min() - margin_deg, track["lon"].max() + margin_deg
    lat_min, lat_max = track["lat"].min() - margin_deg, track["lat"].max() + margin_deg

    m = folium.Map(location=[track["lat"].median(), track["lon"].median()], zoom_start=14,
                   tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

    # Road network: only arcs touching the trip's bounding box (keeps big trips light)
    inside = nodes["lon"].between(lon_min, lon_max) & nodes["lat"].between(lat_min, lat_max)
    keep = arcs.index[inside.to_numpy()[arcs["from_node"]] | inside.to_numpy()[arcs["to_node"]]]
    roads = folium.FeatureGroup(name=f"Road network ({len(keep):,} arcs shown)")
    if len(keep):
        folium.PolyLine(arc_lines(keep, arcs, nodes), color="#888888", weight=2, opacity=0.6).add_to(roads)
    roads.add_to(m)

    if route is not None:
        ref = folium.FeatureGroup(name="Reference route (.route)")
        folium.PolyLine(arc_lines(route["arc_id"].to_numpy(), arcs, nodes),
                        color="#2CA02C", weight=6, opacity=0.8).add_to(ref)
        ref.add_to(m)

    gps = folium.FeatureGroup(name=f"Raw GPS observations ({len(track):,})")
    step = max(1, len(track) // 3000)   # thin very long tracks for the browser
    for i, r in track.iloc[::step].iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=2.5, color="#1F77B4", fill=True,
                            fill_opacity=0.8, weight=0, tooltip=f"gps_index {i}, t={r['t']:.0f}s").add_to(gps)
    gps.add_to(m)

    if features is not None and gps_index is not None:
        rows = features[(features["trip_id"].astype(str) == str(trip["trip_id"])) & (features["gps_index"] == gps_index)]
        cand = folium.FeatureGroup(name=f"Candidates of gps_index {gps_index} ({len(rows)})")
        for _, r in rows.iterrows():
            is_ref = r.get("label", 0) == 1
            folium.PolyLine(arc_lines([int(r["candidate_arc_id"])], arcs, nodes),
                            color="#D62728" if is_ref else "#FF7F0E", weight=7 if is_ref else 4,
                            tooltip=f"arc {int(r['candidate_arc_id'])}: {r['distance']:.1f} m"
                                    + (" (reference)" if is_ref else "")).add_to(cand)
        p = track.loc[gps_index]
        folium.Marker([p["lat"], p["lon"]], tooltip=f"GPS observation {gps_index}",
                      icon=folium.Icon(color="red", icon="map-marker")).add_to(cand)
        cand.add_to(m)

    legend = """
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white;
                padding: 8px 12px; border: 1px solid #999; border-radius: 4px; font-size: 13px;">
      <b>Trip {tid}</b><br>
      <span style="color:#888888">&#9473;&#9473;</span> Road network<br>
      <span style="color:#2CA02C">&#9473;&#9473;</span> Reference route<br>
      <span style="color:#1F77B4">&#9679;</span> Raw GPS<br>
      <span style="color:#FF7F0E">&#9473;&#9473;</span> Candidate arcs
      <span style="color:#D62728">&#9473;&#9473;</span> reference candidate
    </div>""".format(tid=trip["trip_id"])
    m.get_root().html.add_child(Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=config.SPLITS)
    ap.add_argument("--trip", default="00000046")
    ap.add_argument("--features-file", type=Path)
    ap.add_argument("--gps-index", type=int)
    a = ap.parse_args()
    trip = read_trip(a.split, a.trip)
    feats = None
    if a.features_file:
        feats = pd.read_parquet(a.features_file) if str(a.features_file).endswith(".parquet") else pd.read_csv(a.features_file)
    out = config.PLOTS_DIR / f"map_{a.split}_{a.trip}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    build_map(trip, feats, a.gps_index).save(out)
    print(f"saved {out.relative_to(config.PROJECT_ROOT)}")
