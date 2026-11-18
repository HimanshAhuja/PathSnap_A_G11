"""Generate plausible directed road-arc candidates for GPS observations.

Input conventions follow the PathSnap dataset:
- nodes: N x 2 array/dataframe in [longitude, latitude] order
- arcs: M x 2 array/dataframe in [from_node_id, to_node_id] order
- track: T x 3 array/dataframe in [longitude, latitude, timestamp_seconds] order

Distances are calculated in a local UTM projection (metres), not in degrees.
Arc IDs are the zero-based row indices of the supplied .arcs file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree


ArrayLike = Union[np.ndarray, pd.DataFrame, Sequence]


@dataclass
class RoadNetworkIndex:
    """Projected directed road arcs and a spatial index over their geometries."""

    transformer: Transformer
    geometries: list
    arc_ids: np.ndarray
    lengths_m: np.ndarray
    bearings_deg: np.ndarray
    tree: STRtree
    # Shapely 1.x returns geometries; Shapely 2.x returns integer indices.
    geometry_id_to_position: dict


def _as_array(data: ArrayLike, expected_cols: int, name: str) -> np.ndarray:
    arr = data.to_numpy() if isinstance(data, pd.DataFrame) else np.asarray(data)
    if arr.ndim != 2 or arr.shape[1] < expected_cols:
        raise ValueError(f"{name} must be a 2-D array with at least {expected_cols} columns.")
    return np.asarray(arr[:, :expected_cols], dtype=float)


def make_local_transformer(nodes_lonlat: ArrayLike) -> Transformer:
    """Create a WGS84 -> local UTM transformer based on the network centroid."""
    nodes = _as_array(nodes_lonlat, 2, "nodes")
    valid = np.isfinite(nodes).all(axis=1)
    if not valid.any():
        raise ValueError("No valid node coordinates were supplied.")

    lon0 = float(np.median(nodes[valid, 0]))
    lat0 = float(np.median(nodes[valid, 1]))
    zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
    epsg = (32600 if lat0 >= 0 else 32700) + zone
    return Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)


def _bearing_from_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compass bearing in degrees clockwise from north, in [0, 360)."""
    dx, dy = x2 - x1, y2 - y1
    return float((np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0)


def build_road_network_index(
    nodes_lonlat: ArrayLike,
    arcs: ArrayLike,
    transformer: Optional[Transformer] = None,
) -> RoadNetworkIndex:
    """Build directed LineStrings and an STRtree for the .nodes/.arcs files."""
    nodes = _as_array(nodes_lonlat, 2, "nodes")
    arc_rows = _as_array(arcs, 2, "arcs")

    if not np.isfinite(arc_rows).all():
        raise ValueError("Arc table contains non-finite values.")
    if not np.equal(arc_rows, np.floor(arc_rows)).all():
        raise ValueError("Arc endpoints must be integer node IDs.")

    arc_rows = arc_rows.astype(np.int64)
    if (arc_rows < 0).any() or (arc_rows >= len(nodes)).any():
        raise ValueError("An arc references a node ID outside the .nodes table.")

    transformer = transformer or make_local_transformer(nodes)
    x, y = transformer.transform(nodes[:, 0], nodes[:, 1])
    projected_nodes = np.column_stack([x, y])

    geometries, arc_ids, lengths, bearings = [], [], [], []
    for arc_id, (start_id, end_id) in enumerate(arc_rows):
        start = projected_nodes[start_id]
        end = projected_nodes[end_id]
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            continue
        line = LineString([tuple(start), tuple(end)])
        length = float(line.length)
        if length <= 0:
            continue
        geometries.append(line)
        arc_ids.append(arc_id)
        lengths.append(length)
        bearings.append(_bearing_from_xy(*start, *end))

    if not geometries:
        raise ValueError("No valid non-zero-length road arcs could be constructed.")

    tree = STRtree(geometries)
    geometry_id_to_position = {id(g): i for i, g in enumerate(geometries)}

    return RoadNetworkIndex(
        transformer=transformer,
        geometries=geometries,
        arc_ids=np.asarray(arc_ids, dtype=np.int64),
        lengths_m=np.asarray(lengths, dtype=float),
        bearings_deg=np.asarray(bearings, dtype=float),
        tree=tree,
        geometry_id_to_position=geometry_id_to_position,
    )


def _tree_query_positions(index: RoadNetworkIndex, query_geom) -> list[int]:
    """Return geometry positions for both Shapely 1.x and 2.x STRtree APIs."""
    hits = index.tree.query(query_geom)
    positions = []
    for hit in hits:
        if isinstance(hit, (int, np.integer)):
            positions.append(int(hit))
        else:
            pos = index.geometry_id_to_position.get(id(hit))
            if pos is None:
                # Defensive fallback for geometry wrappers returned by some builds.
                for i, geom in enumerate(index.geometries):
                    if geom.equals(hit):
                        pos = i
                        break
            if pos is not None:
                positions.append(pos)
    return positions


def generate_candidates(
    track: ArrayLike,
    index: RoadNetworkIndex,
    search_radius_m: float = 40.0,
    max_candidates: int = 8,
) -> pd.DataFrame:
    """Find nearby directed road arcs for each GPS observation.

    Returns one row per GPS-observation/candidate-arc pair. An observation can
    have zero candidates if no road falls within the configured radius.
    """
    if search_radius_m <= 0:
        raise ValueError("search_radius_m must be positive.")
    if max_candidates < 1:
        raise ValueError("max_candidates must be at least 1.")

    gps = _as_array(track, 3, "track")
    if not np.isfinite(gps[:, :2]).all():
        raise ValueError("Track contains invalid longitude/latitude values.")

    gx, gy = index.transformer.transform(gps[:, 0], gps[:, 1])
    rows = []

    for gps_index, (x, y) in enumerate(zip(gx, gy)):
        point = Point(float(x), float(y))
        search_area = box(
            x - search_radius_m, y - search_radius_m,
            x + search_radius_m, y + search_radius_m
        )
        positions = _tree_query_positions(index, search_area)

        scored = []
        for pos in positions:
            line = index.geometries[pos]
            distance = float(line.distance(point))
            if distance <= search_radius_m:
                projected_distance = float(line.project(point))
                fraction = projected_distance / float(line.length)
                scored.append((distance, pos, fraction))

        scored.sort(key=lambda item: item[0])
        for distance, pos, fraction in scored[:max_candidates]:
            rows.append({
                "gps_index": int(gps_index),
                "candidate_arc_id": int(index.arc_ids[pos]),
                "distance_m": distance,
                "projection_fraction": float(fraction),
                "road_bearing_deg": float(index.bearings_deg[pos]),
                "segment_length_m": float(index.lengths_m[pos]),
            })

    columns = [
        "gps_index", "candidate_arc_id", "distance_m",
        "projection_fraction", "road_bearing_deg", "segment_length_m"
    ]
    return pd.DataFrame(rows, columns=columns)


def candidate_coverage(
    candidates: pd.DataFrame,
    reference_arc_ids_by_gps: dict[int, set[int]],
    number_of_observations: int,
) -> dict:
    """Measure reference-arc candidate coverage when a defensible GPS/route
    alignment is available.

    reference_arc_ids_by_gps maps gps_index -> set of acceptable reference arc
    IDs for that observation. This function does NOT infer the alignment.
    """
    if number_of_observations <= 0:
        raise ValueError("number_of_observations must be positive.")

    candidate_sets = (
        candidates.groupby("gps_index")["candidate_arc_id"]
        .apply(lambda s: set(map(int, s)))
        .to_dict()
    )
    evaluable = 0
    hits = 0
    for gps_index, reference_ids in reference_arc_ids_by_gps.items():
        if not reference_ids:
            continue
        evaluable += 1
        if candidate_sets.get(int(gps_index), set()).intersection(reference_ids):
            hits += 1

    return {
        "evaluable_observations": evaluable,
        "covered_observations": hits,
        "candidate_recall": (hits / evaluable) if evaluable else float("nan"),
        "candidate_miss_rate": (1 - hits / evaluable) if evaluable else float("nan"),
        "total_observations": int(number_of_observations),
    }
