"""
Integration adapter (owner: Himansh).

Calls the team loader (preprocessing.load_data.load_trip) and converts its
output into the agreed trip format, whichever of these the loader returns:
  - load_trip(split, trip_id)      or  load_trip(data_root, trip_id)
  - a dict                         or  a dataclass/object (e.g. TripData)
  - column names lon/lat/t, from_node/to_node, arc_id
    or longitude/latitude/time_value, source_id/target_id, identifier

Agreed output: dict(trip_id, split, track[lon, lat, t], nodes[lon, lat],
                    arcs[from_node, to_node] (int), route[arc_id] (int) or None)
"""
from __future__ import annotations

import inspect

import config

COLUMN_ALIASES = {
    "longitude": "lon", "latitude": "lat", "time_value": "t", "timestamp": "t",
    "source_id": "from_node", "target_id": "to_node",
    "identifier": "arc_id",
}
INTEGER_COLUMNS = {"arcs": ["from_node", "to_node"], "route": ["arc_id"]}


def _as_dict(raw) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    # dataclass / simple object: read its attributes without copying the tables
    return {k: getattr(raw, k) for k in ("trip_id", "split", "track", "nodes", "arcs", "route") if hasattr(raw, k)}


def normalise_trip(raw, split: str, trip_id: str) -> dict:
    trip = _as_dict(raw)
    for key in ("track", "nodes", "arcs", "route"):
        df = trip.get(key)
        if df is None:
            continue
        df = df.rename(columns=COLUMN_ALIASES)
        for col in INTEGER_COLUMNS.get(key, []):
            if col in df.columns:
                df[col] = df[col].astype("int64")   # ids index rows, so they must be integers
        trip[key] = df
    trip["trip_id"] = str(trip.get("trip_id", trip_id))
    trip["split"] = split
    return trip


def load_trip_normalised(load_trip, split: str, trip_id: str) -> dict:
    """Call the team's load_trip with whichever first argument it expects."""
    first_param = next(iter(inspect.signature(load_trip).parameters), "split")
    if first_param in ("data_root", "root", "data_dir", "partition_dir", "path"):
        raw = load_trip(str(config.DATA_ROOT / split), trip_id)
    else:
        raw = load_trip(split, trip_id)
    return normalise_trip(raw, split, trip_id)
