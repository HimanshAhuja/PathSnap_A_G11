"""Extract motion and candidate-road features for PathSnap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer


def angular_difference_deg(a, b):
    """Smallest absolute angular difference in degrees (0..180)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def _bearing_from_xy(dx, dy):
    """Compass bearing, clockwise from north."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def extract_track_motion_features(
    track: pd.DataFrame,
    transformer: Transformer,
) -> pd.DataFrame:
    """Compute per-GPS-point speed, movement bearing and heading change.

    `track` must have longitude, latitude and timestamp columns, in that order.
    The returned dataframe preserves gps_index and uses metres/seconds.
    """
    arr = track.iloc[:, :3].to_numpy(dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("track must contain longitude, latitude and timestamp.")
    if not np.isfinite(arr).all():
        raise ValueError("track contains non-finite values.")

    x, y = transformer.transform(arr[:, 0], arr[:, 1])
    times = arr[:, 2]
    n = len(arr)

    dx = np.full(n, np.nan)
    dy = np.full(n, np.nan)
    dt = np.full(n, np.nan)

    if n > 1:
        dx[1:] = np.diff(x)
        dy[1:] = np.diff(y)
        dt[1:] = np.diff(times)

    distance = np.hypot(dx, dy)
    valid_step = (dt > 0) & np.isfinite(dt)
    speed = np.full(n, np.nan)
    speed[valid_step] = distance[valid_step] / dt[valid_step]

    movement_bearing = np.full(n, np.nan)
    moving = valid_step & (distance > 0.5)  # suppress unstable headings at near-zero movement
    movement_bearing[moving] = _bearing_from_xy(dx[moving], dy[moving])

    heading_change = np.full(n, np.nan)
    if n > 2:
        heading_change[2:] = angular_difference_deg(
            movement_bearing[2:], movement_bearing[1:-1]
        )

    return pd.DataFrame({
        "gps_index": np.arange(n, dtype=int),
        "speed_mps": speed,
        "gps_bearing_deg": movement_bearing,
        "heading_change_deg": heading_change,
        "timestamp_s": times,
    })


def attach_candidate_features(
    candidates: pd.DataFrame,
    track_motion: pd.DataFrame,
) -> pd.DataFrame:
    """Join per-point motion features to each candidate and derive bearing gap."""
    if candidates.empty:
        result = candidates.copy()
        for col in ("speed_mps", "gps_bearing_deg", "heading_change_deg",
                    "timestamp_s", "bearing_difference_deg"):
            result[col] = pd.Series(dtype=float)
        return result

    result = candidates.merge(track_motion, on="gps_index", how="left", validate="many_to_one")
    result["bearing_difference_deg"] = angular_difference_deg(
        result["gps_bearing_deg"].to_numpy(),
        result["road_bearing_deg"].to_numpy(),
    )
    return result


def build_candidate_feature_table(
    track: pd.DataFrame,
    candidates: pd.DataFrame,
    transformer: Transformer,
) -> pd.DataFrame:
    """Convenience function: compute motion features and attach to candidates."""
    motion = extract_track_motion_features(track, transformer)
    return attach_candidate_features(candidates, motion)
