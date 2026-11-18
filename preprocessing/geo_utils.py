"""
Shared geometry helpers.

Why a per-trip local projection?
  The trips are spread across the world (e.g. ~2.9E/48.5N, ~35.5E/49.6N,
  ~106.7E/15.3N), so no single UTM zone fits all of them. Each trip covers at
  most a few tens of kilometres, so we project longitude/latitude onto a flat
  plane centred on that trip (equirectangular / local tangent plane):

      x = R * (lon - lon0) * cos(lat0)     (metres east)
      y = R * (lat - lat0)                 (metres north)

  Within a few tens of km of (lon0, lat0) the distance error is well below 1%,
  which is far smaller than typical GPS noise. The SAME origin must be used for
  a trip's GPS track and its road nodes.
"""
from __future__ import annotations
import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def trip_origin(lon, lat):
    """Projection origin for one trip: median of the GPS coordinates."""
    return float(np.nanmedian(lon)), float(np.nanmedian(lat))


def project(lon, lat, origin):
    """Convert lon/lat (degrees) to local x/y (metres) around origin=(lon0, lat0)."""
    lon0, lat0 = origin
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = EARTH_RADIUS_M * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = EARTH_RADIUS_M * np.radians(lat - lat0)
    return x, y


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (used to sanity-check the projection)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing_deg(dx, dy):
    """Compass bearing of a displacement (0 = north, 90 = east), in [0, 360)."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    """Smallest absolute difference between two bearings, in [0, 180]."""
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0
    return np.minimum(d, 360.0 - d)


def point_to_segment(px, py, ax, ay, bx, by):
    """Distance from point(s) P to segment(s) AB, plus the projection fraction t.

    All arguments broadcast. t in [0, 1]: 0 = at A (arc start), 1 = at B (arc end).
    Returns (distance_m, t).
    """
    abx, aby = bx - ax, by - ay
    seg_len2 = abx * abx + aby * aby
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * abx + (py - ay) * aby) / seg_len2
    t = np.where(seg_len2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    cx, cy = ax + t * abx, ay + t * aby
    return np.hypot(px - cx, py - cy), t
