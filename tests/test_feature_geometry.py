"""Small synthetic tests for candidate generation and feature extraction."""
import numpy as np
import pandas as pd

from features.candidate_generation import (
    build_road_network_index,
    generate_candidates,
)
from features.feature_extraction import (
    extract_track_motion_features,
    build_candidate_feature_table,
)


def test_candidate_generation_returns_nearby_arc():
    # A short east-west road around the PathSnap dataset's approximate region.
    nodes = np.array([
        [8.6000, 49.8000],
        [8.6010, 49.8000],
    ])
    arcs = np.array([[0, 1]])
    index = build_road_network_index(nodes, arcs)

    track = pd.DataFrame([
        [8.6005, 49.80002, 1.0],
    ], columns=["longitude", "latitude", "timestamp"])
    candidates = generate_candidates(track, index, search_radius_m=50, max_candidates=5)

    assert len(candidates) == 1
    assert int(candidates.iloc[0]["candidate_arc_id"]) == 0
    assert candidates.iloc[0]["distance_m"] < 10


def test_feature_table_contains_motion_and_road_features():
    nodes = np.array([
        [8.6000, 49.8000],
        [8.6010, 49.8000],
    ])
    arcs = np.array([[0, 1]])
    index = build_road_network_index(nodes, arcs)

    track = pd.DataFrame([
        [8.6001, 49.8000, 1.0],
        [8.6002, 49.8000, 2.0],
        [8.6003, 49.8000, 3.0],
    ], columns=["longitude", "latitude", "timestamp"])
    candidates = generate_candidates(track, index, search_radius_m=50)
    features = build_candidate_feature_table(track, candidates, index.transformer)

    assert "distance_m" in features.columns
    assert "bearing_difference_deg" in features.columns
    assert "speed_mps" in features.columns
    assert len(features) > 0
