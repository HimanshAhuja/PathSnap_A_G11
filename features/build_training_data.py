"""Build labelled PathSnap candidate-level datasets.

Usage (from the repository root):
    python -m features.build_training_data
    python -m features.build_training_data --split train --limit 5
    python -m features.build_training_data --split all --limit 5

The supplied .route file is used ONLY to create supervision labels. It is never
included as a model feature. GPS observations without a generated candidate
that occurs in the reference route are counted as candidate misses and omitted
from the supervised CSV (they are not incorrectly labelled all-negative).

IMPORTANT: Labels are inferred by sequence-aligning generated candidates to
reference-route occurrences. They are not independently verified ground truth.
Inspect the alignment diagnostics and validate several trips before training.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from features.candidate_generation import (
    build_road_network_index,
    generate_candidates,
)
from features.feature_extraction import build_candidate_feature_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_RADIUS_M = 40.0
DEFAULT_MAX_CANDIDATES = 8

META_COLUMNS = ["trip_id", "gps_index", "candidate_arc_id"]
FEATURE_RENAME = {
    "distance_m": "distance",
    "bearing_difference_deg": "bearing_difference",
    "speed_mps": "speed",
    "heading_change_deg": "heading_change",
    "segment_length_m": "segment_length",
}
MODEL_FEATURE_COLUMNS = [
    "distance",
    "bearing_difference",
    "speed",
    "heading_change",
    "segment_length",
]
OUTPUT_COLUMNS = META_COLUMNS + MODEL_FEATURE_COLUMNS + ["label"]


def find_data_root() -> Path:
    """Resolve dataset root from PATHSNAP_DATA or common project locations."""
    configured = os.environ.get("PATHSNAP_DATA")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            PROJECT_ROOT / "data",
            PROJECT_ROOT.parent / "Student folder",
            PROJECT_ROOT.parent / "data",
        ]
    )
    for candidate in candidates:
        if (candidate / "train").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find dataset root containing a 'train' directory. "
        "Put the dataset in <project>/data or set PATHSNAP_DATA."
    )


def _load_numeric_file(path: Path, expected_columns: int | None = None) -> np.ndarray:
    """Load a whitespace-separated numeric file without a header."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    values = np.loadtxt(path, dtype=float, ndmin=2)
    if expected_columns is not None and values.shape[1] != expected_columns:
        raise ValueError(
            f"{path.name} must have {expected_columns} columns; "
            f"found {values.shape[1]}."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains non-finite values.")
    return values


def load_trip_data(trip_dir: Path) -> dict[str, Any]:
    """Load one trip's .track, .nodes, .arcs and .route files."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    track_values = _load_numeric_file(trip_dir / f"{trip_id}.track", 3)
    nodes = _load_numeric_file(trip_dir / f"{trip_id}.nodes", 2)
    arcs_values = _load_numeric_file(trip_dir / f"{trip_id}.arcs", 2)
    route_values = _load_numeric_file(trip_dir / f"{trip_id}.route")

    if route_values.shape[1] != 1:
        raise ValueError(f"{trip_id}.route must contain exactly one column.")
    if not np.equal(arcs_values, np.floor(arcs_values)).all():
        raise ValueError(f"{trip_id}.arcs contains non-integer node IDs.")
    if not np.equal(route_values, np.floor(route_values)).all():
        raise ValueError(f"{trip_id}.route contains non-integer arc IDs.")
    if (arcs_values < 0).any() or (arcs_values >= len(nodes)).any():
        raise ValueError(f"{trip_id}.arcs contains invalid node references.")

    arcs = arcs_values.astype(np.int64)
    route = route_values[:, 0].astype(np.int64)
    if len(route) == 0:
        raise ValueError(f"{trip_id}.route is empty.")
    if (route < 0).any() or (route >= len(arcs)).any():
        raise ValueError(f"{trip_id}.route contains invalid arc IDs.")
    if len(track_values) == 0:
        raise ValueError(f"{trip_id}.track is empty.")
    if len(track_values) > 1 and np.any(np.diff(track_values[:, 2]) < 0):
        raise ValueError(f"{trip_id}.track timestamps are not chronological.")

    track = pd.DataFrame(
        track_values,
        columns=["longitude", "latitude", "timestamp_s"],
    )
    return {
        "trip_id": trip_id,
        "trip_dir": trip_dir,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }


def generate_trip_features(
    trip: dict[str, Any],
    search_radius_m: float = DEFAULT_SEARCH_RADIUS_M,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[pd.DataFrame, Any]:
    """Generate candidate-level geometry and motion features for one trip."""
    index = build_road_network_index(trip["nodes"], trip["arcs"])
    candidates = generate_candidates(
        trip["track"],
        index,
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
    )
    if candidates.empty:
        return pd.DataFrame(), index

    feature_table = build_candidate_feature_table(
        trip["track"], candidates, index.transformer
    )
    feature_table.insert(0, "trip_id", trip["trip_id"])
    return feature_table, index


def _project_track(track: pd.DataFrame, transformer) -> np.ndarray:
    x, y = transformer.transform(
        track["longitude"].to_numpy(dtype=float),
        track["latitude"].to_numpy(dtype=float),
    )
    return np.column_stack((x, y))


def align_candidates_to_route(
    features: pd.DataFrame,
    trip: dict[str, Any],
    index,
    distance_scale_m: float = 12.0,
    transition_scale_m: float = 20.0,
    backward_tolerance_m: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Infer one route-consistent positive candidate per alignable GPS point.

    Viterbi-style dynamic programming is performed over *route occurrences*.
    Repeated arc IDs are supported: the same candidate arc can correspond to
    more than one position in the ordered reference route.

    Emissions penalize GPS-to-road distance. Transitions enforce non-decreasing
    cumulative route progress and weakly compare route distance with straight-
    line GPS displacement. The transition model is intentionally conservative
    because GPS sampling intervals can be irregular and displacement is only a
    lower bound on travelled road distance.
    """
    if features.empty:
        diagnostics = {
            "aligned_observations": 0,
            "candidate_misses": len(trip["track"]),
            "candidate_coverage_pct": 0.0,
            "alignment_method": "viterbi_route_occurrence",
            "selected_arc_ids": {},
            "candidate_miss_gps_indices": list(range(len(trip["track"]))),
            "final_path_cost": None,
        }
        empty = features.copy()
        empty["label"] = pd.Series(dtype="int64")
        return empty, diagnostics

    route = np.asarray(trip["route"], dtype=np.int64)
    arc_lengths_by_id = {
        int(arc_id): float(length)
        for arc_id, length in zip(index.arc_ids, index.lengths_m)
    }
    missing_route_arcs = sorted(set(map(int, route)) - set(arc_lengths_by_id))
    if missing_route_arcs:
        raise ValueError(
            f"Reference route contains {len(missing_route_arcs)} arcs that "
            "could not be represented in the road index; examples: "
            f"{missing_route_arcs[:10]}"
        )

    route_lengths = np.array(
        [arc_lengths_by_id[int(arc_id)] for arc_id in route], dtype=float
    )
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(route_lengths)))

    # Preserve every occurrence of an arc in the reference sequence.
    occurrences: dict[int, list[int]] = defaultdict(list)
    for position, arc_id in enumerate(route):
        occurrences[int(arc_id)].append(position)

    gps_xy = _project_track(trip["track"], index.transformer)
    states: list[dict[str, Any]] = []
    candidate_misses: list[int] = []

    for gps_index in range(len(trip["track"])):
        group = features.loc[features["gps_index"] == gps_index]
        expanded_states = []
        for row_index, row in group.iterrows():
            arc_id = int(row["candidate_arc_id"])
            for route_position in occurrences.get(arc_id, []):
                fraction = float(row["projection_fraction"])
                fraction = float(np.clip(fraction, 0.0, 1.0))
                progress = (
                    cumulative_lengths[route_position]
                    + fraction * route_lengths[route_position]
                )
                expanded_states.append(
                    {
                        "row_index": row_index,
                        "arc_id": arc_id,
                        "route_position": route_position,
                        "progress": progress,
                        "distance": float(row["distance_m"]),
                    }
                )

        if not expanded_states:
            candidate_misses.append(gps_index)
            continue

        progress = np.array([s["progress"] for s in expanded_states])
        distances = np.array([s["distance"] for s in expanded_states])
        # Gaussian-like distance emission; 12 m is a scale, not a hard cutoff.
        emission = 0.5 * (distances / distance_scale_m) ** 2
        states.append(
            {
                "gps_index": gps_index,
                "states": expanded_states,
                "progress": progress,
                "emission": emission,
            }
        )

    if not states:
        labelled = features.iloc[0:0].copy()
        labelled["label"] = pd.Series(dtype="int64")
        diagnostics = {
            "aligned_observations": 0,
            "candidate_misses": len(trip["track"]),
            "candidate_coverage_pct": 0.0,
            "alignment_method": "viterbi_route_occurrence",
            "selected_arc_ids": {},
            "candidate_miss_gps_indices": list(range(len(trip["track"]))),
            "final_path_cost": None,
        }
        return labelled, diagnostics

    dp_costs: list[np.ndarray] = []
    backpointers: list[np.ndarray] = []

    for t, observation in enumerate(states):
        current_progress = observation["progress"]
        current_emission = observation["emission"]
        current_cost = np.full(len(current_progress), np.inf, dtype=float)
        current_back = np.full(len(current_progress), -1, dtype=np.int64)

        if t == 0:
            # The route normally starts near the first GPS observation.
            start_cost = 0.5 * (current_progress / 100.0) ** 2
            current_cost = current_emission + start_cost
        else:
            previous = states[t - 1]
            previous_progress = previous["progress"]
            previous_cost = dp_costs[t - 1]
            previous_gps = previous["gps_index"]
            current_gps = observation["gps_index"]
            gps_displacement = float(
                np.linalg.norm(gps_xy[current_gps] - gps_xy[previous_gps])
            )
            elapsed = float(
                trip["track"].iloc[current_gps]["timestamp_s"]
                - trip["track"].iloc[previous_gps]["timestamp_s"]
            )
            # A weak movement prior: allow detours and irregular sampling.
            scale = max(
                transition_scale_m,
                0.5 * gps_displacement + 5.0,
                1.5 * max(elapsed, 0.0),
            )

            for j, progress_j in enumerate(current_progress):
                route_delta = progress_j - previous_progress
                transition = 0.5 * (
                    (route_delta - gps_displacement) / scale
                ) ** 2
                # Do not make the path infeasible when noisy GPS or a poor
                # candidate set suggests a small reversal. Penalize backward
                # progress strongly, while preserving a complete path.
                backward_amount = np.maximum(
                    0.0, -route_delta - backward_tolerance_m
                )
                transition += np.where(
                    backward_amount > 0.0,
                    4.0 + 0.5 * (backward_amount / 5.0) ** 2,
                    0.0,
                )
                total = previous_cost + transition
                best_previous = int(np.argmin(total))
                if np.isfinite(total[best_previous]):
                    current_cost[j] = (
                        total[best_previous] + current_emission[j]
                    )
                    current_back[j] = best_previous

        if not np.isfinite(current_cost).any():
            raise ValueError(
                "No feasible route sequence at GPS observation "
                f"{observation['gps_index']}. Consider inspecting this trip's "
                "reference route/candidate coverage before creating labels."
            )
        dp_costs.append(current_cost)
        backpointers.append(current_back)

    final_state = int(np.argmin(dp_costs[-1]))
    final_cost = float(dp_costs[-1][final_state])
    selected_states = [final_state]
    for t in range(len(states) - 1, 0, -1):
        previous_state = int(backpointers[t][selected_states[-1]])
        if previous_state < 0:
            raise RuntimeError(
                f"Broken Viterbi backpointer at GPS {states[t]['gps_index']}"
            )
        selected_states.append(previous_state)
    selected_states.reverse()

    selected_arc_ids: dict[int, int] = {}
    selected_rows: dict[int, Any] = {}
    for observation, state_number in zip(states, selected_states):
        state = observation["states"][state_number]
        gps_index = observation["gps_index"]
        selected_arc_ids[gps_index] = int(state["arc_id"])
        selected_rows[gps_index] = state["row_index"]

    # Only GPS observations with a selected route candidate enter supervised
    # learning. Every other generated candidate at those points is negative.
    labelled = features.loc[
        features["gps_index"].isin(selected_rows)
    ].copy()
    labelled["label"] = 0
    for gps_index, row_index in selected_rows.items():
        arc_id = selected_arc_ids[gps_index]
        positive = (
            (labelled["gps_index"] == gps_index)
            & (labelled["candidate_arc_id"] == arc_id)
        )
        labelled.loc[positive, "label"] = 1

    aligned_count = len(selected_rows)
    diagnostics = {
        "aligned_observations": aligned_count,
        "candidate_misses": len(trip["track"]) - aligned_count,
        "candidate_coverage_pct": 100.0 * aligned_count / len(trip["track"]),
        "alignment_method": "viterbi_route_occurrence",
        "selected_arc_ids": selected_arc_ids,
        "candidate_miss_gps_indices": candidate_misses,
        "final_path_cost": final_cost,
    }
    return labelled, diagnostics


def process_trip(
    trip_dir: Path,
    search_radius_m: float = DEFAULT_SEARCH_RADIUS_M,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, featurize, align and label one trip."""
    trip = load_trip_data(Path(trip_dir))
    features, index = generate_trip_features(
        trip,
        search_radius_m=search_radius_m,
        max_candidates=max_candidates,
    )
    labelled, diagnostics = align_candidates_to_route(features, trip, index)

    if not labelled.empty:
        labelled = labelled.rename(columns=FEATURE_RENAME)
        labelled["trip_id"] = trip["trip_id"]
        labelled["gps_index"] = labelled["gps_index"].astype(int)
        labelled["candidate_arc_id"] = labelled["candidate_arc_id"].astype(int)
        labelled["label"] = labelled["label"].astype(int)
        missing = set(OUTPUT_COLUMNS) - set(labelled.columns)
        if missing:
            raise ValueError(f"Missing required output columns: {sorted(missing)}")
        labelled = labelled[OUTPUT_COLUMNS]

    diagnostics["trip_id"] = trip["trip_id"]
    diagnostics["gps_observations"] = len(trip["track"])
    diagnostics["reference_route_arcs"] = len(trip["route"])
    diagnostics["candidate_rows_before_alignment"] = len(features)
    diagnostics["labelled_candidate_rows"] = len(labelled)
    diagnostics["positive_rows"] = int(labelled["label"].sum()) if not labelled.empty else 0
    diagnostics["negative_rows"] = (
        int((labelled["label"] == 0).sum()) if not labelled.empty else 0
    )
    return labelled, diagnostics


def process_split(
    data_root: Path,
    split: str,
    output_dir: Path,
    limit: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Process one partition and save its candidate-level CSV and diagnostics."""
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")
    trip_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    if limit is not None:
        trip_dirs = trip_dirs[:limit]
    if not trip_dirs:
        raise FileNotFoundError(f"No trip directories found in {split_dir}")

    frames = []
    diagnostics_rows = []
    for number, trip_dir in enumerate(trip_dirs, start=1):
        print(f"\n[{split}] Processing trip {trip_dir.name} ({number}/{len(trip_dirs)})")
        frame, diagnostics = process_trip(trip_dir)
        frames.append(frame)
        diagnostics_rows.append(diagnostics)
        print(
            f"  GPS={diagnostics['gps_observations']} | "
            f"aligned={diagnostics['aligned_observations']} | "
            f"misses={diagnostics['candidate_misses']} | "
            f"coverage={diagnostics['candidate_coverage_pct']:.2f}% | "
            f"rows={diagnostics['labelled_candidate_rows']} | "
            f"positive={diagnostics['positive_rows']}"
        )

    nonempty = [frame for frame in frames if not frame.empty]
    combined = (
        pd.concat(nonempty, ignore_index=True)
        if nonempty
        else pd.DataFrame(columns=OUTPUT_COLUMNS)
    )
    combined = combined.reindex(columns=OUTPUT_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_name = "training_data.csv" if split == "train" else "student_test_data.csv"
    combined.to_csv(output_dir / csv_name, index=False)
    pd.DataFrame(diagnostics_rows).to_csv(
        output_dir / f"{split}_alignment_diagnostics.csv", index=False
    )

    print(f"\n========== {split.upper()} SUMMARY ==========")
    print("Trips processed:", len(trip_dirs))
    print("Candidate rows:", len(combined))
    print("Positive labels:", int(combined["label"].sum()) if len(combined) else 0)
    print("Negative labels:", int((combined["label"] == 0).sum()) if len(combined) else 0)
    print("Saved:", output_dir / csv_name)
    print("Diagnostics:", output_dir / f"{split}_alignment_diagnostics.csv")
    return combined, diagnostics_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build PathSnap candidate-level labelled datasets."
    )
    parser.add_argument(
        "--split",
        choices=["train", "student_test", "all"],
        default="all",
        help="Dataset partition to process (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of trips per selected split (for validation).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset root containing train/ and student_test/ (or set PATHSNAP_DATA).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory for generated CSVs and diagnostics.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    data_root = args.data_root.expanduser().resolve() if args.data_root else find_data_root()
    output_dir = args.output_dir.expanduser().resolve()
    print("Dataset root:", data_root)
    print("Output directory:", output_dir)
    print("NOTE: student_test labels are for development evaluation only; do not fit models on them.")

    splits = ["train", "student_test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(data_root, split, output_dir, args.limit)


if __name__ == "__main__":
    main()
