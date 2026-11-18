
from pathlib import Path
from dataclasses import dataclass

import pandas as pd


# Expected columns based on inspected dataset samples
FILE_COLUMNS = {
    "track": ["longitude", "latitude", "time_value"],
    "nodes": ["longitude", "latitude"],
    "arcs": ["source_id", "target_id"],
    "route": ["identifier"],
}


@dataclass
class TripData:
    trip_id: str
    track: pd.DataFrame
    nodes: pd.DataFrame
    arcs: pd.DataFrame
    route: pd.DataFrame


def load_table(file_path: Path, columns: list) -> pd.DataFrame:
    """Load a whitespace-separated dataset file."""

    if not file_path.is_file():
        raise FileNotFoundError(f"Missing file: {file_path}")

    data = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=columns,
        dtype="float64",
        engine="python"
    )

    if data.empty:
        raise ValueError(f"Empty dataset file: {file_path}")

    return data


def load_trip(data_root: str, trip_id: str) -> TripData:
    """Load all four files belonging to one trip."""

    trip_folder = Path(data_root) / trip_id

    if not trip_folder.is_dir():
        raise FileNotFoundError(
            f"Trip folder not found: {trip_folder}"
        )

    track = load_table(
        trip_folder / f"{trip_id}.track",
        FILE_COLUMNS["track"]
    )

    nodes = load_table(
        trip_folder / f"{trip_id}.nodes",
        FILE_COLUMNS["nodes"]
    )

    arcs = load_table(
        trip_folder / f"{trip_id}.arcs",
        FILE_COLUMNS["arcs"]
    )

    route = load_table(
        trip_folder / f"{trip_id}.route",
        FILE_COLUMNS["route"]
    )

    return TripData(
        trip_id=trip_id,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route
    )


def load_all_trip_ids(data_root: str) -> list:
    """Get all trip folder names in sorted order."""

    root = Path(data_root)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}"
        )

    return sorted(
        folder.name
        for folder in root.iterdir()
        if folder.is_dir()
    )


def load_partition(data_root: str) -> list:
    """
    Load every trip in a partition.

    Returns:
        successfully loaded trips
        failed trip information
    """

    trip_ids = load_all_trip_ids(data_root)

    loaded_trips = []
    failed_trips = []

    print(f"\nLoading partition: {data_root}")
    print(f"Total trips found: {len(trip_ids)}")

    for index, trip_id in enumerate(trip_ids, start=1):

        try:
            trip = load_trip(data_root, trip_id)
            loaded_trips.append(trip)

            print(
                f"[{index}/{len(trip_ids)}] "
                f"{trip_id} loaded successfully"
            )

        except Exception as error:

            failed_trips.append({
                "trip_id": trip_id,
                "error": str(error)
            })

            print(
                f"[{index}/{len(trip_ids)}] "
                f"{trip_id} FAILED: {error}"
            )

    print("\nLOADING SUMMARY")
    print(f"Total trips: {len(trip_ids)}")
    print(f"Loaded: {len(loaded_trips)}")
    print(f"Failed: {len(failed_trips)}")

    return loaded_trips, failed_trips


if __name__ == "__main__":

    partitions = {
        "TRAIN": "data/train",
        "STUDENT TEST": "data/student_test"
    }

    for name, path in partitions.items():

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        trips, failures = load_partition(path)

        if trips:
            print("\nFirst successfully loaded trip:")
            sample = trips[0]

            print("Trip ID:", sample.trip_id)
            print("Track:", len(sample.track))
            print("Nodes:", len(sample.nodes))
            print("Arcs:", len(sample.arcs))
            print("Route:", len(sample.route))

    print("\nALL PARTITIONS PROCESSED")
