
from pathlib import Path

import pandas as pd

from preprocessing.load_data import (
    load_all_trip_ids,
    load_trip
)


PARTITIONS = {
    "train": "data/train",
    "student_test": "data/student_test"
}


def get_trip_statistics(trip, partition):
    """Calculate basic descriptive statistics for one trip."""

    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs
    route = trip.route

    return {
        "partition": partition,
        "trip_id": trip.trip_id,

        "track_rows": len(track),
        "node_rows": len(nodes),
        "arc_rows": len(arcs),
        "route_rows": len(route),

        "longitude_min": track["longitude"].min(),
        "longitude_max": track["longitude"].max(),
        "latitude_min": track["latitude"].min(),
        "latitude_max": track["latitude"].max(),

        "time_value_min": track["time_value"].min(),
        "time_value_max": track["time_value"].max(),

        "unique_track_coordinates": len(
            track[["longitude", "latitude"]].drop_duplicates()
        ),

        "unique_nodes": len(
            nodes[["longitude", "latitude"]].drop_duplicates()
        ),

        "unique_arcs": len(arcs.drop_duplicates()),
        "unique_route_identifiers": route["identifier"].nunique()
    }


def generate_statistics():

    all_statistics = []

    for partition, data_root in PARTITIONS.items():

        print("\n" + "=" * 65)
        print(f"DATASET STATISTICS: {partition.upper()}")
        print("=" * 65)

        trip_ids = load_all_trip_ids(data_root)

        print(f"Trips found: {len(trip_ids)}")

        failed = []

        for index, trip_id in enumerate(trip_ids, start=1):

            try:
                trip = load_trip(data_root, trip_id)

                statistics = get_trip_statistics(
                    trip,
                    partition
                )

                all_statistics.append(statistics)

                print(
                    f"[{index}/{len(trip_ids)}] "
                    f"{trip_id} processed"
                )

            except Exception as error:

                failed.append({
                    "partition": partition,
                    "trip_id": trip_id,
                    "error": str(error)
                })

                print(
                    f"[{index}/{len(trip_ids)}] "
                    f"{trip_id} FAILED: {error}"
                )

        print(f"\nSuccessful: {len(trip_ids) - len(failed)}")
        print(f"Failed: {len(failed)}")

    if not all_statistics:
        print("No statistics generated.")
        return

    output_dir = Path("preprocessing/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    statistics_df = pd.DataFrame(all_statistics)

    output_file = output_dir / "trip_statistics.csv"
    statistics_df.to_csv(output_file, index=False)

    print("\n" + "=" * 65)
    print("OVERALL DATASET SUMMARY")
    print("=" * 65)

    print(
        statistics_df.groupby("partition")[
            ["track_rows", "node_rows", "arc_rows", "route_rows"]
        ].agg(["count", "mean", "min", "max"])
    )

    print(f"\nStatistics saved to: {output_file}")


if __name__ == "__main__":
    generate_statistics()
