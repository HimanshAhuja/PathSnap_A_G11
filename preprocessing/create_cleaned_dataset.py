
from pathlib import Path
import csv
import math
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_ROOT = PROJECT_ROOT / "data"
PROCESSED_ROOT = RAW_ROOT / "processed"

PARTITIONS = ["train", "student_test"]

def valid_track_row(line):
    """Validate longitude, latitude and timestamp."""
    parts = line.split()

    if len(parts) != 3:
        return None

    try:
        longitude, latitude, timestamp = map(float, parts)
    except ValueError:
        return None

    if not all(map(math.isfinite, [longitude, latitude, timestamp])):
        return None

    if not (-180 <= longitude <= 180):
        return None

    if not (-90 <= latitude <= 90):
        return None

    return longitude, latitude, timestamp


def process_trip(trip_dir, output_dir):
    trip_id = trip_dir.name

    track_file = trip_dir / f"{trip_id}.track"
    nodes_file = trip_dir / f"{trip_id}.nodes"
    arcs_file = trip_dir / f"{trip_id}.arcs"
    route_file = trip_dir / f"{trip_id}.route"

    required_files = [track_file, nodes_file, arcs_file, route_file]

    if not all(path.exists() for path in required_files):
        print(f"SKIPPED {trip_id}: missing files")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_lines = []
    timestamps = []
    coordinates = []

    total_rows = 0
    removed_rows = 0

    with track_file.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue

            total_rows += 1
            parsed = valid_track_row(line)

            if parsed is None:
                removed_rows += 1
                continue

            longitude, latitude, timestamp = parsed

            valid_lines.append(line.strip() + "\n")
            timestamps.append(timestamp)
            coordinates.append((longitude, latitude))

    # Preserve original observation order.
    cleaned_track = output_dir / track_file.name

    with cleaned_track.open("w", encoding="utf-8") as output:
        output.writelines(valid_lines)

    # Preserve graph and route files exactly.
    for source_file in [nodes_file, arcs_file, route_file]:
        shutil.copy2(source_file, output_dir / source_file.name)

    # Report timestamp irregularities without deleting observations.
    out_of_order = sum(
        timestamps[i] < timestamps[i - 1]
        for i in range(1, len(timestamps))
    )

    duplicate_coordinates = len(coordinates) - len(set(coordinates))

    irregular_gaps = sum(
        not math.isclose(
            timestamps[i] - timestamps[i - 1],
            1.0,
            abs_tol=1e-6
        )
        for i in range(1, len(timestamps))
        if timestamps[i] >= timestamps[i - 1]
    )

    return {
        "trip_id": trip_id,
        "original_track_rows": total_rows,
        "cleaned_track_rows": len(valid_lines),
        "removed_track_rows": removed_rows,
        "duplicate_coordinates": duplicate_coordinates,
        "out_of_order_timestamps": out_of_order,
        "non_1_second_gaps": irregular_gaps,
    }


def main():
    all_reports = []

    for partition in PARTITIONS:
        source_partition = RAW_ROOT / partition
        output_partition = PROCESSED_ROOT / partition

        if not source_partition.exists():
            print(f"Partition not found: {source_partition}")
            continue

        trip_dirs = sorted(
            path for path in source_partition.iterdir()
            if path.is_dir()
        )

        print(f"\nProcessing {partition}: {len(trip_dirs)} trips")

        for trip_dir in trip_dirs:
            report = process_trip(
                trip_dir,
                output_partition / trip_dir.name
            )

            if report:
                report["partition"] = partition
                all_reports.append(report)

                print(
                    f"{partition}/{report['trip_id']}: "
                    f"{report['cleaned_track_rows']}/"
                    f"{report['original_track_rows']} GPS rows retained"
                )

    report_dir = PROJECT_ROOT / "preprocessing" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file = report_dir / "cleaning_report.csv"

    if all_reports:
        with report_file.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "partition",
                    "trip_id",
                    "original_track_rows",
                    "cleaned_track_rows",
                    "removed_track_rows",
                    "duplicate_coordinates",
                    "out_of_order_timestamps",
                    "non_1_second_gaps",
                ],
            )
            writer.writeheader()
            writer.writerows(all_reports)

    print("\nCleaning completed.")
    print(f"Processed data: {PROCESSED_ROOT}")
    print(f"Cleaning report: {report_file}")


if __name__ == "__main__":
    main()
