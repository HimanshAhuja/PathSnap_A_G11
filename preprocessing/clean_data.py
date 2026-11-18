
import numpy as np
import pandas as pd

from preprocessing.load_data import load_partition


def validate_numeric(df, columns):
    """Check for missing, non-numeric and infinite values."""

    issues = []

    for col in columns:

        if col not in df.columns:
            issues.append(f"Missing column: {col}")
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        missing = values.isna().sum()

        if missing:
            issues.append(
                f"{col}: {missing} missing/non-numeric values"
            )

        infinite = np.isinf(values.dropna()).sum()

        if infinite:
            issues.append(
                f"{col}: {infinite} infinite values"
            )

    return issues


def validate_coordinates(df):
    """Check longitude and latitude ranges."""

    issues = []

    if not {"longitude", "latitude"}.issubset(df.columns):
        return ["Coordinate columns missing"]

    longitude = pd.to_numeric(
        df["longitude"], errors="coerce"
    )

    latitude = pd.to_numeric(
        df["latitude"], errors="coerce"
    )

    invalid_lon = (
        longitude.notna() &
        ~longitude.between(-180, 180)
    )

    invalid_lat = (
        latitude.notna() &
        ~latitude.between(-90, 90)
    )

    if invalid_lon.any():
        issues.append(
            f"Invalid longitude values: {invalid_lon.sum()}"
        )

    if invalid_lat.any():
        issues.append(
            f"Invalid latitude values: {invalid_lat.sum()}"
        )

    return issues


def validate_track(track):
    """Validate GPS observations and time values."""

    issues = []

    issues.extend(
        validate_numeric(
            track,
            ["longitude", "latitude", "time_value"]
        )
    )

    issues.extend(validate_coordinates(track))

    duplicate_rows = track.duplicated().sum()

    if duplicate_rows:
        issues.append(
            f"Duplicate GPS rows: {duplicate_rows}"
        )

    time = pd.to_numeric(
        track["time_value"], errors="coerce"
    ).dropna()

    if len(time) > 1:

        differences = time.diff().dropna()

        if (differences < 0).any():
            issues.append(
                "Time values are not monotonically increasing"
            )

        if (differences == 0).any():
            issues.append("Duplicate time values found")

        positive_gaps = differences[differences > 0]

        if not positive_gaps.empty:
            issues.append(
                f"Maximum positive time-value gap: "
                f"{positive_gaps.max()}"
            )

    return issues


def validate_nodes(nodes):
    """Validate road-node coordinates."""

    issues = []

    issues.extend(
        validate_numeric(nodes, ["longitude", "latitude"])
    )

    issues.extend(validate_coordinates(nodes))

    duplicates = nodes.duplicated().sum()

    if duplicates:
        issues.append(
            f"Duplicate node coordinates: {duplicates}"
        )

    return issues


def validate_arcs(arcs):
    """Validate road-network arc identifiers."""

    issues = []

    issues.extend(
        validate_numeric(arcs, ["source_id", "target_id"])
    )

    for col in ["source_id", "target_id"]:

        values = pd.to_numeric(
            arcs[col], errors="coerce"
        ).dropna()

        if not np.isfinite(values).all():
            continue

        if not np.equal(values, np.floor(values)).all():
            issues.append(
                f"{col} contains non-integer identifiers"
            )

    duplicates = arcs.duplicated().sum()

    if duplicates:
        issues.append(f"Duplicate arcs: {duplicates}")

    return issues


def validate_route(route):
    """Validate reference route identifiers."""

    issues = []

    issues.extend(
        validate_numeric(route, ["identifier"])
    )

    values = pd.to_numeric(
        route["identifier"], errors="coerce"
    ).dropna()

    if np.isfinite(values).all():

        if not np.equal(values, np.floor(values)).all():
            issues.append(
                "Route contains non-integer identifiers"
            )

    return issues


def validate_trip(trip):
    """Validate all four files in a trip."""

    return {
        "track": validate_track(trip.track),
        "nodes": validate_nodes(trip.nodes),
        "arcs": validate_arcs(trip.arcs),
        "route": validate_route(trip.route)
    }


def validate_partition(data_root, partition_name):
    """Validate every successfully loaded trip in a partition."""

    print("\n" + "=" * 70)
    print(f"VALIDATING {partition_name}")
    print("=" * 70)

    trips, load_failures = load_partition(data_root)

    total_issues = 0
    clean_trips = 0
    affected_trips = 0

    detailed_results = []

    for index, trip in enumerate(trips, start=1):

        report = validate_trip(trip)

        issue_count = sum(
            len(messages)
            for messages in report.values()
        )

        total_issues += issue_count

        if issue_count == 0:
            clean_trips += 1
            status = "CLEAN"
        else:
            affected_trips += 1
            status = "ISSUES FOUND"

        print(
            f"[{index}/{len(trips)}] "
            f"{trip.trip_id}: {status} "
            f"({issue_count} issues)"
        )

        for file_type, messages in report.items():

            for message in messages:
                print(f"    {file_type}: {message}")

        detailed_results.append({
            "trip_id": trip.trip_id,
            "track_rows": len(trip.track),
            "node_rows": len(trip.nodes),
            "arc_rows": len(trip.arcs),
            "route_rows": len(trip.route),
            "issue_count": issue_count,
            "status": status
        })

    print("\nPARTITION SUMMARY")
    print(f"Trips found:       {len(trips) + len(load_failures)}")
    print(f"Successfully read: {len(trips)}")
    print(f"Load failures:     {len(load_failures)}")
    print(f"Clean trips:       {clean_trips}")
    print(f"Affected trips:    {affected_trips}")
    print(f"Total issues:      {total_issues}")

    return detailed_results, load_failures


if __name__ == "__main__":

    partitions = [
        ("data/train", "TRAIN"),
        ("data/student_test", "STUDENT TEST")
    ]

    all_results = {}

    for data_root, name in partitions:

        results, failures = validate_partition(
            data_root,
            name
        )

        all_results[name] = results

        if failures:
            print("\nTRIPS THAT FAILED TO LOAD:")

            for failure in failures:
                print(failure)

    print("\n" + "=" * 70)
    print("COMPLETE DATASET VALIDATION FINISHED")
    print("=" * 70)
