
import pandas as pd


def coordinate_validity(longitude, latitude):
    """
    Check whether longitude and latitude are within
    valid global coordinate ranges.

    Returns a Boolean value or Boolean Series.
    """

    longitude = pd.to_numeric(
        longitude, errors="coerce"
    )

    latitude = pd.to_numeric(
        latitude, errors="coerce"
    )

    return (
        longitude.notna()
        & latitude.notna()
        & longitude.between(-180, 180)
        & latitude.between(-90, 90)
    )


def get_coordinate_bounds(data):
    """Return the bounding box of valid coordinates."""

    valid = coordinate_validity(
        data["longitude"],
        data["latitude"]
    )

    valid_data = data.loc[valid]

    if valid_data.empty:
        return None

    return {
        "longitude_min": valid_data["longitude"].min(),
        "longitude_max": valid_data["longitude"].max(),
        "latitude_min": valid_data["latitude"].min(),
        "latitude_max": valid_data["latitude"].max()
    }


def inspect_coordinates(data):
    """Summarize coordinate validity and duplicate coordinates."""

    valid = coordinate_validity(
        data["longitude"],
        data["latitude"]
    )

    return {
        "total_coordinates": len(data),
        "valid_coordinates": int(valid.sum()),
        "invalid_coordinates": int((~valid).sum()),
        "duplicate_coordinates": int(
            data[["longitude", "latitude"]].duplicated().sum()
        ),
        "bounds": get_coordinate_bounds(data)
    }
