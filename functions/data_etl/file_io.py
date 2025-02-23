"""Functions for reading and writing data from files."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from functions.project_settings import WGS84_CRS


def read_gdf_from_csv(
    file_path: str | Path, crs: CRS = WGS84_CRS, lat_col: str = "latitude", lon_col: str = "longitude"
) -> gpd.GeoDataFrame:
    """Read a GeoDataFrame from a CSV file. It should contain columns for latitude and longitude."""
    if not Path(file_path).exists():
        err_msg = f"File not found: {file_path}"
        raise FileNotFoundError(err_msg)

    # Read the CSV file with low_memory=False to handle mixed types
    df = pd.read_csv(Path(file_path), low_memory=False)

    # Create GeoDataFrame, ensuring coordinates are numeric
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            pd.to_numeric(df[lon_col], errors="coerce"), pd.to_numeric(df[lat_col], errors="coerce")
        ),
        crs=crs,
    )
