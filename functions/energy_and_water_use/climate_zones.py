"""Functions to work with climate zones and data centers."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

from functions.project_settings import WGS84_CRS


def read_koppen_tif(input_raster_path: Path, crs: str = WGS84_CRS) -> gpd.GeoDataFrame:
    """Read a tif file with Koppen climate classifications into a geoDataFrame."""
    # Open the raster file
    with rasterio.open(input_raster_path) as src:
        # Read the raster data as a numpy array
        raster_data = src.read(1)  # assuming climate classifications are in the first band

        # Retrieve transform information
        transform = src.transform

        # Polygonize the raster
        polygons = []
        for geom, value in shapes(raster_data, transform=transform):
            if not np.isnan(value):  # Ignore NoData areas
                polygons.append({"geometry": shape(geom), "classification": int(value)})

    # Convert to GeoDataFrame
    return gpd.GeoDataFrame(polygons, crs=crs)
