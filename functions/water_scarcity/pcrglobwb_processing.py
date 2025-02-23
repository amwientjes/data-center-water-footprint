"""Functions to process PCRGlobWB data for water scarcity analysis."""

from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.io import DatasetReader
from tqdm import tqdm

from functions.project_settings import GLOBAL_WARMING_SCENARIOS_FUTURE, WGS84_CRS

# Constants
EARTH_RADIUS_M = 6_371_000  # Approximate Earth radius in meters
SECONDS_PER_MONTH = 60 * 60 * 24 * 30.44  # Average number of seconds in a month
#  Map of filenames to climate scenarios
GLOBAL_WARMING_SCENARIO_MAP = {"0C": "hist", **{scenario: scenario for scenario in GLOBAL_WARMING_SCENARIOS_FUTURE}}


def reproject_raster_to_crs(src_path: Path | str, dst_path: Path | str, crs: CRS = WGS84_CRS) -> None:
    """Reproject a single raster file to specified CRS."""
    with rasterio.open(src_path) as src:
        # Calculate transformation parameters
        target_transform, width, height = rasterio.warp.calculate_default_transform(
            src.crs, crs, src.width, src.height, *src.bounds
        )

        # Update metadata for output
        meta = src.meta
        meta.update({"crs": crs.to_string(), "transform": target_transform, "width": width, "height": height})

        # Write reprojected raster
        with rasterio.open(dst_path, "w", **meta) as dst:
            for band_idx in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=crs,
                    resampling=rasterio.warp.Resampling.nearest,
                )


def reproject_rasters_to_crs(
    src_paths: list[Path] | list[str], dst_dir: Path | str, crs: CRS = WGS84_CRS, reproj_suffix: str = "_reprojected"
) -> None:
    """Reproject multiple raster files to specified CRS."""
    for src_path in tqdm(src_paths, total=len(src_paths), desc="Reprojecting rasters", unit="file"):
        src_path_obj = Path(src_path)  # Ensure Path object
        dst_path = Path(dst_dir) / (src_path_obj.stem + reproj_suffix + src_path_obj.suffix)
        reproject_raster_to_crs(src_path_obj, dst_path, crs)


def calculate_cell_area(raster: DatasetReader) -> np.ndarray:
    """Calculates the cell area in square meters for a given raster.

    Assumes raster CRS is in degrees (EPSG:4326).
    """
    # Verify CRS
    expected_crs = "EPSG:4326"
    if raster.crs.to_string() != expected_crs:
        err_msg = f"Raster must be in WGS84 (EPSG:4326), got {raster.crs}"
        raise ValueError(err_msg)

    # Extract transformation parameters
    transform = raster.transform
    res_x = transform[0]  # Resolution in X direction
    res_y = -transform[4]  # Resolution in Y direction

    # Create coordinate grids
    _, lat_grid = np.meshgrid(
        np.arange(raster.bounds.left, raster.bounds.right, res_x),
        np.arange(raster.bounds.top, raster.bounds.bottom, -res_y),
    )

    # Calculate areas using spherical geometry
    lat_rad = np.radians(lat_grid)
    area = (EARTH_RADIUS_M**2) * np.abs(np.radians(res_x) * (np.sin(np.radians(lat_grid + res_y)) - np.sin(lat_rad)))

    return area


def get_scenario_folder(filename: str) -> str:
    """Extract climate scenario from filename ending."""
    stem = Path(filename).stem
    if not (folder := GLOBAL_WARMING_SCENARIO_MAP.get(stem)):
        err_msg = f"Climate scenario not found for filename {filename}"
        raise ValueError(err_msg)

    return folder


def process_discharge_raster(q_path: Path, out_dir: Path) -> None:
    """Process water discharge raster and convert to m³/month, and save the result to a new file.

    Args:
        q_path: Path to reprojected discharge raster
        out_dir: Base output directory for processed files
    """
    # Read discharge data
    with rasterio.open(q_path) as discharge_src:
        discharge_data = discharge_src.read()
        meta = discharge_src.meta

    # Convert discharge from m³/second to m³/month
    discharge_data_m3 = discharge_data * SECONDS_PER_MONTH

    # Set up output path
    output_name = q_path.name.replace("q", "q_m3").replace("_reprojected", "")

    scenario_dir = out_dir / get_scenario_folder(output_name)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    output_path = scenario_dir / output_name

    # Save discharge raster
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(discharge_data_m3.astype(np.float32))


def process_abstraction_rasters(sab_path: Path, gab_path: Path, out_dir: Path) -> None:
    """Process surface and groundwater abstraction data, convert to m³/month, and save the result to a new file.

    Args:
        sab_path: Path to surface abstraction raster
        gab_path: Path to groundwater abstraction raster
        out_dir: Base output directory for processed files
    """
    # Load abstraction data
    with rasterio.open(sab_path) as surface_src, rasterio.open(gab_path) as groundwater_src:
        surface_data = surface_src.read()
        groundwater_data = groundwater_src.read()
        meta = surface_src.meta

        # Calculate area raster
        area_raster = calculate_cell_area(surface_src)

    # Combine surface and groundwater abstraction and convert from m/month to m³/month
    abstraction_data_m3 = (surface_data + groundwater_data) * area_raster[np.newaxis, :, :]

    # Set up output path
    output_name = sab_path.name.replace("sAb", "ab_m3_").replace("_reprojected", "")

    scenario_dir = out_dir / get_scenario_folder(output_name)
    scenario_dir.mkdir(parents=True, exist_ok=True)

    output_path = scenario_dir / output_name

    # Save abstraction raster
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(abstraction_data_m3.astype(np.float32))
