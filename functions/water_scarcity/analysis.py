"""This script contains functions to calculate water use by basin, process discharge and abstraction rasters in m3, reproject rasters to a chosen CRS, calculate monthly basin sum, and calculate water scarcity."""

from pathlib import Path
from typing import TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import geometry_mask
from tqdm import tqdm

# DataFrame TypeVar
DFT = TypeVar("DFT", pd.DataFrame, gpd.GeoDataFrame)

# Constants
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def find_water_extraction_sites(buffer_zones: gpd.GeoDataFrame, discharge_file_path: Path | str) -> gpd.GeoDataFrame:
    """Find  water extraction sites based on maximum discharge within buffer zones."""
    # Create a copy of the gdf DataFrame
    buffer_zones = buffer_zones.copy()

    # Read the raster data
    with rasterio.open(Path(discharge_file_path)) as src:
        discharge_data = src.read(2)  # # Using March (index 2) as it has the highest global discharge of all months
        transform = src.transform
        out_shape = src.shape

    # Initialize the progress bar
    with tqdm(total=len(buffer_zones), desc="Processing locations") as pbar:
        for index, zone in buffer_zones.iterrows():
            # Create a raster mask from buffer geometries with geometry_mask
            mask = geometry_mask([zone.geometry], out_shape=out_shape, transform=transform, invert=True)

            # Mask data (invert mask to keep only the cells within the buffer zone)
            masked_data = np.ma.masked_array(discharge_data, mask=~mask)
            if masked_data.count() > 0:
                # Find the coordinates of the cell with the highest discharge
                max_cell = np.unravel_index(masked_data.argmax(), masked_data.shape)
                buffer_zones.loc[index, "lon_water_extraction"], buffer_zones.loc[index, "lat_water_extraction"] = (
                    src.xy(max_cell[0], max_cell[1])
                )

            # Update the progress bar
            pbar.update(1)

    return buffer_zones


def summing_dc_water_use_per_extraction_site(water_use_df: DFT) -> DFT:
    """This function calculates the total data center driven water use per extraction site."""
    extraction_site_coords = ["lat_water_extraction", "lon_water_extraction"]

    # Calculate site totals
    site_totals = (
        water_use_df.groupby(extraction_site_coords)  # Group water use by extraction site
        .agg(
            site_annual_water_use_m3=("water_use_m3", "sum"),  # Sum the water use
            site_dc_count=("water_use_m3", "count"),  # Count the number of data centers at each site
        )
        .assign(  # Divide by 12 to get monthly water use
            dc_cumulative_monthly_water_use_m3=lambda x: x.site_annual_water_use_m3 / 12
        )
        .reset_index()
        .drop(columns="site_annual_water_use_m3")
    )

    # Merge the water_use_df_grouped with the original dataframe
    water_use_df = (  # Rename the water use column to specify it is data-center-driven
        water_use_df.rename(columns={"water_use_m3": "dc_annual_water_use_m3"})
        .merge(site_totals, on=extraction_site_coords, how="left")  # Merge the dataframes
        .assign(  # Create a boolean column for shared extraction sites
            shared_extraction_site=lambda x: x.site_dc_count > 1
        )
        .drop(columns=["site_dc_count"])
    )

    return water_use_df


def repeat_rows_for_months(df: DFT, month_names: list[str] = MONTH_NAMES) -> DFT:
    """Repeat each row for every month."""
    return df.loc[df.index.repeat(len(month_names))].assign(month=month_names * len(df)).reset_index(drop=True)


def get_global_warming_scenario_files(scenario_folder: Path | str) -> tuple[list[Path], ...]:
    """Get discharge and abstraction files for a scenario folder."""
    # Get files using pathlib pattern matching
    discharge_files = list(Path(scenario_folder).glob("q*"))  # q for discharge
    abstraction_files = list(Path(scenario_folder).glob("ab*"))  # ab for abstraction

    return discharge_files, abstraction_files


def calculate_monthly_discharge_abstraction(
    gdf: gpd.GeoDataFrame,
    discharge_files: list[Path] | list[str],
    abstraction_files: list[Path] | list[str],
    lat_col: str,
    lon_col: str,
    month_names: list[str] = MONTH_NAMES,
) -> pd.DataFrame:
    """Calculate monthly discharge and abstraction values at extraction sites."""
    # Create a copy of the gdf DataFrame
    gdf = gdf.copy()

    # Create a subset with only unique lat and lon values
    gdf_unique = gdf[[lat_col, lon_col]].drop_duplicates()

    # Create a dataframe with each lat and lon value repeated for each month
    gdf_coords = repeat_rows_for_months(gdf_unique)

    # Extract coordinates
    coords = list(zip(gdf_unique[lon_col], gdf_unique[lat_col], strict=True))

    # Initialize dictionaries to store discharge and abstraction data
    discharge_data_dict = {month: [] for month in month_names}
    abstraction_data_dict = {month: [] for month in month_names}

    # Loop through each discharge and abstraction file and extract the data for each location
    for discharge_file, abstraction_file in tqdm(
        zip(discharge_files, abstraction_files, strict=True), total=len(discharge_files), desc="Processing files"
    ):
        # Read the discharge raster data
        with rasterio.open(discharge_file) as src:
            discharge_data = list(src.sample(coords))

        # Read the abstraction raster data
        with rasterio.open(abstraction_file) as src:
            abstraction_data = list(src.sample(coords))

        # Store the discharge and abstraction values for each month
        for month_idx, month in enumerate(month_names):
            discharge_data_dict[month].append([d[month_idx] for d in discharge_data])
            abstraction_data_dict[month].append([a[month_idx] for a in abstraction_data])

    # Calculate median, max, and min for each month and update the dataframe
    for month in month_names:
        discharge_values = list(zip(*discharge_data_dict[month], strict=False))
        abstraction_values = list(zip(*abstraction_data_dict[month], strict=False))

        gdf_unique[f"discharge_m3_{month}_median"] = [np.median(values) for values in discharge_values]
        gdf_unique[f"discharge_m3_{month}_max"] = [np.max(values) for values in discharge_values]
        gdf_unique[f"discharge_m3_{month}_min"] = [np.min(values) for values in discharge_values]

        gdf_unique[f"abstraction_m3_{month}_median"] = [np.median(values) for values in abstraction_values]
        gdf_unique[f"abstraction_m3_{month}_max"] = [np.max(values) for values in abstraction_values]
        gdf_unique[f"abstraction_m3_{month}_min"] = [np.min(values) for values in abstraction_values]

    # Transpose the discharge and abstraction data
    discharge_median, discharge_max, discharge_min = [
        gdf_unique[[f"discharge_m3_{month}_{stat}" for month in month_names]].to_numpy().flatten()
        for stat in ["median", "max", "min"]
    ]
    abstraction_median, abstraction_max, abstraction_min = [
        gdf_unique[[f"abstraction_m3_{month}_{stat}" for month in month_names]].to_numpy().flatten()
        for stat in ["median", "max", "min"]
    ]

    # Create a new DataFrame with the transposed data
    transposed_data = pd.DataFrame(
        {
            "discharge_m3_median": discharge_median,
            "discharge_m3_max": discharge_max,
            "discharge_m3_min": discharge_min,
            "abstraction_m3_median": abstraction_median,
            "abstraction_m3_max": abstraction_max,
            "abstraction_m3_min": abstraction_min,
        }
    )

    # Concat the transposed data with the coordinates
    transposed_data = pd.concat([gdf_coords, transposed_data], axis=1)

    # Merge the data based on the coordinates and month
    discharge_abstraction_by_location = gdf.merge(transposed_data, on=[lat_col, lon_col, "month"], how="left")

    return discharge_abstraction_by_location


def compute_efr(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Environmental Flow Requirements using VMF and 60% threshold methods."""
    # Create copy and calculate flows
    result = df.copy()

    # Calculate mean monthly flow by summing discharge and abstraction
    result["mean_monthly_flow"] = (
        result.filter(like="discharge_m3_median").iloc[:, 0] + result.filter(like="abstraction_m3_median").iloc[:, 0]
    )

    result["mean_annual_flow"] = (
        result.groupby(["name", "latitude", "longitude"])["mean_monthly_flow"].transform("sum") / 12
    )

    # Calculate Variable Monthly Flow (VMF) ratios:
    # - Flow ratio = mean monthly flow / mean annual flow
    # - High flow (ratio > 0.8): EFR = 30% of flow
    # - Low flow (ratio ≤ 0.4): EFR = 60% of flow
    # - Intermediate flow (0.4 < ratio ≤ 0.8): EFR = 45% of flow
    monthly_to_annual = result["mean_monthly_flow"] / result["mean_annual_flow"]
    result["EFR_ratio"] = np.where(monthly_to_annual <= 0.4, 0.6, np.where(monthly_to_annual > 0.8, 0.3, 0.45))

    # Assign season and calculate EFR
    result["season"] = np.select(
        [result["EFR_ratio"] == 0.6, result["EFR_ratio"] == 0.3], ["low", "high"], default="intermediate"
    )

    result["EFR_m3_month"] = result["EFR_ratio"] * result["mean_monthly_flow"]

    # For a sensitivity analysis with 0.6 EFR
    result["EFR_m3_month_0p6"] = 0.6 * result["mean_monthly_flow"]

    return result


def calculate_water_scarcity(
    df: pd.DataFrame,
    abstraction_column: str,
    discharge_column: str,
    efr_column: str,
    *,
    efr_sensitivity_analysis: bool = False,
) -> pd.DataFrame:
    """Calculate water scarcity indices with optional sensitivity analysis."""
    result = df.copy()

    # Calculate available water
    available_water = {
        "base": (result[discharge_column] + result[abstraction_column]), # Base case without accounting for data center water use
        "dc": (result[discharge_column] + result[abstraction_column]), # With data center water use
    }

    if efr_sensitivity_analysis:
        available_water.update(
            {
                "base_0p6": (result[discharge_column] + result[abstraction_column]), # Base case with 0.6 EFR
                "dc_0p6": (result[discharge_column] + result[abstraction_column]), # With data center water use and 0.6 EFR
            }
        )

    # Calculate indices
    indices = {
        # Water scarcity index
        "water_scarcity_index": result[abstraction_column] / (available_water["base"]-result[efr_column]),
        "water_scarcity_index_dc": (result[abstraction_column] + result["dc_cumulative_monthly_water_use_m3"])
        / (available_water["dc"]-result[efr_column]),

        # Vulnerability index
        "vulnerability_index": result[abstraction_column] / available_water["base"],
        "vulnerability_index_dc": (result[abstraction_column] + result["dc_cumulative_monthly_water_use_m3"])
        / available_water["dc"],
    }

    if efr_sensitivity_analysis:
        indices.update(
            {
                # Water scarcity index with 0.6 EFR
                "water_scarcity_index_0p6": result[abstraction_column] / (available_water["base_0p6"]-result[f"{efr_column}_0p6"]),
                "water_scarcity_index_dc_0p6": (
                    result[abstraction_column] + result["dc_cumulative_monthly_water_use_m3"]
                )
                / (available_water["dc_0p6"]-result[f"{efr_column}_0p6"]),
            }
        )

    # Assign the calculated indices to new columns in the result DataFrame
    for key, value in indices.items():
        result[key] = value

    return result


def get_water_scarcity_summary(
    water_scarcity_df: pd.DataFrame, cols_to_keep: list, *, efr_sensitivity_analysis: bool = False
) -> pd.DataFrame:
    """Create summary of water scarcity results with optional EFR sensitivity analysis.

    Args:
    water_scarcity_df (pd.DataFrame): DataFrame with water scarcity results
    cols_to_keep (list): Columns to keep in the summary
    efr_sensitivity_analysis (bool): Whether to include EFR sensitivity analysis

    Returns:
    pd.DataFrame: Summary of water scarcity results
    """
    # Local variables
    data_center_type = "data_center"
    month_col = "month"

    # Keep only one row per location
    result = water_scarcity_df[water_scarcity_df["dc_cumulative_monthly_water_use_m3"] > 0].drop_duplicates(
        subset=[*cols_to_keep, month_col]
    )

    # Define aggregation metrics
    base_metrics = {
        "": {
            "months_WSI": ("water_scarcity_index", lambda x: ((x >= 1) | (x < 0)).sum()),  # Count months with WSI > 1
            "months_WSI_dc": (
                "water_scarcity_index_dc",
                lambda x: ((x >= 1) | (x < 0)).sum(),
            ),  # Count months with WSI_dc > 1
            "WSI_mean": ("water_scarcity_index", "mean"),
            "WSI_dc_mean": ("water_scarcity_index_dc", "mean"),
            "WSI_sd": ("water_scarcity_index", "std"),  # Standard deviation of WSI
            "WSI_dc_sd": ("water_scarcity_index_dc", "std"),  # Standard deviation of WSI_dc
            "months_vulnerability": ("vulnerability_index", lambda x: ((x >= 1) | (x < 0)).sum()),  # Count months with vulnerability > 1
            "months_vulnerability_dc": (
                "vulnerability_index_dc",
                lambda x: ((x >= 1) | (x < 0)).sum(),
            ),  # Count months with vulnerability_dc > 1
            "vulnerability_mean": ("vulnerability_index", "mean"),
            "vulnerability_dc_mean": ("vulnerability_index_dc", "mean"),
            "vulnerability_sd": ("vulnerability_index", "std"),  # Standard deviation of vulnerability
            "vulnerability_dc_sd": ("vulnerability_index_dc", "std"),  # Standard deviation of vulnerability
        }
    }

    # Sensitivity analysis with 0.6 EFR
    if efr_sensitivity_analysis:
        base_metrics["_0p6"] = {
            "months_WSI": ("water_scarcity_index_0p6", lambda x: ((x >= 1) | (x < 0)).sum()),
            "months_WSI_dc": ("water_scarcity_index_dc_0p6", lambda x: ((x >= 1) | (x < 0)).sum()),
            "WSI_mean": ("water_scarcity_index_0p6", "mean"),
            "WSI_dc_mean": ("water_scarcity_index_dc_0p6", "mean"),
        }

    # Process each metric set
    summary_dfs = []
    for suffix, metrics in base_metrics.items():
        # Calculate base metrics
        agg_dict = {f"{name}{suffix}": metric for name, metric in metrics.items()}
        summary = result.groupby(cols_to_keep).agg(**agg_dict).reset_index()

        # Calculate derived metrics
        wsi_cols = {"months": f"months_WSI{suffix}", "months_dc": f"months_WSI_dc{suffix}", "mean": f"WSI_mean{suffix}"}

        # Check if data centers increase water scarcity by at least one month, or if the region is always water scarce
        summary[f"dc_direct_increases_WSI_months{suffix}"] = (
            (summary[wsi_cols["months_dc"]] > summary[wsi_cols["months"]]) & (summary[wsi_cols["mean"]] != 0)
        ) | (
            (summary[wsi_cols["months_dc"]] == 12)
            & (summary[wsi_cols["months"]] == 12)
            & (summary[wsi_cols["mean"]] != 0)
        )
        # Calculate the number of months data centers increase water scarcity
        summary[f"dc_direct_increase_months_count{suffix}"] = summary[wsi_cols["months_dc"]] - summary[wsi_cols["months"]]

        # Water scarcity regions and indirect effects
        power_zones_increase = summary[  # Power zones where data centers increase water scarcity
            summary[f"dc_direct_increases_WSI_months{suffix}"] & (summary["type"] != data_center_type)
        ]["power_grid_zone"].unique()

        power_zones_scarce = summary[  # Power zones where data centers indirectly extract from water scarce sites
            (summary["type"] != data_center_type) & (summary[wsi_cols["months"]] > 0)
        ]["power_grid_zone"].unique()

        # Add boolean columns
        summary[f"dc_indirect_increases_WSI_months{suffix}"] = (  # Data centers increase water scarcity at power plants
            summary["power_grid_zone"].isin(power_zones_increase) & (summary["type"] == data_center_type)
        )
        summary[f"WS_region{suffix}"] = (
            summary[wsi_cols["months"]] > 0
        )  # Water scarce region, with at least one month of water scarcity
        summary[f"dc_indirectly_extracts_water_scarce_site{suffix}"] = summary["power_grid_zone"].isin(
            power_zones_scarce
        ) & (summary["type"] == data_center_type)

        # Relative index
        summary[f"relative_index{suffix}"] = np.where(  # Difference between WSI and WSI_dc
            (summary[wsi_cols["months_dc"]] > 0) & (summary[wsi_cols["mean"]] != 0),
            abs(summary[wsi_cols["mean"]] - summary[f"WSI_dc_mean{suffix}"]),
            0,
        )

        summary_dfs.append(summary)

    # Combine results
    final_summary = summary_dfs[0] if len(summary_dfs) == 1 else summary_dfs[0].merge(summary_dfs[1], on=cols_to_keep)

    return final_summary.fillna(0)
