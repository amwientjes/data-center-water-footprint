"""Functions to calculate indirect water use of data centers at power plants."""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import nearest_points

from functions.project_settings import MOLLWEIDE_CRS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def assign_grid_zones(
    points_gdf: gpd.GeoDataFrame,
    zones_gdf: gpd.GeoDataFrame,
    zone_col_right: str = "power_grid_zone",
    zone_col_left: str = "zoneName",
    *,
    use_buffer: bool = False,
    buffer_distance_m: float = 10_000,
) -> gpd.GeoDataFrame:
    """Assign grid zones to points, using nearest zone within buffer if no direct match."""
    # Project to metric CRS (Mollweide / ESRI 54009) for distance calculations
    points_proj = points_gdf.to_crs(MOLLWEIDE_CRS)
    zones_proj = zones_gdf.to_crs(MOLLWEIDE_CRS)

    # Initial spatial join
    joined = gpd.sjoin(points_proj, zones_proj[["geometry", zone_col_left]], how="left", predicate="within").rename(
        columns={zone_col_left: zone_col_right}
    )

    # For points with multiple matches, select the smallest polygon
    if not joined.index.is_unique:
        zone_areas = zones_proj.geometry.area

        # Group by the original point index and select smallest polygon
        joined = (
            joined.merge(zone_areas.rename("area").to_frame(), left_on="index_right", right_index=True, how="left")
            .sort_values("area")  # Sort by area so smallest comes first
            .groupby(level=0)
            .first()  # Take first (smallest) for each point
            .drop(columns="area")
        )
        # Ensure the result is a GeoDataFrame
        joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=points_proj.crs)

    joined = joined.drop(columns="index_right")

    if use_buffer:
        # Find closest zone for unmatched points
        unmatched = joined[joined[zone_col_right].isna()]

        for idx, point in unmatched.geometry.items():
            distances = zones_proj.geometry.distance(point)
            nearby_zones = distances[distances <= buffer_distance_m]

            # Assign closest zone if any within max_distance, otherwise keep NaN
            if not nearby_zones.empty:
                joined.loc[idx, zone_col_right] = zones_proj.loc[nearby_zones.idxmin(), zone_col_left]

    return joined.to_crs(points_gdf.crs)


def assign_multi_source_grid_zones(
    points_gdf: gpd.GeoDataFrame,
    electricity_maps_gdf: gpd.GeoDataFrame,
    ecoinvent_gdf: gpd.GeoDataFrame,
    buffer_distance_m: float = 10_000,
) -> gpd.GeoDataFrame:
    """Assign zones from ElectricityMap, falling back to Ecoinvent for unmatched/CN points."""
    # First pass: Try ElectricityMap zones
    result = assign_grid_zones(
        points_gdf,
        electricity_maps_gdf,
        zone_col_right="power_grid_zone",
        zone_col_left="zoneName",
        use_buffer=True,
        buffer_distance_m=buffer_distance_m,
    )
    china_iso_2 = "CN"
    # Set CN zones to NaN since we'll use Ecoinvent for China
    result.loc[result["power_grid_zone"] == china_iso_2, "power_grid_zone"] = np.nan

    # Second pass: Try Ecoinvent zones for unmatched points
    unmatched_mask = result["power_grid_zone"].isna()
    if unmatched_mask.any():
        ecoinvent_zones = assign_grid_zones(
            result[unmatched_mask],
            ecoinvent_gdf,
            zone_col_right="power_grid_zone_ecoinvent",
            zone_col_left="shortname",
            use_buffer=True,
            buffer_distance_m=buffer_distance_m,
        )
        result.loc[unmatched_mask, "power_grid_zone"] = ecoinvent_zones["power_grid_zone_ecoinvent"]

    return result


def find_nearest_power_plant(data_center: pd.Series, power_plants: pd.DataFrame) -> pd.Series:
    """Find the nearest power plant for a given data center."""
    nearest_geom = nearest_points(data_center.geometry, power_plants.geometry.union_all())[1]
    nearest_power_plant = power_plants[power_plants.geometry == nearest_geom]
    return nearest_power_plant


def replace_zones_with_nearest(
    data_centers: gpd.GeoDataFrame,
    power_plants: gpd.GeoDataFrame,
    grid_zone_col: str = "power_grid_zone",
) -> pd.DataFrame:
    """Replace zones with no power plants in them with the nearest zone."""
    # Create a copy of the data centers and power plants
    data_centers = data_centers.copy()
    power_plants = power_plants.copy()

    # Find zones which are in data centers but not in power plants
    data_centers_zones = data_centers[grid_zone_col].unique()
    power_plants_zones = power_plants[grid_zone_col].unique()
    missing_zones = [zone for zone in data_centers_zones if zone not in power_plants_zones]

    logger.info("Missing zones: %s", missing_zones)
    logger.info(
        "Number of data centers in the missing zones: %s",
        len(data_centers[data_centers[grid_zone_col].isin(missing_zones)]),
    )

    # Filter data centers in the missing zones
    data_centers_missing_zones = data_centers[data_centers[grid_zone_col].isin(missing_zones)]

    # Apply the function to each data center in the missing zones
    nearest_power_plants = data_centers_missing_zones.apply(
        lambda row: find_nearest_power_plant(row, power_plants), axis=1
    )

    # Combine the results into a single DataFrame
    nearest_power_plants = pd.concat(nearest_power_plants.values)
    nearest_power_plants = nearest_power_plants.add_suffix("_1")
    data_centers_missing_zones = data_centers_missing_zones.add_suffix("_2")
    nearest_power_plants = pd.concat(
        [
            data_centers_missing_zones.reset_index(drop=True),
            nearest_power_plants.reset_index(drop=True),
        ],
        axis=1,
    )

    # Replace the zones in the original data center DataFrame based on company, name, and address
    for _, row in nearest_power_plants.iterrows():
        company = row["company_2"]
        name = row["name_2"]
        address = row["address_2"]
        new_zone = row[f"{grid_zone_col}_1"]

        # Find the matching row in the original data centers DataFrame
        matching_index = data_centers[
            (data_centers["company"] == company) & (data_centers["name"] == name) & (data_centers["address"] == address)
        ].index

        # Update the power_grid_zone for the matching row
        if not matching_index.empty:
            data_centers.at[matching_index[0], grid_zone_col] = new_zone

    return data_centers


def get_power_grid_stats(
    power_plants_df: pd.DataFrame,
    data_centers_df: pd.DataFrame,
    capacity_col: str = "capacity_mw",
    zone_col: str = "power_grid_zone",
    water_intensity_col: str = "water_intensity_m3/MWh",
) -> pd.DataFrame:
    """Find summary statistics for the power grid zones, including the weighted average water intensity, number of power plants, and total capacity.

    Args:
        power_plants_df (DataFrame): DataFrame containing power plants and their water intensities.
        data_centers_df (DataFrame): DataFrame containing data centers and their power grid zones.
        capacity_col (str): Column name for the power capacity of the power plants.
        zone_col (str): Column name for the power grid zones.
        water_intensity_col (str): Column name for the water intensity of the power plants.

    Returns:
        DataFrame: DataFrame containing the weighted average water intensity of each zone,
        along with the number of power plants and total capacity in each zone.
    """
    # Ensure numeric capacity
    power_plants_df[capacity_col] = pd.to_numeric(power_plants_df[capacity_col])

    # Calculate the average water intensity within each zone, weighted by the capacity of the power plant.
    power_plants_df["grid_contribution_weight"] = power_plants_df[water_intensity_col] * (
        power_plants_df[capacity_col] / power_plants_df.groupby(zone_col)[capacity_col].transform("sum")
    )

    # Calculate the total water intensity for each zone
    power_grid_summary = (
        power_plants_df.groupby(zone_col)
        .apply(
            lambda x: pd.Series(
                {
                    water_intensity_col: np.average(x[water_intensity_col], weights=x[capacity_col]),
                    "number_of_power_plants": x[zone_col].count(),
                    "total_capacity_mw": x[capacity_col].sum(),
                }
            )
        )
        .reset_index()
    )

    # Record the number of data centers in each zone
    data_centers_count = data_centers_df[zone_col].value_counts().reset_index()
    data_centers_count.columns = [zone_col, "number_of_data_centers"]

    # Merge the data centers count with the power plants water intensity weighted dataframe
    power_grid_summary = power_grid_summary.merge(data_centers_count, on=zone_col, how="left").fillna(
        {"number_of_data_centers": 0}
    )

    return power_grid_summary


def calculate_marginal_increase(row: pd.Series, summary: pd.DataFrame) -> pd.Series:
    """Calculate the marginal increase in electricity and water use for planned vs. operational data centers."""
    # Find the corresponding row for operational data centers
    corresponding_row = summary.query(
        "cooling_tech_scenario == @row['cooling_tech_scenario'] and "
        "tech_performance == @row['tech_performance'] and "
        "power_scenario == @row['power_scenario'] and "
        "only_operational == True"
    )

    # Calculate the marginal increase due to planned data centers
    if not corresponding_row.empty:
        row["marginal_electricity_increase_TWh"] = (
            row["total_electricity_use_TWh"] - corresponding_row.iloc[0]["total_electricity_use_TWh"]
        )
        row["marginal_direct_water_increase_m3"] = (
            row["direct_water_use_m3"] - corresponding_row.iloc[0]["direct_water_use_m3"]
        )
        row["marginal_indirect_water_increase_m3"] = (
            row["indirect_water_use_m3"] - corresponding_row.iloc[0]["indirect_water_use_m3"]
        )
        row["marginal_total_water_increase_m3"] = (
            row["total_water_use_m3"] - corresponding_row.iloc[0]["total_water_use_m3"]
        )
    return row


def results_summary(data_centers_df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary of the results for the total electricity and water use for each scenario."""
    # Create a DataFrame for both operational and all cases at once
    summary_df = pd.concat(
        [
            data_centers_df.assign(only_operational=True)[data_centers_df["operational"]],
            data_centers_df.assign(only_operational=False),
        ],
        ignore_index=True,
    )

    # Group and aggregate
    summary = (
        summary_df.groupby(
            [
                "cooling_tech_scenario",
                "tech_performance",
                "power_scenario",
                "only_operational",
            ]
        )
        .agg(
            total_electricity_use_TWh=("annual_electricity_use_MWh", lambda x: x.sum() / 1e6),
            direct_water_use_m3=("annual_direct_water_use_m3", "sum"),
            indirect_water_use_m3=("indirect_water_use_m3", "sum"),
            total_water_use_m3=("total_water_use_m3", "sum"),
        )
        .reset_index()
    )

    # Calculate the marginal increases due to planned data centers
    summary = summary.apply(
        lambda row: calculate_marginal_increase(row, summary) if not row["only_operational"] else row, axis=1
    )

    return summary


def results_average_wue_pue(data_centers_df: pd.DataFrame) -> pd.DataFrame:
    """Creates a summary of the average water use efficiency and power usage efficiency for each scenario."""
    # Add an "only_operational" column for operational status
    data_centers_df["only_operational"] = data_centers_df["operational"]

    # Create a copy of the data to add a combined category for all data centers (operational + non-operational)
    all_data_centers = data_centers_df.copy()
    all_data_centers["only_operational"] = False

    operational_data_centers = data_centers_df[data_centers_df["only_operational"]]

    # Concatenate the two DataFrames: one for only operational, one for all data centers
    combined_df = pd.concat([operational_data_centers, all_data_centers], ignore_index=True)

    # Group by all relevant columns, including "only_operational"
    grouped = combined_df.groupby(
        [
            "cooling_tech_scenario",
            "tech_performance",
            "power_scenario",
            "only_operational",
            "ashrae_zone",
            "size",
        ]
    )

    # Aggregate the sums of electricity and water use
    summary = grouped.agg(
        total_electricity_use_MWh=("annual_electricity_use_MWh", "sum"),
        total_direct_water_use_m3=("annual_direct_water_use_m3", "sum"),
        total_indirect_water_use_m3=("indirect_water_use_m3", "sum"),
        total_gross_power_MW=("critical_power_mw", "sum"),
    ).reset_index()

    # Calculate the average electricity and water use efficiency
    summary["electricity_use_efficiency_MWh_MWh"] = summary["total_electricity_use_MWh"] / (
        summary["total_gross_power_MW"] * 8760  # Hours in a year
    )
    summary["water_use_efficiency_m3_MWh"] = summary["total_direct_water_use_m3"] / (
        summary["total_gross_power_MW"] * 8760
    )

    return summary


def power_plant_water_use_for_scenario(
    scenario_df: pd.DataFrame,
    power_grid_summary: pd.DataFrame,
    power_plants_water_intensities: pd.DataFrame,
) -> pd.DataFrame:
    """Finds the data-center-driven water use at power plants for a given scenario."""
    # Calculate total water use per zone
    total_indirect_water_use = scenario_df.groupby("power_grid_zone")["indirect_water_use_m3"].sum().reset_index()

    # Process through power plants
    weighted_intensity = power_grid_summary.merge(total_indirect_water_use, on="power_grid_zone", how="left").rename(
        columns={"indirect_water_use_m3": "total_indirect_water_use_m3"}
    )

    result = power_plants_water_intensities.merge(
        weighted_intensity[["power_grid_zone", "total_indirect_water_use_m3"]], on="power_grid_zone", how="left"
    ).rename(columns={"total_indirect_water_use_m3": "water_use_per_zone_m3"})

    # Calculate final values
    result["grid_contribution_weight"] = result["capacity_mw"] / result.groupby("power_grid_zone")[
        "capacity_mw"
    ].transform("sum")
    result["water_use_m3"] = result["grid_contribution_weight"] * result["water_use_per_zone_m3"]

    return result


def assign_water_use_to_power_plants(
    data_centers_with_water_intensity: pd.DataFrame,
    power_plants_water_intensity_weighted: pd.DataFrame,
    power_plants_water_intensities: pd.DataFrame,
    *,
    consider_op_status: bool = False,
) -> pd.DataFrame:
    """Main function with optional status consideration."""
    results = []

    # Get unique scenarios
    scenarios = [
        (c, p, t)
        for c in data_centers_with_water_intensity["cooling_tech_scenario"].unique()
        for p in data_centers_with_water_intensity["power_scenario"].unique()
        for t in data_centers_with_water_intensity["tech_performance"].unique()
    ]

    for cooling_tech, power_scenario, tech_level in scenarios:
        # Base scenario filter
        scenario_mask = (
            (data_centers_with_water_intensity["cooling_tech_scenario"] == cooling_tech)
            & (data_centers_with_water_intensity["power_scenario"] == power_scenario)
            & (data_centers_with_water_intensity["tech_performance"] == tech_level)
        )

        if consider_op_status:
            # Process each operational status separately
            for op_status in [True, False]:
                scenario_data = data_centers_with_water_intensity[
                    scenario_mask & (data_centers_with_water_intensity["operational"] == op_status)
                ]

                if len(scenario_data) > 0:
                    result = power_plant_water_use_for_scenario(
                        scenario_data, power_plants_water_intensity_weighted, power_plants_water_intensities
                    )

                    # Add scenario info and status
                    result = result.assign(
                        cooling_tech_scenario=cooling_tech,
                        power_scenario=power_scenario,
                        technology_performance_level=tech_level,
                        status="operational" if op_status else "planned",
                    )
                    results.append(result)
        else:
            # Process without status consideration
            scenario_data = data_centers_with_water_intensity[scenario_mask]

            if len(scenario_data) > 0:
                result = power_plant_water_use_for_scenario(
                    scenario_data, power_plants_water_intensity_weighted, power_plants_water_intensities
                )

                # Add scenario info only
                result = result.assign(
                    cooling_tech_scenario=cooling_tech,
                    power_scenario=power_scenario,
                    technology_performance_level=tech_level,
                )
                results.append(result)

    return pd.concat(results, ignore_index=True)


def combine_dcs_and_pps(
    data_centers_df: pd.DataFrame, power_plants_df: pd.DataFrame, status: str | None = None
) -> pd.DataFrame:
    """Create DataFrame with water use by location for data centers and power plants."""
    # Local variables
    status_all = "all"
    status_operational = "operational"

    if status is None:
        status = status_all

    # Define column mappings
    dc_columns = {
        "annual_direct_water_use_m3": "water_use_m3",
        "GU_A3": "country",
        "critical_power_mw": "tcp_mw",
        "tech_performance": "technology_performance_level",
    }

    # Select and rename data center columns
    data_centers = (
        data_centers_df[
            [
                "name",
                "ISO_A3",
                "latitude",
                "longitude",
                "critical_power_mw",
                "power_grid_zone",
                "annual_direct_water_use_m3",
                "power_scenario",
                "cooling_tech_scenario",
                "tech_performance",
                "operational",
            ]
        ]
        .assign(type="data_center")
        .rename(columns=dc_columns)
    )

    # Filter by operational status, planned or operational
    if status != status_all:
        data_centers = data_centers[data_centers["operational"] == (status == status_operational)]

    # Process power plants
    power_plants = (
        power_plants_df[
            power_plants_df["water_use_m3"].notna()
        ]  # Filter out power plants without data center water use
        .assign(operational=status in (status_all, status_operational), tcp_mw=0)
        .rename(columns={"primary_fuel": "type"})
    )

    # Filter power plants by status
    if status != status_all:
        power_plants = power_plants[power_plants["status"] == status]

    # Select final power plant columns
    power_plants = power_plants[
        [
            "name",
            "country",
            "latitude",
            "longitude",
            "power_grid_zone",
            "water_use_m3",
            "power_scenario",
            "cooling_tech_scenario",
            "technology_performance_level",
            "type",
            "operational",
            "tcp_mw",
        ]
    ]

    # Combine datasets
    return pd.concat([data_centers, power_plants], ignore_index=True)
