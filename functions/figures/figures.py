"""This module contains functions to create results figures."""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from cartopy import feature as cfeature
from matplotlib import ticker
from matplotlib.offsetbox import AnchoredText


def calculate_water_use_by_basin(
    water_use: gpd.GeoDataFrame,
    sub_basin_boundaries: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """This function calculates the total water use by basin based on the water use data, the basins shapefile."""
    # Make a copy of the water use dataframe
    water_use = water_use.copy()

    # Create subset of basins just for the geometry and the sub basin ID
    sub_basin_boundaries = sub_basin_boundaries[["HYBAS_ID", "geometry"]]

    # Spatial join the water use data to the basins shapefile
    water_use_by_subbasin = gpd.sjoin(water_use, sub_basin_boundaries, predicate="within")

    # Rename 'water_use_m3' to 'dc_annual_water_use_m3'
    water_use_by_subbasin = water_use_by_subbasin.rename(columns={"water_use_m3": "dc_annual_water_use_m3"})

    # Group by the HYBAS_ID and sum based on water use
    water_use_by_basin_grouped = water_use_by_subbasin.groupby("HYBAS_ID")["dc_annual_water_use_m3"].sum().reset_index()

    # Merge the water use data back to the basins shapefile
    water_use_per_basin = sub_basin_boundaries.merge(water_use_by_basin_grouped, on="HYBAS_ID", how="left")

    # Drop the basins were no data center driven water use occurs or is 0
    water_use_per_basin = water_use_per_basin[
        (water_use_per_basin["dc_annual_water_use_m3"].notna()) & (water_use_per_basin["dc_annual_water_use_m3"] != 0)
    ]

    return water_use_per_basin


def calculate_water_use_by_basin2(
    water_use: gpd.GeoDataFrame,
    sub_basin_boundaries: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Sum water use by hydrological basin."""
    # Select only necessary columns from basins
    basins = sub_basin_boundaries[["HYBAS_ID", "geometry"]]

    # Rename water use column
    water_use = water_use.rename(columns={"water_use_m3": "dc_annual_water_use_m3"})

    # Join and aggregate water use by basin
    water_use_by_basin = (
        gpd.sjoin(water_use, basins, predicate="within").groupby("HYBAS_ID").agg({"water_use_m3": "sum"}).reset_index()
    )

    # Merge back with basin geometries and filter non-zero values
    return (
        basins.merge(water_use_by_basin, on="HYBAS_ID", how="left")
        .query("dc_annual_water_use_m3 > 0")
        .dropna(subset=["dc_annual_water_use_m3"])
    )


def plot_water_use_map(
    water_use_per_basin: gpd.GeoDataFrame,
    basins_level5: gpd.GeoDataFrame,
    data_center_baseline: pd.DataFrame,
    figure_dir: Path,
    status: str = "all",
    figsize: tuple = (15, 15),
    cmap: str = "BuPu",
) -> None:
    """Create map of data center driven water use per watershed sub basin.

    This includes an inlay for each the top 5 water-consuming countries.
    """
    # Filter data by status
    if status != "all":
        if status == "operational":
            data_center_baseline = data_center_baseline[data_center_baseline["operational"]]
        elif status == "planned":
            data_center_baseline = data_center_baseline[~data_center_baseline["operational"]]

    # Calculate water use per country for top 5
    water_use_per_country = (
        data_center_baseline.groupby("ADMIN").agg({"total_water_use_m3": "sum", "geometry": "first"}).reset_index()
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": ccrs.Robinson()})

    # Main map settings
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="white", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

    # Set global extent
    ax.set_global()
    ax.spines["geo"].set_edgecolor("black")
    ax.spines["geo"].set_linewidth(2)

    # Plot log-transformed water use in thousands of m3
    water_use_per_basin["water_use_log_thousand_m3"] = np.log10((water_use_per_basin["dc_annual_water_use_m3"]) / 1_000)
    water_use_per_basin.plot(
        column="water_use_log_thousand_m3",
        ax=ax,
        transform=ccrs.PlateCarree(),
        legend=True,
        cmap=cmap,
        legend_kwds={"label": "Water use (thousand m3)", "shrink": 0.4},
    )

    # Customize legend
    cbar = ax.get_figure().get_axes()[1]
    ticks_loc = cbar.get_yticks()
    cbar.yaxis.set_major_locator(ticker.FixedLocator(ticks_loc))
    cbar.set_yticklabels([f"{10**tick:.0f}" if 10**tick >= 1 else f"{10**tick:.2f}" for tick in ticks_loc])

    # Plot the sub basin boundaries
    basins_level5.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), color="grey", linewidth=0.05, zorder=1)

    # Add a box with the top 5 country water users
    top_5_water_use = water_use_per_country.nlargest(5, "total_water_use_m3")
    top_5_water_use["text"] = (
        top_5_water_use["ADMIN"]
        + ": "
        + (top_5_water_use["total_water_use_m3"] / 1e6).round(0).astype(int).astype(str)
        + " million m3"
    )
    anchored_text = AnchoredText(
        "Top 5 water users\n" + "\n".join(top_5_water_use["text"]),
        loc="lower left",
        prop={"size": 9},
        frameon=True,
    )
    anchored_text.patch.set_boxstyle("round,pad=0.5,rounding_size=0.5")
    ax.add_artist(anchored_text)

    ax.set_axis_off()

    # Save plot
    plt.savefig(f"{figure_dir}/water_use_by_basin_{status}.png", dpi=300, bbox_inches="tight")


def plot_vulnerability_at_extraction_sites(
    water_scarcity_summary: gpd.GeoDataFrame,
    figure_dir: Path,
    warming_scenario: str,
    vulnerability_column_name: str,
    save: bool = True,
) -> None:
    """Create a map of water scarcity at data center extraction sites under a given climate change scenario."""
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={"projection": ccrs.Robinson()})

    # Main map settings
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="white", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

    # Set global extent
    ax.set_global()
    ax.spines["geo"].set_edgecolor("black")
    ax.spines["geo"].set_linewidth(2)

    # Plot water scarcity points
    water_scarcity_summary.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        color="lightgrey",
        alpha=0.7,
        markersize=water_scarcity_summary["dc_cumulative_monthly_water_use_m3"] / 30000,
        marker="o",
    )

    # Define bins and colors
    bins = [0, 3, 5, 8, 12]
    colors = ["lightgrey"] + [plt.cm.plasma_r(i) for i in np.linspace(0.1, 0.95, len(bins) - 1)]

    unique_geometries = water_scarcity_summary.drop_duplicates(subset="geometry")
    for i in range(len(bins) - 1):
        bin_data = unique_geometries[
            (unique_geometries[vulnerability_column_name] > bins[i])
            & (unique_geometries[vulnerability_column_name] <= bins[i + 1])
        ]
        bin_data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            color=colors[i + 1],
            markersize=np.sqrt(bin_data["dc_cumulative_monthly_water_use_m3"]) / 10,
            marker="o",
        )

    # Legends
    legend_labels = ["0 months", "1-3 months", "3-5 months", "5-8 months", "8-12 months"]
    legend_patches = [
        plt.Line2D([0], [0], color=color, marker="o", markersize=10, linestyle="", label=label)
        for color, label in zip(colors, legend_labels, strict=False)
    ]

    sizes = [1000, 5000, 10000, 25000]
    size_labels = [f"{int(size):,}" for size in sizes]
    size_patches = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            markersize=np.sqrt(size) / 10,
            linestyle="",
            markerfacecolor="none",
            label=label,
        )
        for size, label in zip(sizes, size_labels, strict=False)
    ]

    # Modify legend placement and styling
    legend1 = ax.legend(
        handles=legend_patches,
        title="Increase in vulnerability",
        loc="lower left",
        bbox_to_anchor=(0, 0.2),
        ncol=1,
        framealpha=1,
        facecolor="white",
    )
    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=size_patches,
        title="Monthly water demand (m3)",
        loc="lower left",
        ncol=1,
        framealpha=1,
        facecolor="white",
    )
    ax.add_artist(legend2)

    # Save the plot if SAVE is True
    if save:
        plt.savefig(f"{figure_dir}/vulnerable_extraction_sites_{warming_scenario}", dpi=300, bbox_inches="tight")

    # Show plot
    plt.show()


def calculate_pp_increased_ws_share(
    water_scarcity_summary: gpd.GeoDataFrame, power_plants: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Calculate the share of each power plant capacity that experiences an increase in water scarcity months."""
    # Merge capacity with water_scarcity_summary based on name
    water_scarcity_summary_grid = water_scarcity_summary.merge(
        power_plants[["name", "capacity_mw"]], left_on="name", right_on="name", how="left"
    )

    # Calculate the ratio of each grid contribution weight within a power_grid_zone
    water_scarcity_summary_grid["share_of_grid_capacity"] = water_scarcity_summary_grid.groupby("power_grid_zone")[
        "capacity_mw"
    ].transform(lambda x: x / x.sum())

    # Find the portion of installed power capacity that each power plant helps supply
    water_scarcity_summary_grid["tcp_mw_share_pp"] = (
        water_scarcity_summary_grid.groupby("power_grid_zone")["tcp_mw"].transform("sum")
        * water_scarcity_summary_grid["share_of_grid_capacity"]
    )

    # Calculate the tcp share which experiences an increase in water scarcity
    water_scarcity_summary_grid["tcp_mw_share_pp_increase"] = water_scarcity_summary_grid.apply(
        lambda row: row["tcp_mw_share_pp"] if row["months_WSI_increase"] > 0 else 0, axis=1
    )

    # Repeat for months_WSI_0p6_increase
    water_scarcity_summary_grid["tcp_mw_share_pp_0p6_increase"] = water_scarcity_summary_grid.apply(
        lambda row: row["tcp_mw_share_pp"] if row["months_WSI_0p6_increase"] > 0 else 0, axis=1
    )

    return water_scarcity_summary_grid


def calculate_total_data_center_capacity_at_risk(water_scarcity_summary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate total capacity at risk for each data center, accounting for both direct and indirect risks."""
    # Calculate grid-level statistics
    grid_stats = (
        water_scarcity_summary.groupby("power_grid_zone")
        .agg({"tcp_mw_share_pp_increase": "sum", "tcp_mw_share_pp_0p6_increase": "sum", "tcp_mw": "sum"})
        .rename(
            columns={
                "tcp_mw_share_pp_increase": "tcp_mw_share_pp_increase_per_grid",
                "tcp_mw_share_pp_0p6_increase": "tcp_mw_share_pp_0p6_increase_per_grid",
                "tcp_mw": "tcp_mw_per_grid",
            }
        )
    )

    # Calculate risk metrics
    result = water_scarcity_summary.merge(grid_stats, on="power_grid_zone", how="left")

    # Calculate relative contributions and risks
    result["relative_dc_tcp_in_grid"] = result["tcp_mw"] / result["tcp_mw_per_grid"]

    # Calculate indirect and direct risks
    result["indirect_tcp_increased_risk"] = (
        result["tcp_mw_share_pp_increase_per_grid"] * result["relative_dc_tcp_in_grid"]
    )
    result["indirect_tcp_increased_risk_0p6"] = (
        result["tcp_mw_share_pp_0p6_increase_per_grid"] * result["relative_dc_tcp_in_grid"]
    )
    result["direct_tcp_increased_risk"] = np.where(result["months_WSI_increase"] > 0, result["tcp_mw"], 0)
    result["direct_tcp_increased_risk_0p6"] = np.where(result["months_WSI_0p6_increase"] > 0, result["tcp_mw"], 0)

    # Calculate total risks
    result["total_capacity_at_risk"] = np.where(
        result["direct_tcp_increased_risk"] > 0,
        result["direct_tcp_increased_risk"],
        result["indirect_tcp_increased_risk"],
    )
    result["total_capacity_at_risk_0p6"] = np.where(
        result["direct_tcp_increased_risk_0p6"] > 0,
        result["direct_tcp_increased_risk_0p6"],
        result["indirect_tcp_increased_risk_0p6"],
    )

    return result


def plot_relative_increase_ws_map(scenario_data: gpd.GeoDataFrame, figure_dir: Path, scenario_name: str) -> None:
    """Plot the relative index of water scarcity increase for direct cooling water use of data centers."""
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={"projection": ccrs.Robinson()})

    # Main map settings
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor="white", linewidth=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

    # Set global extent
    ax.set_global()
    ax.spines["geo"].set_edgecolor("black")
    ax.spines["geo"].set_linewidth(2)

    # Create a subset of the data where months_WSI_dc is greater than 0
    scenario_subset = scenario_data[scenario_data["months_WSI_dc"] > 0]

    # Define color bins based on relative_index
    bins = [0, 0.01, 0.05, 0.1, 0.5, 1, 15]
    colors = plt.cm.coolwarm(np.linspace(0.5, 1, len(bins) - 1))

    # Plot each bin with corresponding color
    for i in range(len(bins) - 1):
        bin_data = scenario_subset[
            (scenario_subset["relative_index"] > bins[i]) & (scenario_subset["relative_index"] <= bins[i + 1])
        ]
        # Only plot the unique geometries
        bin_data = bin_data.drop_duplicates(subset="geometry")
        bin_data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            color=colors[i],
            markersize=np.sqrt(bin_data["dc_cumulative_monthly_water_use_m3"]) / 10,
            marker="o",
        )

    # Legends
    legend_labels = ["0-0.05", "0.05-0.1", "0.1-0.3", "0.3-0.7", "0.7-1", ">1"]
    legend_patches = [
        plt.Line2D([0], [0], color=color, marker="o", markersize=10, linestyle="", label=label)
        for color, label in zip(colors, legend_labels, strict=False)
    ]

    sizes = [1000, 5000, 10000, 25000]
    size_labels = [f"{int(size):,}" for size in sizes]
    size_patches = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            markersize=np.sqrt(size) / 10,
            linestyle="",
            markerfacecolor="none",
            label=label,
        )
        for size, label in zip(sizes, size_labels, strict=False)
    ]

    # Modify legend placement and styling
    legend1 = ax.legend(
        handles=legend_patches,
        title="Increase in WSI",
        loc="lower left",
        bbox_to_anchor=(0, 0.2),
        ncol=1,
        framealpha=1,
        facecolor="white",
    )
    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=size_patches,
        title="Monthly water demand (m3)",
        loc="lower left",
        ncol=1,
        framealpha=1,
        facecolor="white",
    )
    ax.add_artist(legend2)

    # Save the plot
    plt.savefig(
        f"{figure_dir}/{scenario_name}_dc_relative_increase_wsi.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Show plot
    plt.show()


def plot_exacerbate_tip_water_scarcity_barchart(
    scenarios: list[str],
    bins: list[tuple[int, int]],
    bin_labels: list[str],
    colors: list[str],
    tip_into_water_scarcity_counts: pd.DataFrame,
    exacerbate_water_scarcity_counts: pd.DataFrame,
    error_tip_into_water_scarcity: pd.DataFrame,
    error_exacerbate_water_scarcity: pd.DataFrame,
    figure_dir: Path,
) -> None:
    """Plot stacked bars showing data centers that tip into or exacerbate water scarcity."""
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # Define x positions for the bars
    x = np.arange(len(scenarios))

    # Define bar width
    bar_width = 0.3

    # Define a gap between bars
    gap = 0.1  # Adjust for spacing between bars

    # Adjust x positions for direct and indirect bars
    x_tipping = x - (bar_width / 2 + gap / 2)
    x_exacerbating = x + (bar_width / 2 + gap / 2)

    # Initialize bottom values for stacking
    bottom_tip_into_water_scarcity = np.zeros(len(scenarios))
    bottom_exacerbate_water_scarcity = np.zeros(len(scenarios))

    # Plot tipping into water scarcity bars with stacking
    for i, (start, end) in enumerate(bins):
        bin_data_tip_into_water_scarcity = tip_into_water_scarcity_counts.iloc[:, start : end + 1].sum(axis=1)
        ax.bar(
            x_tipping,
            bin_data_tip_into_water_scarcity,
            bottom=bottom_tip_into_water_scarcity,
            width=bar_width,
            color=colors[i % len(colors)],
            label=bin_labels[i] if i == 0 else "",
        )
        bottom_tip_into_water_scarcity += bin_data_tip_into_water_scarcity

    # Plot exacerbating water scarcity bars with stacking
    for i, (start, end) in enumerate(bins):
        bin_data_exacerbate_water_scarcity = exacerbate_water_scarcity_counts.iloc[:, start : end + 1].sum(axis=1)
        ax.bar(
            x_exacerbating,
            bin_data_exacerbate_water_scarcity,
            bottom=bottom_exacerbate_water_scarcity,
            width=bar_width,
            color=colors[i % len(colors)],
        )
        bottom_exacerbate_water_scarcity += bin_data_exacerbate_water_scarcity

    # Add error bars
    for i, scenario in enumerate(scenarios):
        total_tipping = tip_into_water_scarcity_counts.iloc[i, 1:].sum()
        total_exacerbating = exacerbate_water_scarcity_counts.iloc[i, 1:].sum()

        tipping_error = (
            error_tip_into_water_scarcity.loc[error_tip_into_water_scarcity["Scenario"] == scenario].iloc[0, 1:].sum()
        )
        exacerbating_error = (
            error_exacerbate_water_scarcity.loc[error_exacerbate_water_scarcity["Scenario"] == scenario]
            .iloc[0, 1:]
            .sum()
        )
        ax.errorbar(
            x_exacerbating[i],
            total_exacerbating,
            yerr=[[0], [exacerbating_error]],
            fmt="none",
            ecolor="grey",
            capsize=5,
        )
        ax.errorbar(x_tipping[i], total_tipping, yerr=[[0], [tipping_error]], fmt="none", ecolor="grey", capsize=5)

    # Add a total line adding the tipping and exacerbating values
    for i in range(len(scenarios)):
        total_tipping = tip_into_water_scarcity_counts.iloc[i, 1:].sum()
        total_exacerbating = exacerbate_water_scarcity_counts.iloc[i, 1:].sum()
        ax.hlines(
            total_tipping + total_exacerbating,
            x_tipping[i] - bar_width / 2,
            x_exacerbating[i] + bar_width / 2,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
        )

    # Add error bar for total based on tipping and exacerbating errors
    for i, scenario in enumerate(scenarios):
        total_tipping = tip_into_water_scarcity_counts.iloc[i, 1:].sum()
        total_exacerbating = exacerbate_water_scarcity_counts.iloc[i, 1:].sum()

        tipping_error = (
            error_tip_into_water_scarcity.loc[error_tip_into_water_scarcity["Scenario"] == scenario].iloc[0, 1:].sum()
        )
        exacerbating_error = (
            error_exacerbate_water_scarcity.loc[error_exacerbate_water_scarcity["Scenario"] == scenario]
            .iloc[0, 1:]
            .sum()
        )
        ax.errorbar(
            x[i],
            total_tipping + total_exacerbating,
            yerr=[[0], [tipping_error + exacerbating_error]],
            fmt="none",
            ecolor="grey",
            capsize=5,
        )

    # Labels and formatting
    ax.set_ylabel("Data centers (%)", fontsize=12)
    ax.set_xlabel("Global warming scenario", fontsize=12, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=0, fontsize=11)
    ax.tick_params(axis="x", labelrotation=0, pad=15)

    # Add labels for Direct and Indirect above x-axis tick labels
    for i in range(len(scenarios)):
        ax.text(x_tipping[i], -0.25, "Tip", ha="center", va="center", fontsize=10, color="black")
        ax.text(x_exacerbating[i], -0.25, "Exacerbate", ha="center", va="center", fontsize=10, color="black")

    # Adjust x-axis limits to ensure all bars are in view
    ax.set_xlim(-0.5, len(scenarios) - 0.5)

    # Add legend for the bins
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(bins))]
    legend1 = ax.legend(handles, bin_labels, title="Increased months of WS due to DC", loc="upper left", fontsize=10)
    ax.add_artist(legend1)

    # Add error bar to the legend
    error_handle = plt.Line2D([0], [0], color="grey", marker="|", markersize=10, linestyle="", label="60% EFR")
    legend2 = ax.legend(handles=[error_handle], loc="upper right", fontsize=10, bbox_to_anchor=(1, 1))
    ax.add_artist(legend2)

    # Add total line to the legend
    total_handle = plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Total")
    legend3 = ax.legend(handles=[total_handle], loc="upper right", fontsize=10, bbox_to_anchor=(1, 0.95))
    ax.add_artist(legend3)

    # Define y axis limits
    ax.set_ylim(0, 13)

    # Save the plot
    plt.savefig(figure_dir / "tipping_exacerbating_water_scarcity_barchart.png", dpi=300, bbox_inches="tight")

    # Layout adjustments
    plt.tight_layout()
    plt.show()
