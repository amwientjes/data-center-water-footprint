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


def plot_months_ws_at_extraction_sites(
    water_scarcity_summary: gpd.GeoDataFrame,
    figure_dir: Path,
    warming_scenario: str,
    ws_column_name: str,
    include_dc_contributions: bool = False,
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
        markersize=np.sqrt(water_scarcity_summary["dc_cumulative_monthly_water_use_m3"] * 5) / 3,
        marker="o",
    )

    # Define bins and colors
    bins = [0, 3, 5, 8, 12]
    if include_dc_contributions: # set opacity of colors based on scenario
        colors = ["lightgrey"] + [plt.cm.viridis_r(i) for i in np.linspace(0.2, 0.90, len(bins) - 1)]
    elif not include_dc_contributions and warming_scenario == "hist":
        colors = ["lightgrey"] + [plt.cm.YlOrRd(i) for i in np.linspace(0.2, 0.95, len(bins) - 1)]
    elif not include_dc_contributions and warming_scenario != "hist":
        colors = ["lightgrey"] + [plt.cm.plasma_r(i) for i in np.linspace(0.1, 0.95, len(bins) - 1)]

    unique_geometries = water_scarcity_summary.drop_duplicates(subset="geometry")
    if include_dc_contributions:
        # Exclude rows where WSI_dc_mean is inf, as the water withdrawal location could not be identified at these locations
        unique_geometries = unique_geometries[unique_geometries["WSI_dc_mean"] != np.inf]
    for i in range(len(bins) - 1):
        bin_data = unique_geometries[
            (unique_geometries[ws_column_name] > bins[i])
            & (unique_geometries[ws_column_name] <= bins[i + 1])
        ]
        if include_dc_contributions:
            # Plot with DC-specific styling (black edge, thicker linewidth)
            bin_data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            color=colors[i + 1],
            markersize=np.sqrt(bin_data["dc_cumulative_monthly_water_use_m3"] * 7) / 3,
            marker="o",
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
            linestyle="-" if bin_data["type"].iloc[0] == "data_center" else ":",
            )
        else:
            # Simple plotting when DC contributions are not included
            bin_data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            color=colors[i + 1],
            markersize=np.sqrt(bin_data["dc_cumulative_monthly_water_use_m3"] * 7) / 3,
            marker="o",
            alpha=0.8,
            edgecolor="none",
            linewidth=0,
            )

    # Legends
    legend_labels = ["0 months", "1-3 months", "3-5 months", "5-8 months", "8-12 months"]
    legend_patches = [
        plt.Line2D([0], [0], color=color, marker="o", markersize=10, linestyle="", label=label)
        for color, label in zip(colors, legend_labels, strict=False)
    ]
    # Modify legend placement and styling
    legend_title = (
        "Months of water scarcity" if warming_scenario == "hist" and not include_dc_contributions else "Increase in months of water scarcity"
    )
    legend1 = ax.legend(
        handles=legend_patches,
        title=legend_title,
        loc="lower left",
        ncol=1,
        framealpha=1,
        facecolor="white",
    )
    ax.add_artist(legend1)

    # Add custom concentric circles legend for marker sizes directly on the main figure
    import matplotlib.patches as mpatches
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    sizes = [1_000, 10_000, 100_000, 1_000_000]
    labels = [f"{int(s):,}" for s in sizes]
    # Scale up the marker sizes by multiplying by 7 to make circles smaller
    marker_sizes = [(np.sqrt(s * 5)/3)**0.53 for s in sizes]

    legend_ax = inset_axes(ax, width="10%", height="20%", loc="center left", bbox_to_anchor=(0, -0.18, 1, 1), bbox_transform=ax.transAxes, borderpad=2)
    legend_ax.axis("off")
    center_x = 0.5
    baseline_y = 0.2  # This is the y-coordinate for the bottom of all circles
    for msize, label in zip(reversed(marker_sizes), reversed(labels)):
        radius = msize / 100
        center_y = baseline_y + radius
        circle = mpatches.Circle((center_x, center_y), radius=radius, fill=False, edgecolor="black", linewidth=1)
        legend_ax.add_patch(circle)
        legend_ax.text(0.9, center_y + msize/130, label, va="center", fontsize=10)
    legend_ax.text(0.8, baseline_y + max(marker_sizes)/100 + 0.5, "Monthly water withdrawal (m3)", ha="center", va="bottom", fontsize=10)

    scenario_text = "Historical" if warming_scenario == "hist" else "1.5°C warming" if warming_scenario == "1_5C" else "2.0°C warming" if warming_scenario == "2_0C" else "3.2°C warming"
    # Add label in top left corner of the map
    ax.text(0.01, 0.99, scenario_text, transform=ax.transAxes, fontsize=12, fontweight="bold")

    # If DC contributions included, add a small legend indicating data centers vs power plants (solid vs dotted)
    if include_dc_contributions:
        from matplotlib.lines import Line2D

        # Use patch circles so we can control the edge linestyle (dotted for power plants)
        dc_handle = Line2D([0], [0], color="black", linewidth=2, linestyle="-")
        pp_handle = Line2D([0], [0], color="black", linewidth=2, linestyle=":")
        legend2 = ax.legend(
            handles=[dc_handle, pp_handle],
            labels=["data centers", "power plants"],
            loc="lower right",
            framealpha=1,
            facecolor="white",
            title="",
        )
        ax.add_artist(legend2)

    # Save the plot if SAVE is True
    if save:
        if include_dc_contributions:
            plt.savefig(f"{figure_dir}/dc_contributions_ws_{warming_scenario}", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(f"{figure_dir}/vulnerable_extraction_sites_{warming_scenario}", dpi=300, bbox_inches="tight")

    # Show plot
    plt.show()
    
def calculate_pp_increased_ws_share(
    water_scarcity_summary: gpd.GeoDataFrame,
    power_plants: gpd.GeoDataFrame,
    metric: str,
    sensitivity_analysis: bool = True,
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
        lambda row: row["tcp_mw_share_pp"] if row[f"months_{metric}_increase"] > 0 else 0, axis=1
    )

    if sensitivity_analysis:
        # Repeat for 0.6 EFR
        water_scarcity_summary_grid["tcp_mw_share_pp_0p6_increase"] = water_scarcity_summary_grid.apply(
            lambda row: row["tcp_mw_share_pp"] if row[f"months_{metric}_0p6_increase"] > 0 else 0, axis=1
        )

    return water_scarcity_summary_grid


def calculate_total_data_center_capacity_at_risk(
    water_scarcity_summary: gpd.GeoDataFrame, metric: str, sensitivity_analysis: bool = True
) -> gpd.GeoDataFrame:
    """Calculate total capacity at risk for each data center, accounting for both direct and indirect risks."""
    # Calculate grid-level statistics
    grid_stats = (
        water_scarcity_summary.groupby("power_grid_zone")
        .agg({"tcp_mw_share_pp_increase": "sum", "tcp_mw": "sum"})
        .rename(
            columns={
                "tcp_mw_share_pp_increase": "tcp_mw_share_pp_increase_per_grid",
                "tcp_mw": "tcp_mw_per_grid",
            }
        )
    )

    if sensitivity_analysis:
        grid_stats["tcp_mw_share_pp_0p6_increase_per_grid"] = water_scarcity_summary.groupby("power_grid_zone")[
            "tcp_mw_share_pp_0p6_increase"
        ].sum()

    # Calculate risk metrics
    result = water_scarcity_summary.merge(grid_stats, on="power_grid_zone", how="left")

    # Calculate relative contributions and risks
    result["relative_dc_tcp_in_grid"] = result["tcp_mw"] / result["tcp_mw_per_grid"]

    # Calculate indirect and direct risks
    result["indirect_tcp_increased_risk"] = (
        result["tcp_mw_share_pp_increase_per_grid"] * result["relative_dc_tcp_in_grid"]
    )
    result["direct_tcp_increased_risk"] = np.where(result[f"months_{metric}_increase"] > 0, result["tcp_mw"], 0)

    if sensitivity_analysis:
        result["indirect_tcp_increased_risk_0p6"] = (
            result["tcp_mw_share_pp_0p6_increase_per_grid"] * result["relative_dc_tcp_in_grid"]
        )
        result["direct_tcp_increased_risk_0p6"] = np.where(result[f"months_{metric}_0p6_increase"] > 0, result["tcp_mw"], 0)

    # Calculate total risks
    result["total_capacity_at_risk"] = np.where(
        result["direct_tcp_increased_risk"] > 0,
        result["direct_tcp_increased_risk"],
        result["indirect_tcp_increased_risk"],
    )

    if sensitivity_analysis:
        result["total_capacity_at_risk_0p6"] = np.where(
            result["direct_tcp_increased_risk_0p6"] > 0,
            result["direct_tcp_increased_risk_0p6"],
            result["indirect_tcp_increased_risk_0p6"],
        )

    return result

# TODO: Add comments to this function
def make_capacity_at_risk_boxplot(
    water_scarcity_summary: gpd.GeoDataFrame,
    water_scarcity_summary_dc: gpd.GeoDataFrame,
    power_plants: gpd.GeoDataFrame,
    GLOBAL_WARMING_SCENARIOS_FUTURE: list,
    FIGURES_DIR: Path,
    fig_name: str,
    geographical_scope: str,
    show_error_bars=True
):
    """
    Calculate and plot the percent of data center capacity at increased risk of water scarcity as a boxplot.
    """

    # Calculate the percentage of months with a WSI increase for each scenario
    future_scenarios_dict = {
        "1.5°C": water_scarcity_summary_dc["1_5C"],
        "2.0°C": water_scarcity_summary_dc["2_0C"],
        "3.2°C": water_scarcity_summary_dc["3_2C"],
    }
    water_scarcity_counts_direct = pd.DataFrame({"Scenario": list(future_scenarios_dict.keys())})
    for month in range(1, 13):
        water_scarcity_counts_direct[f"{month} month"] = [
            100 * (df.loc[df["months_WSI_increase"] == month, "tcp_mw"].sum() / df["tcp_mw"].sum())
            for df in future_scenarios_dict.values()
        ]
    error_data_direct = pd.DataFrame(
        {
            "Scenario": list(future_scenarios_dict.keys()),
            "Direct_increase_error": [
                100
                * (
                    df.loc[df["months_WSI_0p6_increase"] > 0, "tcp_mw"].sum()
                    - df.loc[df["months_WSI_increase"] > 0, "tcp_mw"].sum()
                )
                / df["tcp_mw"].sum()
                for df in future_scenarios_dict.values()
            ],
        }
    )

    # Calculate the share of each power plant's capacity that is affected by increased water scarcity
    water_scarcity_summary_boxplot = {}

    for scenario in GLOBAL_WARMING_SCENARIOS_FUTURE:
        water_scarcity_summary_boxplot[scenario] = calculate_pp_increased_ws_share(
            water_scarcity_summary[scenario], power_plants, "WSI"
        )

    # Calculate the total capacity at risk for each data center
    for scenario in GLOBAL_WARMING_SCENARIOS_FUTURE:
        water_scarcity_summary_boxplot[scenario] = calculate_total_data_center_capacity_at_risk(
            water_scarcity_summary_boxplot[scenario], "WSI"
        )

    # Calculate the water scarcity increase counts as percentages for each month from 1 to 12
    water_scarcity_counts_indirect = pd.DataFrame({"Scenario": ["1.5°C", "2.0°C", "3.2°C"]})
    for month in range(1, 13):
        water_scarcity_counts_indirect[f"{month} month"] = [
            100
            * (
                water_scarcity_summary_boxplot[scenario]
                .loc[water_scarcity_summary_boxplot[scenario]["months_WSI_increase"] == month, "tcp_mw_share_pp"]
                .sum()
                / water_scarcity_summary_boxplot[scenario]["tcp_mw_share_pp"].sum()
            )
            for scenario in GLOBAL_WARMING_SCENARIOS_FUTURE
        ]
    error_data_indirect = pd.DataFrame({"Scenario": ["1.5°C", "2.0°C", "3.2°C"]})
    error_data_indirect["Direct_increase_error"] = [
        100
        * (
            (
                water_scarcity_summary_boxplot[scenario]
                .loc[water_scarcity_summary_boxplot[scenario]["months_WSI_0p6_increase"] > 0, "tcp_mw_share_pp"]
                .sum()
            )
            - (
                water_scarcity_summary_boxplot[scenario]
                .loc[water_scarcity_summary_boxplot[scenario]["months_WSI_increase"] > 0, "tcp_mw_share_pp"]
                .sum()
            )
        )
        / water_scarcity_summary_boxplot[scenario]["tcp_mw_share_pp"].sum()
        for scenario in GLOBAL_WARMING_SCENARIOS_FUTURE
    ]

    # Calculate the total global capacity at risk for each scenario
    water_scarcity_total_capacity_at_risk = pd.DataFrame(
        {
            "Scenario": ["1.5°C", "2.0°C", "3.2°C"],
            "Total_capacity_at_risk": [
                100
                * water_scarcity_summary_boxplot[scenario]["total_capacity_at_risk"].sum()
                / water_scarcity_summary_boxplot[scenario]["tcp_mw"].sum()
                for scenario in GLOBAL_WARMING_SCENARIOS_FUTURE
            ],
        }
    )
    error_data_total = pd.DataFrame(
        {
            "Scenario": ["1.5°C", "2.0°C", "3.2°C"],
            "Total_capacity_at_risk_error": [
                100
                * (
                    water_scarcity_summary_boxplot[scenario]["total_capacity_at_risk_0p6"].sum()
                    / water_scarcity_summary_boxplot[scenario]["tcp_mw"].sum()
                )
                - 100
                * (
                    water_scarcity_summary_boxplot[scenario]["total_capacity_at_risk"].sum()
                    / water_scarcity_summary_boxplot[scenario]["tcp_mw"].sum()
                )
                for scenario in GLOBAL_WARMING_SCENARIOS_FUTURE
            ],
        }
    )
    total_tcp_gw = water_scarcity_summary_boxplot["1_5C"]["tcp_mw"].sum() / 1000

    # Plotting the boxplot
    future_scenarios_for_plotting = ["1.5°C", "2.0°C", "3.2°C"]
    bins = [(1, 3), (4, 6), (7, 9), (10, 12)]
    bin_labels = ["1-3", "4-6", "7-9", "10-12"] # Month ranges
    num_bins = len(bins)
    colors = plt.cm.plasma_r(np.linspace(0.13, 0.9, num_bins))
    fig, ax1 = plt.subplots(figsize=(7, 7))
    x = np.arange(len(future_scenarios_for_plotting))
    bar_width = 0.3
    gap = 0.1  # Spacing between bars
    x_direct = x + (bar_width / 2 + gap / 2)
    x_indirect = x - (bar_width / 2 + gap / 2)
    bottom_direct = np.zeros(len(future_scenarios_for_plotting))
    bottom_indirect = np.zeros(len(future_scenarios_for_plotting))

    # Plot Indirect bars with stacking (dashed edges)
    for i, (start, end) in enumerate(bins):
        bin_data_indirect = water_scarcity_counts_indirect.iloc[:, start : end + 1].sum(axis=1)
        ax1.bar(
            x_indirect,
            bin_data_indirect,
            bottom=bottom_indirect,
            width=bar_width,
            color=colors[i],
            label=bin_labels[i] if i == 0 else "",
        )
        bottom_indirect += bin_data_indirect

    # Plot Direct bars with stacking
    for i, (start, end) in enumerate(bins):
        bin_data_direct = water_scarcity_counts_direct.iloc[:, start : end + 1].sum(axis=1)
        ax1.bar(x_direct, bin_data_direct, bottom=bottom_direct, width=bar_width, color=colors[i])
        bottom_direct += bin_data_direct

    # Plot Total capacity at risk, as a horizontal line for each scenario
    for i, scenario in enumerate(future_scenarios_for_plotting):
        total_capacity = water_scarcity_total_capacity_at_risk.loc[
            water_scarcity_total_capacity_at_risk["Scenario"] == scenario, "Total_capacity_at_risk"
        ].to_numpy()[0]
        ax1.hlines(
            total_capacity,
            x_indirect[i] - bar_width / 2,
            x_direct[i] + bar_width / 2,
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
        )
        if show_error_bars:
            error = error_data_total.loc[error_data_total["Scenario"] == scenario, "Total_capacity_at_risk_error"].values[0]
            ax1.errorbar(x[i], total_capacity, yerr=[[0], [error]], fmt="none", ecolor="grey", capsize=5)
        print(f"Total capacity at risk for {scenario}: {total_capacity:.0f} %")

    if show_error_bars:
        for i, scenario in enumerate(future_scenarios_for_plotting):
            total_direct = water_scarcity_counts_direct.iloc[i, 1:].sum()
            total_indirect = water_scarcity_counts_indirect.iloc[i, 1:].sum()
            direct_error = error_data_direct.loc[error_data_direct["Scenario"] == scenario, "Direct_increase_error"].values[0]
            indirect_error = error_data_indirect.loc[
                error_data_indirect["Scenario"] == scenario, "Direct_increase_error"
            ].values[0]
            ax1.errorbar(x_direct[i], total_direct, yerr=[[0], [direct_error]], fmt="none", ecolor="grey", capsize=5)
            ax1.errorbar(x_indirect[i], total_indirect, yerr=[[0], [indirect_error]], fmt="none", ecolor="grey", capsize=5)

    # Labels and formatting
    ax1.set_ylabel(f"{geographical_scope} data center capacity at increased risk (%)", fontsize=12)
    ax1.set_xlabel("Global warming scenario", fontsize=12, labelpad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(future_scenarios_for_plotting, rotation=0, fontsize=11)
    ax1.tick_params(axis="x", labelrotation=0, pad=15 if show_error_bars else 20) # Adjust padding based on error bars which affect y-axis range
    for i in range(len(future_scenarios_for_plotting)):
        ax1.text(x_direct[i], -1.2, "Direct", ha="center", va="center", fontsize=10, color="black")
        ax1.text(x_indirect[i], -1.2, "Indirect", ha="center", va="center", fontsize=10, color="black")
    ax1.set_xlim(-0.5, len(future_scenarios_for_plotting) - 0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(bins))]
    legend1 = ax1.legend(handles, bin_labels, title="Increased months of WS", loc="upper left", fontsize=10)
    ax1.add_artist(legend1)
    total_handle = plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Total")
    legend2 = ax1.legend(handles=[total_handle], loc="upper left", fontsize=10, bbox_to_anchor=(0, 0.8))
    ax1.add_artist(legend2)
    if show_error_bars:
        error_handle = plt.Line2D([0], [0], color="grey", marker="|", markersize=10, linestyle="", label="60% EFR")
        legend3 = ax1.legend(handles=[error_handle], loc="upper left", fontsize=10, bbox_to_anchor=(0, 0.75))
        ax1.add_artist(legend3)
    ax2 = ax1.twinx()
    ax2.set_ylabel(f"{geographical_scope} data center capacity at increased risk (GW)", fontsize=12, rotation=270, labelpad=20)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_yticklabels([f"{ytick * total_tcp_gw / 100:.0f}" for ytick in ax1.get_yticks()])

    # Save the plot
    plt.savefig(FIGURES_DIR / fig_name, bbox_inches="tight", dpi=300)

    plt.tight_layout()
    plt.show()

def filter_by_country(data_dict, country, country_column = "country", include=True):
    if include:
        return {scenario: df[df[country_column] == country] for scenario, df in data_dict.items()}
    else:
        return {scenario: df[df[country_column] != country] for scenario, df in data_dict.items()}


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
    show_error_bars=False,
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
    if show_error_bars:
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
    if show_error_bars:
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
    if show_error_bars:
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


def plot_exacerbate_tip_water_scarcity_barchart(
    scenarios,
    tip_into_water_scarcity_counts: pd.DataFrame,
    exacerbate_water_scarcity_counts: pd.DataFrame,
    error_tip_into_water_scarcity: pd.DataFrame,
    error_exacerbate_water_scarcity: pd.DataFrame,
    figure_dir: Path,
    show_error_bars: bool = False,
) -> None:
    """Plot stacked bars showing data centers that tip into or exacerbate water scarcity."""
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))

    # scenarios = ["Historical", "1.5°C", "2.0°C", "3.2°C"]
    bins = [(1, 3), (4, 6), (7, 9), (10, 12)]
    bin_labels = ["1-3", "4-6", "7-9", "10-12"]
    colors = plt.cm.viridis_r(np.linspace(0.2, 1, len(bins)))

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
    if show_error_bars:
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
    if show_error_bars:
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

    scenarios_for_labels = ["Historical", "1.5°C", "2.0°C", "3.2°C"]

    # Labels and formatting
    ax.set_ylabel("Data centers (%)", fontsize=12)
    ax.set_xlabel("Global warming scenario", fontsize=12, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_for_labels, rotation=0, fontsize=11)
    ax.tick_params(axis="x", labelrotation=0, pad=15)

    # Add labels for Direct and Indirect above x-axis tick labels
    for i in range(len(scenarios)):
        ax.text(x_tipping[i], -0.15, "Tip", ha="center", va="center", fontsize=10, color="black")
        ax.text(x_exacerbating[i], -0.15, "Exacerbate", ha="center", va="center", fontsize=10, color="black")

    # Adjust x-axis limits to ensure all bars are in view
    ax.set_xlim(-0.5, len(scenarios) - 0.5)

    # Add legend for the bins
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(bins))]
    legend1 = ax.legend(handles, bin_labels, title="Increased months of WS due to DC", loc="upper left", fontsize=10)
    ax.add_artist(legend1)

    # Add error bar to the legend
    if show_error_bars:
        error_handle = plt.Line2D([0], [0], color="grey", marker="|", markersize=10, linestyle="", label="60% EFR")
        legend2 = ax.legend(handles=[error_handle], loc="upper left", fontsize=10, bbox_to_anchor=(0, 0.75))
        ax.add_artist(legend2)

    # Add total line to the legend
    total_handle = plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1, label="Total")
    legend3 = ax.legend(handles=[total_handle], loc="upper left", fontsize=10, bbox_to_anchor=(0, 0.8))
    ax.add_artist(legend3)

    # Define y axis limits
    ax.set_ylim(0, 5)

    # Save the plot
    plt.savefig(figure_dir / "tipping_exacerbating_water_scarcity_barchart.png", dpi=300, bbox_inches="tight")

    # Layout adjustments
    plt.tight_layout()
    plt.show()
