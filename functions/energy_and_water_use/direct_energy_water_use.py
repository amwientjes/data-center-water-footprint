"""Functions to calculate the direct energy and water use for data centers based on PUE and WUE values."""

import numpy as np
import pandas as pd


def create_case_column(data_centers_df: pd.DataFrame, size_to_case_mapping: dict[str, list[int]]) -> pd.DataFrame:
    """Assign a case to each data center based on its size category."""
    # Validate case numbers
    if not all(1 <= case <= 10 for cases in size_to_case_mapping.values() for case in cases):
        err_msg = "All case numbers must be between 1 and 10"
        raise ValueError(err_msg)

    # Create expanded dataframe
    df_repeated = pd.concat(
        [
            (size_df := data_centers_df[data_centers_df["size"] == size])
            .loc[size_df.index.repeat(len(cases))]
            .assign(case=np.tile(cases, len(size_df)))
            for size, cases in size_to_case_mapping.items()
        ],
        ignore_index=True,
    )

    return df_repeated


def create_tech_perf_column(
    data_centers_df: pd.DataFrame, tech_perf_levels: tuple[str, ...] = ("best", "medium", "worst")
) -> pd.DataFrame:
    """Create technological performance levels by repeating dataframe."""
    # Create expanded dataframe
    df_repeated = data_centers_df.loc[data_centers_df.index.repeat(len(tech_perf_levels))].reset_index(drop=True)

    # Assign performance levels
    df_repeated["tech_performance"] = np.tile(tech_perf_levels, len(data_centers_df))

    return df_repeated


def assign_pue_wue(
    data_centers_df: pd.DataFrame,
    pue_wue_df: pd.DataFrame,
    size_to_case_mapping: dict[str, list[int]],
    tech_perf_level_to_quantile_mapping: dict[str, int],
    conversion_factor_consumption_to_withdrawal: float = 1.3,  # Conversion factor from WUE consumption to withdrawal
) -> pd.DataFrame:
    """Assign PUE and WUE values to data centers based on technology case and performance levels."""
    # Copy the data centers DataFrame to avoid modifying the original
    data_centers_df = data_centers_df.copy()

    # First, create the case column
    data_centers_df = create_case_column(data_centers_df, size_to_case_mapping)

    # Then, create the technology performance level column
    data_centers_df = create_tech_perf_column(data_centers_df, tuple(tech_perf_level_to_quantile_mapping.keys()))

    # Map technology performance levels to quantile values
    data_centers_df["quantile"] = data_centers_df["tech_performance"].map(tech_perf_level_to_quantile_mapping)

    # Merge the data centers with the PUE and WUE information
    merged_df = data_centers_df.merge(
        pue_wue_df,
        left_on=["case", "quantile", "ashrae_zone"],
        right_on=["Case", "quantile", "climate zone"],
        how="left",
    )

    # Rename WUE column to 'WUE_consumption'
    merged_df = merged_df.rename(columns={"WUE": "WUE_consumption"})

    # Create a new column for WUE withdrawal
    merged_df["WUE_withdrawal"] = merged_df["WUE_consumption"] * conversion_factor_consumption_to_withdrawal

    # Drop the redundant 'climate zone' and 'quantile' columns from the merged DataFrame
    return merged_df.drop(columns=["climate zone", "quantile"])


def assign_scenarios(data_centers_df: pd.DataFrame, scenario_mappings: dict[str, dict[int, list[str]]]) -> pd.DataFrame:
    """Assigns scenarios based on size and case mappings.

    Args:
        data_centers_df: DataFrame with data centers information.
        scenario_mappings: Dictionary with size and case mappings to scenarios.

    Returns: DataFrame with cooling technology scenarios assigned based on size and case.
    """
    # Create result DataFrame
    result_dfs = []

    # Process each size category
    for size, case_scenarios in scenario_mappings.items():
        size_mask = data_centers_df["size"] == size

        # Process each case number
        for case, scenarios in case_scenarios.items():
            case_mask = data_centers_df["case"] == case

            # Create copies for each scenario
            for scenario in scenarios:
                scenario_df = data_centers_df[size_mask & case_mask].copy()
                scenario_df["cooling_tech_scenario"] = scenario
                result_dfs.append(scenario_df)

    return pd.concat(result_dfs, ignore_index=True)
