"""This script contains functions to impute missing data to the data center information DataFrame."""

from collections.abc import Callable
from enum import StrEnum

from pandas import DataFrame, Series


class PowerCapacityScenario(StrEnum):
    """Enum class for aggregation methods.

    Each method represents an imputation method for floor area from the big 5 tech companies.
    """

    MIN = "min"
    MAX = "max"
    AVG = "avg"

    @property
    def imputation_method(self) -> Callable[[Series], float]:
        match self:
            case self.MIN:
                return Series.min
            case self.MAX:
                return Series.max
            case self.AVG:
                return Series.mean


def impute_missing_values(
    df: DataFrame,
    company_name: str,
    power_capacity_scenario: PowerCapacityScenario,
    name_should_not_contain: str | None = None,
    target_column: str = "total_space_m2",
    nan_columns: tuple[str, ...] = ("white_space_m2", "critical_power_mw"),
) -> DataFrame:
    """Fill missing values in the target column for a specific company using the specified aggregation method.

    Args:
        df (DataFrame): The DataFrame containing the datacenter features.
        company_name (str): The name of the company to filter the data.
        power_capacity_scenario (PowerCapacityScenario): The power capacity scenario to use for imputation.
        name_should_not_contain (str | None): Optional string that excludes datacenters with this string in their name.
        target_column (str): The column in which to fill missing values.
        nan_columns (tuple[str, ...]): The columns that should be NaN to fill the target column.

    Returns:
        DataFrame: The DataFrame with missing values filled in the target column.
    """
    # Find company data
    company_mask = df["company"] == company_name
    agg_mask = company_mask.copy()

    # Don't use datacenters with specific string in the name to calculate fill value
    if name_should_not_contain:
        agg_mask &= ~df["name"].str.contains(name_should_not_contain, case=False)

    # Get fill value
    fill_value = df[agg_mask][target_column].agg(power_capacity_scenario.imputation_method)

    # Fill only where the target column and the specified NaN columns are NaN
    fill_mask = company_mask & df[target_column].isna() & df[list(nan_columns)].isna().all(axis=1)

    # Return full dataframe with imputed values.
    # Note that where() keeps original values for True condition, so we invert the mask
    return df.where(~fill_mask, df.fillna({target_column: fill_value}))
