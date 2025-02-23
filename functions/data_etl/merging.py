"""Fuzzy matching and merging for datacenter GeoDataFrames."""

import logging

import pandas as pd
from fuzzywuzzy import process

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_str(s: str) -> str:
    """Helper function to clean and normalize string values."""
    return " ".join(s.lower().strip().split())


def generate_fuzzy_matches(
    df1: pd.DataFrame, df2: pd.DataFrame, match_columns: list[str], country_col: str = "ISO_A3", threshold: int = 60
) -> pd.DataFrame:
    """Generate fuzzy matches between dataframes, matching only within same country groups.

    Args:
        df1: First DataFrame to match
        df2: Second DataFrame to match
        match_columns: Columns to use for fuzzy matching
        country_col: Column containing country codes, for pre-filtering
        threshold: Minimum score for matches
    """
    matches = []

    # Create merge keys
    for df in [df1, df2]:
        df["merge_key"] = df[match_columns].astype(str).agg(" ".join, axis=1).apply(normalize_str)

    # Process each country group
    for country in df1[country_col].unique():
        # Get country-specific data
        df1_country = df1[df1[country_col] == country].reset_index(drop=True)
        df2_country = df2[df2[country_col] == country]

        if df2_country.empty:
            continue

        df1_len = len(df1_country)
        logger.info(
            "Processing country %s: matching %d rows from df1 to %d rows from df2 ...",
            country,
            df1_len,
            len(df2_country),
        )

        # Find best matches within country
        for i, row in df1_country.iterrows():
            if i + 1 % 100 == 0:  # Print progress
                logger.info("Processing row %d (%d%%) ...", i, i + 1 / df1_len * 100)

            best_match = process.extractOne(row["merge_key"], df2_country["merge_key"])

            if best_match and best_match[1] >= threshold:
                matched_row = df2_country[df2_country["merge_key"] == best_match[0]].iloc[0]
                matches.append(
                    {
                        **{f"{k}_df1": v for k, v in row.items()},
                        **{f"{k}_df2": v for k, v in matched_row.items()},
                        "best_score": best_match[1],
                    }
                )

    return pd.DataFrame(matches) if matches else pd.DataFrame()
