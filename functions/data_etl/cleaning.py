"""This module contains functions for cleaning and transforming data."""

import numpy as np


def convert_area_to_sqm(area: int | float | str) -> float:
    """Cleans, checks, and converts the manually collected floor space information to square meters.

    There are four options for the input value:
    - A float or int, which is already in square meters. If the value is 0, it is converted to NaN.
    - A string with a number followed by 'sq.f.' or'f', which is in square feet.
    - A string with a number followed by 'sq.m.', which is in square meters.
    - An empty string or a dash, which is converted to NaN.
    """
    if isinstance(area, int | float):
        if area == 0:
            return np.nan
        return area
    if isinstance(area, str):
        area = area.strip()

        # If the string is empty or a dash, return NaN
        if area in ("", "-"):
            return np.nan

        # Extract the value before the first whitespace and remove commas
        area_num = area.split(" ")[0].replace(",", "")

        # Convert to float
        area_num = float(area_num)

        # Check if the value is in square feet and convert to square meters if needed
        if "f" in area.lower():
            return area_num * 0.092903
        return float(area_num)

    err_msg = "Invalid input type. Please provide a string or a number."
    raise ValueError(err_msg)
