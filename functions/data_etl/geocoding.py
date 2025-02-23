"""Functions for geocoding addresses with the Google Maps API."""

import json
import logging
from pathlib import Path

import geocoder
import pandas as pd
import pycountry_convert as pc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def geocode_addresses(addresses: list[str], api_key: str, output_path: Path) -> None:
    """Geocode a list of addresses using the Google Geocoding API and store the results in a file."""
    address_count = len(addresses)

    for i, address in enumerate(addresses):
        response = geocoder.google(address, key=api_key)

        if i % 100 == 0:  # Print progress every 100 addresses.
            logger.info("Geocoding address %d out of %d (%d%%)", i, address_count, i / address_count * 100)

        # Write results to a file within the loop to avoid losing data if the process is interrupted
        with output_path.open("a") as file:
            file.write(f"INDEX: {i}\nADDRESS: {address}\nGEOCODE_JSON: {response.json}\n")


def extract_geocoded_data_from_txt(file_path: Path) -> pd.DataFrame:
    """Extract geocoded data from a text file and return it as a DataFrame."""
    # Initialize list to store the extracted data
    data = []

    # Read the file, spitting the content into entries based on the INDEX: keyword
    with Path(file_path).open() as file:
        entries = file.read().split("INDEX: ")[1:]

    for entry in entries:
        # Extract the address and geocode JSON from the text
        address = entry.split("ADDRESS: ")[1].split("\nGEOCODE_JSON:")[0].strip()
        geocode_json: dict[str, float | int | str] = json.loads(entry.split("GEOCODE_JSON: ")[1].strip())

        # Append the extracted data to the list
        data.append({"address": address, "latitude": geocode_json.get("lat"), "longitude": geocode_json.get("lng")})

    return pd.DataFrame(data)


def country_alpha3_to_alpha2(alpha_3_code: str) -> str:
    """Convert a country ISO_3 code to an ISO_2 code.

    A custom mapping is used for some countries not available in pycountry_convert.
    """
    custom_mappings = {
        "KOS": "XK",
        "XKX": "XK",
        "TWN": "TW",
        "PSX": "PS",
        "PSE": "PS",
    }
    if alpha_3_code in custom_mappings:
        return custom_mappings[alpha_3_code]
    return pc.country_alpha3_to_country_alpha2(alpha_3_code)


def get_continent_from_country(alpha_3_code: str) -> str | None:
    """This function converts a country ISO_3 code to a continent."""
    try:
        # Get the alpha-2 country code
        alpha_2_code = country_alpha3_to_alpha2(alpha_3_code)
        # Get the continent based on the country code
        continent = pc.country_alpha2_to_continent_code(alpha_2_code)
    except KeyError:
        return None
    else:
        return continent
