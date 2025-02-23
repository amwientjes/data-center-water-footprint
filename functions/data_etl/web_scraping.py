"""Functions used for web scraping data center information from datacenters.com with httpx."""

import logging
from pathlib import Path
from random import uniform
from time import sleep

import httpx
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def human_like_sleep(min_time_s: int = 1, max_time_s: int = 2) -> None:
    """Sleeps for a random time mimic human behavior on websites."""
    sleep(uniform(min_time_s, max_time_s))  # noqa: S311 # Pseudo-random sleep time is acceptable for this use case


BASE_URL = "https://www.datacenters.com"
API_PATH = "/api/v1/locations"


# Fetch basic data center info via API
def fetch_datacenter_com_master_list(
    output_path: Path,
    start_page: int = 1,
    end_page: int | None = None,
) -> pd.DataFrame:
    """This function constructs the master list of data centers from datacenters.com."""
    master_list = []
    current_page = start_page

    logger.info("Starting to fetch data center info from page %s ... ", current_page)

    while True:
        try:
            response = httpx.get(f"{BASE_URL + API_PATH}?page={current_page}", timeout=10)
        except httpx.RequestError:
            logger.exception("An error occurred on page %s", current_page)
            break

        data = response.json()

        # Extract data and append to master list
        data_centers = [
            {
                "name": dc["name"],
                "address": dc["fullAddress"],
                "url": BASE_URL + dc["url"],  # Append base URL to relative URL
                "company": dc["providerName"],
                "latitude": dc["latitude"],
                "longitude": dc["longitude"],
            }
            for dc in data["preloadedSearchLocations"]
        ]
        master_list.extend(data_centers)

        if current_page % 10 == 0:
            logger.info("  Page %s done", current_page)

        # Move to next page
        if current_page >= data["totalPages"] or (end_page and current_page >= end_page):
            break
        current_page += 1
        human_like_sleep()

    # Convert to DataFrame and save to csv
    basic_data_center_info = pd.DataFrame(master_list)
    basic_data_center_info.to_csv(output_path, index=False)
    logger.info("Fetched data centers from page %s to  %s and saved to  %s.", start_page, current_page, output_path)

    return basic_data_center_info
